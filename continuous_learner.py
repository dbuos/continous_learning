"""
Continuous Learning System for Agentic LLMs

This implements continuous learning using:
- Soft Prompting: Learnable prompt embeddings
- Gradient Descent: Updates from feedback signals
- Real-time Learning: Adapts during inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
import time


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress"""
    episode: int
    loss: float
    accuracy: float
    task_type: str
    timestamp: float


class SoftPromptLearner(nn.Module):
    """
    Implements soft prompting - learnable continuous prompt embeddings
    that are prepended to the input and optimized via gradient descent
    """

    def __init__(
        self,
        n_tokens: int = 10,
        embed_dim: int = 768,
        initialize_from_vocab: bool = True,
        model=None,
        tokenizer=None,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

        # Initialize soft prompt embeddings
        if initialize_from_vocab and model is not None and tokenizer is not None:
            # Initialize from actual token embeddings for better starting point
            init_text = "Solve this step by step. Think carefully."
            init_ids = tokenizer.encode(init_text, add_special_tokens=False)[:n_tokens]
            while len(init_ids) < n_tokens:
                init_ids.append(tokenizer.pad_token_id or 0)
            init_ids = init_ids[:n_tokens]

            # Get embeddings from model
            with torch.no_grad():
                embedding_weights = model.get_input_embeddings().weight
                init_embeddings = embedding_weights[init_ids].clone()

            self.soft_prompt = nn.Parameter(init_embeddings)
        else:
            # Random initialization
            self.soft_prompt = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.01)

    def forward(self, embedded_inputs: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompt to input embeddings

        Args:
            embedded_inputs: [batch_size, seq_len, embed_dim]

        Returns:
            [batch_size, n_tokens + seq_len, embed_dim]
        """
        batch_size = embedded_inputs.shape[0]

        # Expand soft prompt for batch
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate soft prompt with input
        return torch.cat([soft_prompt_batch, embedded_inputs], dim=1)


class ContinuousLearningAgent:
    """
    Agent that learns continuously from feedback using gradient descent
    """

    def __init__(
        self,
        model_name: str,
        n_soft_tokens: int = 10,
        learning_rate: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.learning_rate = learning_rate

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use float32 for better training stability
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=self.device,
        )

        # Get embedding dimension
        self.embed_dim = self.model.config.hidden_size

        # Initialize soft prompt learner
        self.soft_prompt_learner = SoftPromptLearner(
            n_tokens=n_soft_tokens,
            embed_dim=self.embed_dim,
            initialize_from_vocab=True,
            model=self.model,
            tokenizer=self.tokenizer,
        ).to(self.device)

        # Freeze base model, only train soft prompts
        for param in self.model.parameters():
            param.requires_grad = False

        # Only soft prompts are trainable
        for param in self.soft_prompt_learner.parameters():
            param.requires_grad = True

        # Optimizer for soft prompts with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.soft_prompt_learner.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-6,  # Avoid division by zero
        )

        # Metrics tracking
        self.metrics_history: List[LearningMetrics] = []
        self.episode_count = 0

        print(f"✓ Model loaded with {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
        print(f"✓ Soft prompt: {n_soft_tokens} tokens, {sum(p.numel() for p in self.soft_prompt_learner.parameters())} trainable parameters")

    def generate_response(
        self,
        question: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        use_soft_prompt: bool = True,
    ) -> str:
        """Generate a response to the question"""

        # Format input
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Get input embeddings
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            input_embeds = self.model.get_input_embeddings()(input_ids)

        # Apply soft prompt if enabled
        if use_soft_prompt:
            input_embeds = self.soft_prompt_learner(input_embeds)

            # Adjust attention mask for soft prompt tokens
            soft_prompt_mask = torch.ones(
                (attention_mask.shape[0], self.soft_prompt_learner.n_tokens),
                dtype=attention_mask.dtype,
                device=self.device,
            )
            attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (remove the prompt)
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()

        return answer

    def compute_loss(
        self,
        question: str,
        correct_answer: str,
    ) -> Tuple[torch.Tensor, str]:
        """
        Compute loss for a question-answer pair

        We use the language modeling loss on the answer tokens
        """

        # Format as "Question: {q}\nAnswer: {a}"
        full_text = f"Question: {question}\nAnswer: {correct_answer}"

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Apply soft prompt
        input_embeds = self.soft_prompt_learner(input_embeds)

        # Adjust attention mask
        soft_prompt_mask = torch.ones(
            (attention_mask.shape[0], self.soft_prompt_learner.n_tokens),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Create labels (we only compute loss on answer tokens)
        # Shift labels to align with model outputs
        labels = input_ids.clone()

        # Find where "Answer:" starts
        answer_text = f"\nAnswer: {correct_answer}"
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

        # Mask out everything except answer tokens in labels
        labels[:] = -100  # Ignore index

        # Find answer position in input_ids
        for i in range(len(input_ids[0]) - len(answer_ids) + 1):
            if input_ids[0, i:i + len(answer_ids)].tolist() == answer_ids:
                labels[0, i:i + len(answer_ids)] = input_ids[0, i:i + len(answer_ids)]
                break

        # Adjust labels for soft prompt (shift right)
        n_soft = self.soft_prompt_learner.n_tokens
        labels_adjusted = torch.full(
            (labels.shape[0], labels.shape[1] + n_soft),
            -100,
            dtype=labels.dtype,
            device=self.device,
        )
        labels_adjusted[:, n_soft:] = labels

        # Forward pass
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels_adjusted,
        )

        loss = outputs.loss

        # Generate prediction for logging
        with torch.no_grad():
            predicted_answer = self.generate_response(question, max_new_tokens=20, temperature=0.1)

        return loss, predicted_answer

    def learn_from_feedback(
        self,
        question: str,
        correct_answer: str,
        task_type: str = "general",
    ) -> Dict:
        """
        Learn from a single question-answer pair using gradient descent

        Returns metrics dict
        """

        self.episode_count += 1

        # Compute loss
        loss, predicted_answer = self.compute_loss(question, correct_answer)

        # Gradient descent step (only if loss is valid)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected ({loss.item()}), skipping update")
        else:
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.soft_prompt_learner.parameters(), max_norm=0.5)

            self.optimizer.step()

        # Check if answer is correct (simple substring match)
        is_correct = correct_answer.lower() in predicted_answer.lower()

        # Log metrics
        metrics = LearningMetrics(
            episode=self.episode_count,
            loss=loss.item(),
            accuracy=1.0 if is_correct else 0.0,
            task_type=task_type,
            timestamp=time.time(),
        )
        self.metrics_history.append(metrics)

        return {
            "episode": self.episode_count,
            "loss": loss.item(),
            "predicted": predicted_answer,
            "correct": correct_answer,
            "is_correct": is_correct,
            "task_type": task_type,
        }

    def evaluate(self, questions: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate on a list of question-answer pairs without learning

        Args:
            questions: List of (question, answer) tuples

        Returns:
            Dict with evaluation metrics
        """

        correct = 0
        total = len(questions)
        predictions = []

        for question, correct_answer in questions:
            predicted = self.generate_response(question, max_new_tokens=20, temperature=0.1)
            is_correct = correct_answer.lower() in predicted.lower()

            if is_correct:
                correct += 1

            predictions.append({
                "question": question,
                "predicted": predicted,
                "correct": correct_answer,
                "is_correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions,
        }

    def save_checkpoint(self, path: str):
        """Save soft prompt checkpoint"""
        torch.save({
            "soft_prompt": self.soft_prompt_learner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "metrics_history": self.metrics_history,
        }, path)
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load soft prompt checkpoint"""
        checkpoint = torch.load(path)
        self.soft_prompt_learner.load_state_dict(checkpoint["soft_prompt"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.episode_count = checkpoint["episode_count"]
        self.metrics_history = checkpoint["metrics_history"]
        print(f"✓ Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test the continuous learner
    print("=== Continuous Learning Agent Test ===\n")

    # Use a small model for testing
    agent = ContinuousLearningAgent(
        model_name="gpt2",  # Small model for quick testing
        n_soft_tokens=5,
        learning_rate=0.1,
    )

    # Test question
    question = "What is 2 + 2?"
    answer = "4"

    print(f"\nQuestion: {question}")
    print(f"Correct Answer: {answer}")

    # Before learning
    print("\n--- Before Learning ---")
    response = agent.generate_response(question)
    print(f"Response: {response}")

    # Learn
    print("\n--- Learning (3 steps) ---")
    for i in range(3):
        result = agent.learn_from_feedback(question, answer)
        print(f"Step {i+1}: Loss={result['loss']:.4f}, Predicted={result['predicted']}")

    # After learning
    print("\n--- After Learning ---")
    response = agent.generate_response(question)
    print(f"Response: {response}")
