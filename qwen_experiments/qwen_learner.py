"""
Continuous Learning System for Qwen Models

Adapted soft prompting system specifically designed for Qwen2.5 models.
Key differences from base implementation:
- Support for Qwen's chat template
- Optimized for Qwen's architecture
- Better handling of Qwen's tokenizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
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
    Learnable continuous prompt embeddings
    """

    def __init__(self, n_tokens: int, embed_dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

        # Initialize with small random values
        self.soft_prompt = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.1)

    def forward(self, embedded_inputs: torch.Tensor) -> torch.Tensor:
        """Prepend soft prompt to input embeddings"""
        batch_size = embedded_inputs.shape[0]
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([soft_prompt_batch, embedded_inputs], dim=1)


class QwenContinuousLearner:
    """
    Continuous learning agent optimized for Qwen models
    """

    def __init__(
        self,
        model_name: str,
        n_soft_tokens: int = 8,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_chat_template: bool = True,
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.use_chat_template = use_chat_template

        print(f"Loading Qwen model: {model_name}")
        print(f"Device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=self.device,
        )

        self.embed_dim = self.model.config.hidden_size

        # Initialize soft prompt learner
        self.soft_prompt_learner = SoftPromptLearner(
            n_tokens=n_soft_tokens,
            embed_dim=self.embed_dim,
        ).to(self.device)

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Only soft prompts trainable
        for param in self.soft_prompt_learner.parameters():
            param.requires_grad = True

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.soft_prompt_learner.parameters(),
            lr=learning_rate,
        )

        # Metrics
        self.metrics_history: List[LearningMetrics] = []
        self.episode_count = 0

        print(f"✓ Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params")
        print(f"✓ Soft prompt: {n_soft_tokens} tokens, {sum(p.numel() for p in self.soft_prompt_learner.parameters())} trainable")
        print(f"✓ Embedding dim: {self.embed_dim}")

    def _format_prompt(self, question: str) -> str:
        """Format prompt with optional chat template"""
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            # Use Qwen's chat template
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions accurately and concisely."},
                {"role": "user", "content": question}
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback if chat template fails
                prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        return prompt

    def generate_response(
        self,
        question: str,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        use_soft_prompt: bool = True,
    ) -> str:
        """Generate response to question"""

        prompt = self._format_prompt(question)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            input_embeds = self.model.get_input_embeddings()(input_ids)

        if use_soft_prompt:
            input_embeds = self.soft_prompt_learner(input_embeds)
            soft_mask = torch.ones(
                (attention_mask.shape[0], self.soft_prompt_learner.n_tokens),
                dtype=attention_mask.dtype,
                device=self.device,
            )
            attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer based on format
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        elif "assistant" in generated_text.lower():
            # For chat template format
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                answer = parts[-1].strip()
            else:
                answer = generated_text.strip()
        else:
            answer = generated_text.strip()

        return answer

    def compute_loss_simple(
        self,
        question: str,
        correct_answer: str,
    ) -> Tuple[torch.Tensor, str]:
        """
        Simplified loss computation using teacher forcing
        """

        # Create full input-output pair
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions accurately and concisely."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": correct_answer}
            ]
            try:
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except:
                full_text = f"Question: {question}\nAnswer: {correct_answer}"
        else:
            full_text = f"Question: {question}\nAnswer: {correct_answer}"

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Apply soft prompt
        input_embeds = self.soft_prompt_learner(input_embeds)

        # Adjust attention mask for soft prompt
        n_soft = self.soft_prompt_learner.n_tokens
        soft_mask = torch.ones((attention_mask.shape[0], n_soft), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat([soft_mask, attention_mask], dim=1)

        # Forward pass
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        logits = outputs.logits  # [batch, seq_len + n_soft, vocab_size]

        # Create labels - shift to predict next token
        # Remove soft prompt tokens from logits and shift
        logits_for_loss = logits[:, n_soft:-1, :]  # Remove soft prompt and last token
        labels_for_loss = input_ids[:, 1:]  # Shift labels

        # Ensure same length
        min_len = min(logits_for_loss.shape[1], labels_for_loss.shape[1])
        logits_for_loss = logits_for_loss[:, :min_len, :]
        labels_for_loss = labels_for_loss[:, :min_len]

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
            labels_for_loss.reshape(-1),
            reduction='mean',
        )

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
        """Learn from feedback using gradient descent"""

        self.episode_count += 1

        # Compute loss
        loss, predicted_answer = self.compute_loss_simple(question, correct_answer)

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            return {
                "episode": self.episode_count,
                "loss": float('nan'),
                "predicted": predicted_answer,
                "correct": correct_answer,
                "is_correct": False,
                "task_type": task_type,
            }

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_prompt_learner.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Check correctness
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

    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            "soft_prompt": self.soft_prompt_learner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "metrics_history": self.metrics_history,
        }, path)
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.soft_prompt_learner.load_state_dict(checkpoint["soft_prompt"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.episode_count = checkpoint["episode_count"]
        self.metrics_history = checkpoint["metrics_history"]
        print(f"✓ Checkpoint loaded from {path}")
