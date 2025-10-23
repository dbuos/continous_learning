"""
Continuous Learning with LoRA (Low-Rank Adaptation)

This implementation uses LoRA for parameter-efficient fine-tuning,
which is more stable than soft prompting for certain models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Tuple
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


class LoRALearner:
    """
    Continuous learning agent using LoRA

    LoRA adds trainable low-rank matrices to attention layers,
    enabling efficient adaptation while keeping base model frozen.
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,  # Rank of LoRA matrices
        lora_alpha: int = 16,  # Scaling factor
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = False,  # Use mixed precision
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.use_fp16 = use_fp16

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        dtype = torch.float16 if use_fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=self.device,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],  # Apply to attention layers
            bias="none",
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Optimizer - only LoRA parameters are trainable
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Metrics
        self.metrics_history: List[LearningMetrics] = []
        self.episode_count = 0

        print(f"✓ Model loaded successfully")

    def generate_response(
        self,
        question: str,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
    ) -> str:
        """Generate response to question"""

        prompt = f"Question: {question}\nAnswer:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate with model in eval mode
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
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

        Uses teacher forcing: given the full QA pair, predict next tokens
        """

        # Create full sequence
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

        # Set model to training mode
        self.model.train()

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # Labels are same as input for causal LM
        )

        loss = outputs.loss

        # Get prediction for logging
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

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected ({loss.item()}), skipping update")
            return {
                "episode": self.episode_count,
                "loss": float('nan'),
                "predicted": predicted_answer,
                "correct": correct_answer,
                "is_correct": False,
                "task_type": task_type,
            }

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0
        )

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
        """Save LoRA checkpoint"""
        # Save LoRA weights only
        self.model.save_pretrained(path)

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "metrics_history": self.metrics_history,
        }, f"{path}/training_state.pt")

        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load LoRA checkpoint"""
        # Load LoRA weights
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)

        # Load optimizer state
        state = torch.load(f"{path}/training_state.pt")
        self.optimizer.load_state_dict(state["optimizer"])
        self.episode_count = state["episode_count"]
        self.metrics_history = state["metrics_history"]

        print(f"✓ Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test the LoRA learner
    print("=== LoRA Learner Test ===\n")

    agent = LoRALearner(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16,
        learning_rate=1e-4,
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

    # Learn (3 steps)
    print("\n--- Learning (3 steps) ---")
    for i in range(3):
        result = agent.learn_from_feedback(question, answer)
        print(f"Step {i+1}: Loss={result['loss']:.4f}, Predicted={result['predicted'][:50]}")

    # After learning
    print("\n--- After Learning ---")
    response = agent.generate_response(question)
    print(f"Response: {response}")
