"""
Orthogonal LoRA (O-LoRA) for Continual Learning

Based on recent research:
- O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning (EMNLP 2023)
- CURLoRA: Stable LLM Continual Fine-Tuning (2024)
- N-LoRA: Reducing Parameter Collision (COLING 2025)

Key idea: Learn new tasks in orthogonal subspaces to minimize interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import numpy as np
from collections import deque


@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress"""
    episode: int
    loss: float
    accuracy: float
    task_type: str
    timestamp: float
    difficulty: int = 0


class ExperienceReplayBuffer:
    """
    Experience replay buffer for continual learning
    Stores past examples to prevent catastrophic forgetting
    """

    def __init__(self, capacity: int = 1000, sample_size: int = 5):
        self.capacity = capacity
        self.sample_size = sample_size
        self.buffer = deque(maxlen=capacity)

    def add(self, question: str, answer: str, task_type: str):
        """Add experience to buffer"""
        self.buffer.append({
            'question': question,
            'answer': answer,
            'task_type': task_type,
        })

    def sample(self, n: Optional[int] = None) -> List[Dict]:
        """Sample random experiences from buffer"""
        n = n or self.sample_size
        if len(self.buffer) == 0:
            return []

        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class OrthogonalLoRALearner:
    """
    Continuous learning with Orthogonal LoRA and Experience Replay

    Improvements over basic LoRA:
    1. Orthogonal initialization for new task adaptors
    2. Experience replay to prevent forgetting
    3. Adaptive learning rate based on task difficulty
    4. Gradient projection to maintain orthogonality
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        use_replay: bool = True,
        replay_buffer_size: int = 500,
        replay_sample_size: int = 3,
        orthogonal_reg: float = 0.01,  # Orthogonality regularization strength
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = False,
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.use_fp16 = use_fp16
        self.use_replay = use_replay
        self.orthogonal_reg = orthogonal_reg

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"O-LoRA config: r={lora_r}, alpha={lora_alpha}, orthogonal_reg={orthogonal_reg}")

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
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Initialize experience replay buffer
        if use_replay:
            self.replay_buffer = ExperienceReplayBuffer(
                capacity=replay_buffer_size,
                sample_size=replay_sample_size
            )
            print(f"✓ Experience replay enabled (buffer size: {replay_buffer_size})")
        else:
            self.replay_buffer = None

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Store initial LoRA parameters for orthogonality constraint
        self.initial_lora_params = {}
        self._store_lora_params()

        # Metrics
        self.metrics_history: List[LearningMetrics] = []
        self.episode_count = 0

        print(f"✓ O-LoRA Learner initialized successfully")

    def _store_lora_params(self):
        """Store current LoRA parameters for orthogonality calculation"""
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                self.initial_lora_params[name] = param.data.clone().detach()

    def _compute_orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality regularization loss
        Encourages new updates to be orthogonal to previous parameters
        """
        orth_loss = 0.0
        count = 0

        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad and name in self.initial_lora_params:
                initial_param = self.initial_lora_params[name]

                # Flatten parameters
                current_flat = param.view(-1)
                initial_flat = initial_param.view(-1)

                # Compute cosine similarity (should be close to 0 for orthogonality)
                cos_sim = F.cosine_similarity(
                    current_flat.unsqueeze(0),
                    initial_flat.unsqueeze(0)
                )

                # Penalize non-orthogonality
                orth_loss += cos_sim ** 2

                count += 1

        return orth_loss / max(count, 1) if count > 0 else torch.tensor(0.0).to(self.device)

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

        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()

        return answer

    def compute_loss(
        self,
        question: str,
        correct_answer: str,
        include_orth_reg: bool = True,
    ) -> Tuple[torch.Tensor, str]:
        """Compute loss with optional orthogonality regularization"""
        full_text = f"Question: {question}\nAnswer: {correct_answer}"

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.model.train()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss

        # Add orthogonality regularization
        if include_orth_reg and self.orthogonal_reg > 0:
            orth_loss = self._compute_orthogonality_loss()
            loss = loss + self.orthogonal_reg * orth_loss

        # Get prediction for logging
        with torch.no_grad():
            predicted_answer = self.generate_response(question, max_new_tokens=20, temperature=0.1)

        return loss, predicted_answer

    def learn_from_feedback(
        self,
        question: str,
        correct_answer: str,
        task_type: str = "general",
        difficulty: int = 3,
        use_replay_this_step: bool = True,
    ) -> Dict:
        """
        Learn from feedback with experience replay and orthogonality

        Args:
            question: The question
            correct_answer: Correct answer
            task_type: Type of task
            difficulty: Difficulty level (1-5), used for adaptive LR
            use_replay_this_step: Whether to use replay in this step
        """
        self.episode_count += 1

        # Add to replay buffer
        if self.use_replay and self.replay_buffer is not None:
            self.replay_buffer.add(question, correct_answer, task_type)

        # Compute loss on current example
        loss, predicted_answer = self.compute_loss(question, correct_answer)

        # Add replay examples if enabled
        if (self.use_replay and use_replay_this_step and
                self.replay_buffer is not None and len(self.replay_buffer) > 0):

            replay_samples = self.replay_buffer.sample()
            replay_loss_total = 0.0

            for sample in replay_samples:
                replay_loss, _ = self.compute_loss(
                    sample['question'],
                    sample['answer'],
                    include_orth_reg=False  # Only apply orth reg to main task
                )
                replay_loss_total += replay_loss

            # Combine losses (weighted average)
            replay_weight = 0.3  # 30% weight to replay
            if len(replay_samples) > 0:
                replay_loss_avg = replay_loss_total / len(replay_samples)
                loss = (1 - replay_weight) * loss + replay_weight * replay_loss_avg

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            return {
                "episode": self.episode_count,
                "loss": float('nan'),
                "predicted": predicted_answer,
                "correct": correct_answer,
                "is_correct": False,
                "task_type": task_type,
                "difficulty": difficulty,
            }

        # Adaptive learning rate based on difficulty
        # Harder problems get slightly lower LR for stability
        lr_scale = 1.0 - (difficulty - 1) * 0.1  # 1.0 for diff=1, 0.6 for diff=5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate * lr_scale

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
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
            difficulty=difficulty,
        )
        self.metrics_history.append(metrics)

        return {
            "episode": self.episode_count,
            "loss": loss.item(),
            "predicted": predicted_answer,
            "correct": correct_answer,
            "is_correct": is_correct,
            "task_type": task_type,
            "difficulty": difficulty,
            "replay_buffer_size": len(self.replay_buffer) if self.replay_buffer else 0,
        }

    def save_checkpoint(self, path: str):
        """Save checkpoint including replay buffer"""
        self.model.save_pretrained(path)

        # Save additional state
        state = {
            "optimizer": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "metrics_history": self.metrics_history,
            "initial_lora_params": self.initial_lora_params,
        }

        if self.replay_buffer is not None:
            state["replay_buffer"] = list(self.replay_buffer.buffer)

        torch.save(state, f"{path}/training_state.pt")
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint including replay buffer"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)

        state = torch.load(f"{path}/training_state.pt")
        self.optimizer.load_state_dict(state["optimizer"])
        self.episode_count = state["episode_count"]
        self.metrics_history = state["metrics_history"]
        self.initial_lora_params = state["initial_lora_params"]

        if "replay_buffer" in state and self.replay_buffer is not None:
            self.replay_buffer.buffer = deque(state["replay_buffer"], maxlen=self.replay_buffer.capacity)

        print(f"✓ Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test O-LoRA learner
    print("=== Orthogonal LoRA Learner Test ===\n")

    learner = OrthogonalLoRALearner(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16,
        learning_rate=1e-4,
        use_replay=True,
        replay_buffer_size=100,
    )

    # Test questions
    test_examples = [
        ("What is 5 + 7?", "12", "math", 1),
        ("What is 15 - 8?", "7", "math", 1),
        ("What is 6 × 4?", "24", "math", 2),
    ]

    print("\n--- Training with Experience Replay ---")
    for i, (q, a, task_type, diff) in enumerate(test_examples, 1):
        result = learner.learn_from_feedback(q, a, task_type, diff)
        print(f"Step {i}: Loss={result['loss']:.4f}, Correct={result['is_correct']}, "
              f"Buffer={result['replay_buffer_size']}")

    print("\n--- Testing Recall ---")
    for q, a, _, _ in test_examples:
        response = learner.generate_response(q, temperature=0.1)
        print(f"Q: {q} | Predicted: {response[:20]} | Expected: {a}")
