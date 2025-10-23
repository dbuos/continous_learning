"""
Training Script for LoRA-based Continuous Learning

Key improvements:
1. Proper train/eval split (NO overlap)
2. LoRA for better stability with LiquidAI models
3. Support for multiple model types (LiquidAI, Qwen, TinyLlama, etc.)
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict

from reasoning_arena import LogicReasoningArena, ReasoningTask
from lora_learner import LoRALearner, LearningMetrics


def extract_answer_from_response(response: str, task: ReasoningTask) -> str:
    """Extract the actual answer from model response"""
    response = response.strip()

    # Remove common prefixes
    prefixes = ["answer:", "answer is:", "the answer is:", "it is:", "it's:", "so,", "therefore,"]
    for prefix in prefixes:
        if response.lower().startswith(prefix):
            response = response[len(prefix):].strip()

    # Take first line or sentence
    if "\n" in response:
        response = response.split("\n")[0].strip()

    # Take first sentence for long responses
    sentences = response.split(".")
    if len(sentences) > 0 and len(sentences[0]) < 100:
        response = sentences[0].strip()

    return response


def check_answer_correctness(predicted: str, correct: str, task_type: str) -> bool:
    """Check if predicted answer matches correct answer"""
    predicted = predicted.lower().strip()
    correct = correct.lower().strip()

    # Exact match
    if predicted == correct:
        return True

    # Contains match
    if correct in predicted:
        return True

    # For numbers, extract and compare digits
    if task_type in ['sequence', 'math_word']:
        import re
        pred_nums = re.findall(r'\d+', predicted)
        correct_nums = re.findall(r'\d+', correct)
        if pred_nums and correct_nums and pred_nums[0] == correct_nums[0]:
            return True

    # For names, check if name appears
    if task_type in ['comparison', 'causal']:
        # Check if the correct name/word is in the prediction
        correct_words = correct.split()
        for word in correct_words:
            if len(word) > 2 and word in predicted:
                return True

    return False


def run_lora_training(
    model_name: str,
    n_train_episodes: int = 100,
    n_eval_tasks: int = 30,
    eval_interval: int = 10,
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "./lora_checkpoints",
    seed: int = 42,
    use_fp16: bool = False,
):
    """
    Run continuous learning experiment with LoRA

    Args:
        model_name: HuggingFace model name
        n_train_episodes: Number of training episodes
        n_eval_tasks: Number of evaluation tasks (held-out, never trained on)
        eval_interval: Evaluate every N episodes
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
        seed: Random seed
        use_fp16: Use mixed precision (FP16)
    """

    print("=" * 70)
    print("CONTINUOUS LEARNING WITH LoRA")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Train Episodes: {n_train_episodes}")
    print(f"  Eval Tasks: {n_eval_tasks}")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Eval Interval: {eval_interval}")
    print(f"  FP16: {use_fp16}")
    print(f"  Seed: {seed}")
    print()

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Initialize arena
    print("Initializing Logic Reasoning Arena...")
    arena = LogicReasoningArena(seed=seed)

    # IMPORTANT: Generate SEPARATE train and eval sets (NO OVERLAP!)
    print("\nGenerating datasets with proper train/eval split...")

    # Generate a large pool of tasks
    all_tasks = arena.generate_batch(n=n_train_episodes + n_eval_tasks + 50, mix=True)

    # Split into train and eval (NO OVERLAP)
    eval_tasks = all_tasks[:n_eval_tasks]
    train_tasks_pool = all_tasks[n_eval_tasks:]

    print(f"✓ Eval set: {len(eval_tasks)} tasks (held-out, never seen during training)")
    print(f"✓ Train pool: {len(train_tasks_pool)} tasks")

    # Show split statistics
    print("\nDataset statistics:")
    print("Eval set breakdown:")
    eval_types = {}
    for task in eval_tasks:
        eval_types[task.task_type] = eval_types.get(task.task_type, 0) + 1
    for task_type, count in sorted(eval_types.items()):
        print(f"  {task_type}: {count}")

    print("\nTrain pool breakdown:")
    train_types = {}
    for task in train_tasks_pool:
        train_types[task.task_type] = train_types.get(task.task_type, 0) + 1
    for task_type, count in sorted(train_types.items()):
        print(f"  {task_type}: {count}")

    # Initialize LoRA agent
    print("\nInitializing LoRA Continuous Learning Agent...")
    agent = LoRALearner(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        use_fp16=use_fp16,
    )

    # Initial evaluation (before any training)
    print("\n" + "=" * 70)
    print("INITIAL EVALUATION (Before Learning)")
    print("=" * 70)

    initial_eval_data = []
    for i, task in enumerate(eval_tasks[:5]):  # Show first 5
        response = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
        answer = extract_answer_from_response(response, task)
        is_correct = check_answer_correctness(answer, task.answer, task.task_type)

        initial_eval_data.append({
            "question": task.question,
            "predicted": answer,
            "correct": task.answer,
            "is_correct": is_correct,
            "task_type": task.task_type,
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} [{task.task_type}] Q: {task.question[:60]}...")
        print(f"  Predicted: {answer[:50]} | Correct: {task.answer}")

    # Full initial evaluation
    initial_correct = 0
    for task in eval_tasks:
        response = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
        answer = extract_answer_from_response(response, task)
        if check_answer_correctness(answer, task.answer, task.task_type):
            initial_correct += 1

    initial_acc = initial_correct / len(eval_tasks)
    print(f"\nInitial Accuracy: {initial_acc:.2%} ({initial_correct}/{len(eval_tasks)})")

    # Continuous learning loop
    print("\n" + "=" * 70)
    print("STARTING CONTINUOUS LEARNING")
    print("=" * 70)

    training_results = []
    eval_results = []
    task_type_accuracy = {tt: [] for tt in ['comparison', 'sequence', 'causal', 'math_word', 'logic_grid']}

    start_time = time.time()

    for episode in range(1, n_train_episodes + 1):
        # Sample from train pool (can repeat tasks, simulating continuous stream)
        task_idx = (episode - 1) % len(train_tasks_pool)
        task = train_tasks_pool[task_idx]

        # Get prediction before learning
        response_before = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
        answer_before = extract_answer_from_response(response_before, task)
        correct_before = check_answer_correctness(answer_before, task.answer, task.task_type)

        # Learn from feedback (gradient descent step)
        result = agent.learn_from_feedback(
            question=task.question,
            correct_answer=task.answer,
            task_type=task.task_type,
        )

        # Get prediction after learning
        response_after = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
        answer_after = extract_answer_from_response(response_after, task)
        correct_after = check_answer_correctness(answer_after, task.answer, task.task_type)

        # Track results
        training_results.append({
            "episode": episode,
            "task_type": task.task_type,
            "question": task.question,
            "correct_answer": task.answer,
            "answer_before": answer_before,
            "answer_after": answer_after,
            "correct_before": correct_before,
            "correct_after": correct_after,
            "loss": result["loss"],
        })

        # Print progress
        if episode % 5 == 0 or episode == 1:
            status_before = "✓" if correct_before else "✗"
            status_after = "✓" if correct_after else "✗"
            print(f"\nEpisode {episode}/{n_train_episodes}")
            print(f"  Task: [{task.task_type}] {task.question[:50]}...")
            print(f"  Before: {status_before} {answer_before[:30]}")
            print(f"  After:  {status_after} {answer_after[:30]}")
            print(f"  Correct: {task.answer}")
            print(f"  Loss: {result['loss']:.4f}")

        # Periodic evaluation on HELD-OUT eval set
        if episode % eval_interval == 0:
            print(f"\n{'=' * 70}")
            print(f"EVALUATION at Episode {episode} (on held-out eval set)")
            print(f"{'=' * 70}")

            # Evaluate on eval set (never trained on!)
            correct_count = 0
            eval_data = []

            for task in eval_tasks:
                response = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
                answer = extract_answer_from_response(response, task)
                is_correct = check_answer_correctness(answer, task.answer, task.task_type)

                if is_correct:
                    correct_count += 1

                eval_data.append({
                    "question": task.question,
                    "predicted": answer,
                    "correct": task.answer,
                    "is_correct": is_correct,
                    "task_type": task.task_type,
                })

                # Track by task type
                task_type_accuracy[task.task_type].append(is_correct)

            accuracy = correct_count / len(eval_tasks)
            eval_results.append({
                "episode": episode,
                "accuracy": accuracy,
                "correct": correct_count,
                "total": len(eval_tasks),
            })

            print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(eval_tasks)})")

            # Per-task-type accuracy
            print("\nPer-task-type accuracy:")
            for task_type, results in task_type_accuracy.items():
                if results:
                    recent_results = results[-(len(eval_tasks)//5 or 1):]
                    acc = sum(recent_results) / len(recent_results) if recent_results else 0
                    print(f"  {task_type}: {acc:.2%}")

            # Save checkpoint
            checkpoint_path = f"{checkpoint_dir}/checkpoint_ep{episode}"
            agent.save_checkpoint(checkpoint_path)

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total Episodes: {n_train_episodes}")
    print(f"Time Elapsed: {elapsed_time:.2f}s")
    print(f"Episodes/sec: {n_train_episodes / elapsed_time:.2f}")

    # Final evaluation
    print(f"\n{'=' * 70}")
    print(f"FINAL EVALUATION (on held-out eval set)")
    print(f"{'=' * 70}")

    final_eval_data = []
    correct_count = 0

    for task in eval_tasks:
        response = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
        answer = extract_answer_from_response(response, task)
        is_correct = check_answer_correctness(answer, task.answer, task.task_type)

        if is_correct:
            correct_count += 1

        final_eval_data.append({
            "question": task.question,
            "predicted": answer,
            "correct": task.answer,
            "is_correct": is_correct,
            "task_type": task.task_type,
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} [{task.task_type}]")
        print(f"  Q: {task.question[:70]}...")
        print(f"  Predicted: {answer[:50]} | Correct: {task.answer}")

    final_accuracy = correct_count / len(eval_tasks)
    print(f"\nFinal Accuracy: {final_accuracy:.2%} ({correct_count}/{len(eval_tasks)})")
    print(f"Improvement: {final_accuracy - initial_acc:+.2%}")

    # Save results
    results = {
        "config": {
            "model_name": model_name,
            "n_train_episodes": n_train_episodes,
            "n_eval_tasks": n_eval_tasks,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "use_fp16": use_fp16,
            "seed": seed,
        },
        "initial_accuracy": initial_acc,
        "final_accuracy": final_accuracy,
        "improvement": final_accuracy - initial_acc,
        "training_results": training_results,
        "eval_results": eval_results,
        "final_eval_data": final_eval_data,
    }

    results_path = f"{checkpoint_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Plot learning curves
    plot_learning_curves(eval_results, training_results, checkpoint_dir)

    return results


def plot_learning_curves(eval_results, training_results, save_dir):
    """Plot and save learning curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Evaluation accuracy over time
    if eval_results:
        episodes = [r["episode"] for r in eval_results]
        accuracies = [r["accuracy"] for r in eval_results]

        axes[0].plot(episodes, accuracies, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        axes[0].set_xlabel("Episode", fontsize=12)
        axes[0].set_ylabel("Accuracy (Held-Out Eval Set)", fontsize=12)
        axes[0].set_title("Evaluation Accuracy Over Time", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

    # Plot 2: Training loss over time
    if training_results:
        episodes = [r["episode"] for r in training_results]
        losses = [r["loss"] for r in training_results if not np.isnan(r["loss"])]
        valid_episodes = [r["episode"] for r in training_results if not np.isnan(r["loss"])]

        # Smooth with moving average
        if len(losses) > 10:
            window = min(10, len(losses) // 5)
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_episodes = valid_episodes[window-1:]
            axes[1].plot(smoothed_episodes, smoothed_losses, linewidth=2, label='Smoothed', color='#e74c3c')

        axes[1].plot(valid_episodes, losses, alpha=0.3, linewidth=1, label='Raw', color='#e67e22')
        axes[1].set_xlabel("Episode", fontsize=12)
        axes[1].set_ylabel("Training Loss", fontsize=12)
        axes[1].set_title("Training Loss Over Time", fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{save_dir}/learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Learning curves saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Continuous Learning with LoRA")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2-350M-Math",
                        help="Model name from HuggingFace")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--eval-tasks", type=int, default=30,
                        help="Number of held-out evaluation tasks")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N episodes")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./lora_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision")

    args = parser.parse_args()

    run_lora_training(
        model_name=args.model,
        n_train_episodes=args.episodes,
        n_eval_tasks=args.eval_tasks,
        eval_interval=args.eval_interval,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
