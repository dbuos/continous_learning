"""
Main Training Script for Continuous Learning System

This script orchestrates:
1. Logic Reasoning Arena (environment)
2. Continuous Learning Agent (with soft prompting)
3. Real-time gradient descent learning
4. Metrics tracking and visualization
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict

from reasoning_arena import LogicReasoningArena, ReasoningTask
from continuous_learner_v2 import ContinuousLearningAgentV2 as ContinuousLearningAgent, LearningMetrics


def extract_answer_from_response(response: str, task: ReasoningTask) -> str:
    """
    Extract the actual answer from model response
    Handles various response formats
    """
    # Clean up response
    response = response.strip()

    # Remove common prefixes
    prefixes = ["answer:", "answer is:", "the answer is:", "it is:", "it's:"]
    for prefix in prefixes:
        if response.lower().startswith(prefix):
            response = response[len(prefix):].strip()

    # Take first line or sentence
    if "\n" in response:
        response = response.split("\n")[0].strip()

    if "." in response:
        response = response.split(".")[0].strip()

    return response


def check_answer_correctness(predicted: str, correct: str, task_type: str) -> bool:
    """
    Check if predicted answer matches correct answer
    Uses flexible matching for different task types
    """
    predicted = predicted.lower().strip()
    correct = correct.lower().strip()

    # Exact match
    if predicted == correct:
        return True

    # Contains match
    if correct in predicted:
        return True

    # For numbers, extract digits
    if task_type in ['sequence', 'math_word']:
        import re
        pred_nums = re.findall(r'\d+', predicted)
        correct_nums = re.findall(r'\d+', correct)
        if pred_nums and correct_nums and pred_nums[0] == correct_nums[0]:
            return True

    return False


def run_continuous_learning(
    model_name: str,
    n_episodes: int = 100,
    eval_interval: int = 10,
    n_soft_tokens: int = 10,
    learning_rate: float = 0.01,
    checkpoint_dir: str = "./checkpoints",
    seed: int = 42,
):
    """
    Run continuous learning experiment

    Args:
        model_name: HuggingFace model name
        n_episodes: Number of learning episodes
        eval_interval: Evaluate every N episodes
        n_soft_tokens: Number of soft prompt tokens
        learning_rate: Learning rate for gradient descent
        checkpoint_dir: Directory to save checkpoints
        seed: Random seed
    """

    print("=" * 70)
    print("CONTINUOUS LEARNING FOR AGENTIC LLMs")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Soft Tokens: {n_soft_tokens}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Eval Interval: {eval_interval}")
    print(f"  Seed: {seed}")
    print()

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Initialize arena and agent
    print("Initializing Logic Reasoning Arena...")
    arena = LogicReasoningArena(seed=seed)

    print("\nInitializing Continuous Learning Agent...")
    agent = ContinuousLearningAgent(
        model_name=model_name,
        n_soft_tokens=n_soft_tokens,
        learning_rate=learning_rate,
    )

    # Generate evaluation set (held out, not used for training)
    print("\nGenerating evaluation set...")
    eval_tasks = arena.generate_batch(n=20, mix=True)
    print(f"✓ Created {len(eval_tasks)} evaluation tasks")

    # Tracking
    training_results = []
    eval_results = []
    task_type_accuracy = {tt: [] for tt in ['comparison', 'sequence', 'causal', 'math_word', 'logic_grid']}

    # Initial evaluation
    print("\n" + "=" * 70)
    print("INITIAL EVALUATION (Before Learning)")
    print("=" * 70)
    eval_data = []
    for task in eval_tasks[:5]:  # Show first 5
        response = agent.generate_response(task.question, max_new_tokens=30, temperature=0.3)
        answer = extract_answer_from_response(response, task)
        is_correct = check_answer_correctness(answer, task.answer, task.task_type)

        eval_data.append({
            "question": task.question,
            "predicted": answer,
            "correct": task.answer,
            "is_correct": is_correct,
            "task_type": task.task_type,
        })

        status = "✓" if is_correct else "✗"
        print(f"{status} [{task.task_type}] Q: {task.question[:60]}...")
        print(f"  Predicted: {answer} | Correct: {task.answer}")

    initial_acc = sum(d["is_correct"] for d in eval_data) / len(eval_data)
    print(f"\nInitial Accuracy: {initial_acc:.2%}")

    # Continuous learning loop
    print("\n" + "=" * 70)
    print("STARTING CONTINUOUS LEARNING")
    print("=" * 70)

    start_time = time.time()

    for episode in range(1, n_episodes + 1):
        # Generate a new task
        task = arena.generate_task()

        # Get prediction before learning
        response_before = agent.generate_response(
            task.question,
            max_new_tokens=30,
            temperature=0.3,
        )
        answer_before = extract_answer_from_response(response_before, task)
        correct_before = check_answer_correctness(answer_before, task.answer, task.task_type)

        # Learn from feedback (gradient descent step)
        result = agent.learn_from_feedback(
            question=task.question,
            correct_answer=task.answer,
            task_type=task.task_type,
        )

        # Get prediction after learning
        response_after = agent.generate_response(
            task.question,
            max_new_tokens=30,
            temperature=0.3,
        )
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
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"  Task: [{task.task_type}] {task.question[:50]}...")
            print(f"  Before: {status_before} {answer_before[:30]}")
            print(f"  After:  {status_after} {answer_after[:30]}")
            print(f"  Correct: {task.answer}")
            print(f"  Loss: {result['loss']:.4f}")

        # Periodic evaluation
        if episode % eval_interval == 0:
            print(f"\n{'=' * 70}")
            print(f"EVALUATION at Episode {episode}")
            print(f"{'=' * 70}")

            # Evaluate on held-out set
            correct_count = 0
            eval_data = []

            for task in eval_tasks:
                response = agent.generate_response(
                    task.question,
                    max_new_tokens=30,
                    temperature=0.3,
                )
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
                    acc = sum(results[-len(eval_tasks)//5:]) / min(len(results), len(eval_tasks)//5)
                    print(f"  {task_type}: {acc:.2%}")

            # Save checkpoint
            checkpoint_path = f"{checkpoint_dir}/checkpoint_ep{episode}.pt"
            agent.save_checkpoint(checkpoint_path)

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total Episodes: {n_episodes}")
    print(f"Time Elapsed: {elapsed_time:.2f}s")
    print(f"Episodes/sec: {n_episodes / elapsed_time:.2f}")

    # Final evaluation
    print(f"\n{'=' * 70}")
    print(f"FINAL EVALUATION")
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
        print(f"  Predicted: {answer} | Correct: {task.answer}")

    final_accuracy = correct_count / len(eval_tasks)
    print(f"\nFinal Accuracy: {final_accuracy:.2%} ({correct_count}/{len(eval_tasks)})")
    print(f"Improvement: {final_accuracy - initial_acc:+.2%}")

    # Save results
    results = {
        "config": {
            "model_name": model_name,
            "n_episodes": n_episodes,
            "n_soft_tokens": n_soft_tokens,
            "learning_rate": learning_rate,
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

        axes[0].plot(episodes, accuracies, marker='o', linewidth=2, markersize=6)
        axes[0].set_xlabel("Episode", fontsize=12)
        axes[0].set_ylabel("Accuracy", fontsize=12)
        axes[0].set_title("Evaluation Accuracy Over Time", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

    # Plot 2: Training loss over time
    if training_results:
        episodes = [r["episode"] for r in training_results]
        losses = [r["loss"] for r in training_results]

        # Smooth with moving average
        window = min(10, len(losses) // 5)
        if window > 0:
            smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_episodes = episodes[window-1:]
            axes[1].plot(smoothed_episodes, smoothed_losses, linewidth=2, label='Smoothed')

        axes[1].plot(episodes, losses, alpha=0.3, linewidth=1, label='Raw')
        axes[1].set_xlabel("Episode", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_title("Training Loss Over Time", fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{save_dir}/learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Learning curves saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Continuous Learning for Agentic LLMs")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2-350M-Math",
                        help="Model name from HuggingFace")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate every N episodes")
    parser.add_argument("--soft-tokens", type=int, default=10,
                        help="Number of soft prompt tokens")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    run_continuous_learning(
        model_name=args.model,
        n_episodes=args.episodes,
        eval_interval=args.eval_interval,
        n_soft_tokens=args.soft_tokens,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
