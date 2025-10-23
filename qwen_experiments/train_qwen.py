"""
Training Script for Qwen Continuous Learning Experiments

Tests soft prompting with various Qwen2.5 models on diverse reasoning tasks.
"""

import sys
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path to import reasoning_arena
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_learner import QwenContinuousLearner
from reasoning_arena import LogicReasoningArena


def plot_learning_curves(results: dict, save_path: str):
    """Plot training and evaluation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss
    ax = axes[0, 0]
    episodes = [r["episode"] for r in results["training"]]
    losses = [r["loss"] for r in results["training"]]
    ax.plot(episodes, losses, 'b-', alpha=0.6, label='Training Loss')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Evaluation accuracy
    ax = axes[0, 1]
    eval_episodes = [r["episode"] for r in results["evaluations"]]
    eval_accs = [r["accuracy"] for r in results["evaluations"]]
    ax.plot(eval_episodes, eval_accs, 'g-', marker='o', label='Evaluation Accuracy')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title('Evaluation Accuracy Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Task type accuracy
    ax = axes[1, 0]
    if results["evaluations"]:
        last_eval = results["evaluations"][-1]
        task_types = list(last_eval["per_task_accuracy"].keys())
        accuracies = list(last_eval["per_task_accuracy"].values())

        ax.barh(task_types, accuracies, color='skyblue')
        ax.set_xlabel('Accuracy')
        ax.set_title('Final Performance by Task Type')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')

    # Cumulative accuracy
    ax = axes[1, 1]
    if len(eval_accs) > 0:
        cumulative_avg = np.cumsum(eval_accs) / np.arange(1, len(eval_accs) + 1)
        ax.plot(eval_episodes, cumulative_avg, 'r-', marker='s', label='Cumulative Avg Accuracy')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Cumulative Average Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Learning curves saved to {save_path}")


def evaluate(agent: QwenContinuousLearner, arena: LogicReasoningArena, n_tasks: int = 20) -> dict:
    """Evaluate agent on held-out tasks"""
    print(f"\n{'='*60}")
    print(f"EVALUATION ({n_tasks} tasks)")
    print(f"{'='*60}")

    task_results = {task_type: [] for task_type in arena.task_generators.keys()}
    all_correct = []

    for i in range(n_tasks):
        task = arena.generate_task()
        response = agent.generate_response(task.question, max_new_tokens=30, temperature=0.1)

        is_correct = task.answer.lower() in response.lower()
        all_correct.append(is_correct)
        task_results[task.task_type].append(is_correct)

        status = "✓" if is_correct else "✗"
        print(f"{status} [{task.task_type}] Expected: {task.answer} | Got: {response[:50]}")

    # Compute per-task-type accuracy
    per_task_accuracy = {}
    for task_type, results in task_results.items():
        if results:
            per_task_accuracy[task_type] = sum(results) / len(results)
        else:
            per_task_accuracy[task_type] = 0.0

    overall_accuracy = sum(all_correct) / len(all_correct)

    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.1%}")
    print(f"Per-task accuracy:")
    for task_type, acc in per_task_accuracy.items():
        print(f"  {task_type}: {acc:.1%}")
    print(f"{'='*60}\n")

    return {
        "overall_accuracy": overall_accuracy,
        "per_task_accuracy": per_task_accuracy,
        "n_correct": sum(all_correct),
        "n_total": len(all_correct),
    }


def train(
    model_name: str,
    n_episodes: int = 100,
    eval_interval: int = 20,
    n_soft_tokens: int = 8,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "./checkpoints",
    use_chat_template: bool = True,
):
    """Main training loop"""

    print(f"\n{'='*60}")
    print(f"QWEN CONTINUOUS LEARNING EXPERIMENT")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Eval interval: {eval_interval}")
    print(f"Soft tokens: {n_soft_tokens}")
    print(f"Learning rate: {learning_rate}")
    print(f"Chat template: {use_chat_template}")
    print(f"{'='*60}\n")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize arena and agent
    arena = LogicReasoningArena()
    agent = QwenContinuousLearner(
        model_name=model_name,
        n_soft_tokens=n_soft_tokens,
        learning_rate=learning_rate,
        use_chat_template=use_chat_template,
    )

    # Results tracking
    results = {
        "model_name": model_name,
        "n_episodes": n_episodes,
        "n_soft_tokens": n_soft_tokens,
        "learning_rate": learning_rate,
        "use_chat_template": use_chat_template,
        "training": [],
        "evaluations": [],
        "start_time": datetime.now().isoformat(),
    }

    # Initial evaluation
    print("\n" + "="*60)
    print("INITIAL EVALUATION (before any learning)")
    print("="*60)
    eval_result = evaluate(agent, arena, n_tasks=20)
    results["evaluations"].append({
        "episode": 0,
        "accuracy": eval_result["overall_accuracy"],
        "per_task_accuracy": eval_result["per_task_accuracy"],
        "n_correct": eval_result["n_correct"],
        "n_total": eval_result["n_total"],
    })

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")

    import time
    start_time = time.time()

    for episode in range(1, n_episodes + 1):
        # Generate task
        task = arena.generate_task()

        # Learn from feedback
        feedback = agent.learn_from_feedback(
            question=task.question,
            correct_answer=task.answer,
            task_type=task.task_type,
        )

        # Log
        status = "✓" if feedback["is_correct"] else "✗"
        print(f"Episode {episode:3d}/{n_episodes} | Loss: {feedback['loss']:.4f} | "
              f"{status} [{task.task_type}] Expected: {task.answer} | "
              f"Got: {feedback['predicted'][:40]}")

        results["training"].append(feedback)

        # Periodic evaluation
        if episode % eval_interval == 0:
            eval_result = evaluate(agent, arena, n_tasks=20)
            results["evaluations"].append({
                "episode": episode,
                "accuracy": eval_result["overall_accuracy"],
                "per_task_accuracy": eval_result["per_task_accuracy"],
                "n_correct": eval_result["n_correct"],
                "n_total": eval_result["n_total"],
            })

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
            agent.save_checkpoint(checkpoint_path)

    elapsed_time = time.time() - start_time
    results["elapsed_time"] = elapsed_time
    results["episodes_per_sec"] = n_episodes / elapsed_time

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_eval = evaluate(agent, arena, n_tasks=50)
    results["final_evaluation"] = {
        "accuracy": final_eval["overall_accuracy"],
        "per_task_accuracy": final_eval["per_task_accuracy"],
        "n_correct": final_eval["n_correct"],
        "n_total": final_eval["n_total"],
    }

    # Save results
    results_path = os.path.join(checkpoint_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Plot learning curves
    curves_path = os.path.join(checkpoint_dir, "learning_curves.png")
    plot_learning_curves(results, curves_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time:.1f}s ({results['episodes_per_sec']:.2f} episodes/sec)")
    print(f"Initial accuracy: {results['evaluations'][0]['accuracy']:.1%}")
    print(f"Final accuracy: {results['final_evaluation']['accuracy']:.1%}")
    improvement = results['final_evaluation']['accuracy'] - results['evaluations'][0]['accuracy']
    print(f"Improvement: {improvement:+.1%}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Qwen models with continuous learning")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Qwen model name (e.g., Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=20, help="Evaluation interval")
    parser.add_argument("--soft-tokens", type=int, default=8, help="Number of soft prompt tokens")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--no-chat-template", action="store_true", help="Disable chat template")

    args = parser.parse_args()

    train(
        model_name=args.model,
        n_episodes=args.episodes,
        eval_interval=args.eval_interval,
        n_soft_tokens=args.soft_tokens,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        use_chat_template=not args.no_chat_template,
    )


if __name__ == "__main__":
    main()
