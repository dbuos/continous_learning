"""
Comprehensive Training Script for Advanced Continuous Learning

Compares multiple approaches:
1. Baseline LoRA (from previous experiments)
2. O-LoRA (Orthogonal LoRA)
3. O-LoRA + Experience Replay
4. O-LoRA + Experience Replay + Curriculum Learning

Uses the Advanced Reasoning Arena with more challenging tasks.
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict

from advanced_reasoning_arena import AdvancedReasoningArena, ReasoningTask
from orthogonal_lora_learner import OrthogonalLoRALearner, LearningMetrics
from lora_learner import LoRALearner  # Baseline


def extract_answer(response: str, task: ReasoningTask) -> str:
    """Extract answer from response"""
    response = response.strip()

    # Remove common prefixes
    prefixes = ["answer:", "answer is:", "the answer is:", "so,", "therefore,"]
    for prefix in prefixes:
        if response.lower().startswith(prefix):
            response = response[len(prefix):].strip()

    # Take first line
    if "\n" in response:
        response = response.split("\n")[0].strip()

    return response


def check_correctness(predicted: str, correct: str, task_type: str) -> bool:
    """Check if prediction is correct"""
    predicted = predicted.lower().strip()
    correct = correct.lower().strip()

    # Exact match
    if predicted == correct:
        return True

    # Contains match
    if correct in predicted:
        return True

    # For numbers
    if task_type in ['multi_step_math', 'algebraic_reasoning', 'probability', 'geometry']:
        import re
        pred_nums = re.findall(r'\d+\.?\d*', predicted)
        correct_nums = re.findall(r'\d+\.?\d*', correct)
        if pred_nums and correct_nums:
            try:
                if float(pred_nums[0]) == float(correct_nums[0]):
                    return True
            except:
                pass

    # For fractions
    if '/' in correct:
        if correct in predicted:
            return True

    return False


def run_experiment(
    model_name: str,
    method_name: str,
    agent,
    train_tasks: List[ReasoningTask],
    eval_tasks: List[ReasoningTask],
    eval_interval: int = 20,
    checkpoint_dir: str = "./advanced_checkpoints",
):
    """Run single experiment with given agent"""

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {method_name}")
    print(f"{'=' * 70}")

    # Create checkpoint dir
    exp_dir = f"{checkpoint_dir}/{method_name.replace(' ', '_').lower()}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Initial evaluation
    print("\n--- Initial Evaluation ---")
    initial_correct = 0
    for task in eval_tasks[:5]:
        response = agent.generate_response(task.question, max_new_tokens=40, temperature=0.3)
        answer = extract_answer(response, task)
        is_correct = check_correctness(answer, task.answer, task.task_type)
        if is_correct:
            initial_correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} [{task.task_type}] Diff={task.difficulty}")

    initial_acc = initial_correct / 5
    print(f"Initial accuracy (5 samples): {initial_acc:.2%}")

    # Training loop
    print("\n--- Training ---")
    training_results = []
    eval_results = []

    start_time = time.time()

    for episode, task in enumerate(train_tasks, 1):
        # Learn from task
        if hasattr(agent, 'learn_from_feedback'):
            result = agent.learn_from_feedback(
                question=task.question,
                correct_answer=task.answer,
                task_type=task.task_type,
                difficulty=task.difficulty,
            )
        else:
            # Baseline LoRA learner
            result = agent.learn_from_feedback(
                question=task.question,
                correct_answer=task.answer,
                task_type=task.task_type,
            )
            result['difficulty'] = task.difficulty

        training_results.append({
            "episode": episode,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "loss": result["loss"],
            "is_correct": result.get("is_correct", False),
        })

        # Print progress
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{len(train_tasks)}: Loss={result['loss']:.4f}, "
                  f"Type={task.task_type}, Diff={task.difficulty}")

        # Periodic evaluation
        if episode % eval_interval == 0:
            print(f"\n--- Evaluation at Episode {episode} ---")

            correct = 0
            per_difficulty = {i: {"correct": 0, "total": 0} for i in range(1, 6)}

            for task in eval_tasks:
                response = agent.generate_response(
                    task.question,
                    max_new_tokens=40,
                    temperature=0.3
                )
                answer = extract_answer(response, task)
                is_correct = check_correctness(answer, task.answer, task.task_type)

                if is_correct:
                    correct += 1
                    per_difficulty[task.difficulty]["correct"] += 1

                per_difficulty[task.difficulty]["total"] += 1

            accuracy = correct / len(eval_tasks)
            eval_results.append({
                "episode": episode,
                "accuracy": accuracy,
                "correct": correct,
                "total": len(eval_tasks),
            })

            print(f"Accuracy: {accuracy:.2%} ({correct}/{len(eval_tasks)})")

            # Per-difficulty breakdown
            print("Per-difficulty:")
            for diff in range(1, 6):
                if per_difficulty[diff]["total"] > 0:
                    acc = per_difficulty[diff]["correct"] / per_difficulty[diff]["total"]
                    print(f"  Diff {diff}: {acc:.1%} ({per_difficulty[diff]['correct']}/{per_difficulty[diff]['total']})")

            # Save checkpoint
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(f"{exp_dir}/checkpoint_ep{episode}")

    elapsed = time.time() - start_time

    # Final evaluation
    print(f"\n--- Final Evaluation ---")
    final_correct = 0
    final_per_type = {}
    final_per_diff = {i: {"correct": 0, "total": 0} for i in range(1, 6)}

    for task in eval_tasks:
        response = agent.generate_response(task.question, max_new_tokens=40, temperature=0.3)
        answer = extract_answer(response, task)
        is_correct = check_correctness(answer, task.answer, task.task_type)

        if is_correct:
            final_correct += 1
            final_per_diff[task.difficulty]["correct"] += 1

        final_per_diff[task.difficulty]["total"] += 1

        if task.task_type not in final_per_type:
            final_per_type[task.task_type] = {"correct": 0, "total": 0}
        final_per_type[task.task_type]["total"] += 1
        if is_correct:
            final_per_type[task.task_type]["correct"] += 1

    final_acc = final_correct / len(eval_tasks)

    print(f"Final Accuracy: {final_acc:.2%} ({final_correct}/{len(eval_tasks)})")
    print(f"Improvement: {final_acc - initial_acc:+.2%}")
    print(f"Time: {elapsed:.1f}s ({len(train_tasks)/elapsed:.2f} eps/s)")

    # Save results
    results = {
        "method": method_name,
        "model": model_name,
        "config": {
            "n_train": len(train_tasks),
            "n_eval": len(eval_tasks),
        },
        "initial_accuracy": initial_acc,
        "final_accuracy": final_acc,
        "improvement": final_acc - initial_acc,
        "time_seconds": elapsed,
        "training_results": training_results,
        "eval_results": eval_results,
        "per_type": final_per_type,
        "per_difficulty": final_per_diff,
    }

    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {exp_dir}/results.json")

    return results


def run_comparison_experiments(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_train: int = 100,
    n_eval: int = 40,
    eval_interval: int = 20,
    use_curriculum: bool = True,
    checkpoint_dir: str = "./advanced_checkpoints",
    seed: int = 42,
):
    """
    Run comprehensive comparison of all methods
    """

    print("=" * 70)
    print("COMPREHENSIVE CONTINUOUS LEARNING COMPARISON")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Train tasks: {n_train}")
    print(f"Eval tasks: {n_eval}")
    print(f"Curriculum: {use_curriculum}")
    print()

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Initialize arena
    arena = AdvancedReasoningArena(seed=seed)

    # Generate datasets
    print("Generating datasets...")

    # Eval set (held-out, mixed difficulty)
    eval_tasks = arena.generate_batch(n=n_eval, mix=True, difficulty_range=(1, 5))

    # Train set (with curriculum if enabled)
    if use_curriculum:
        print("Using curriculum learning (difficulty 1→5)")
        train_tasks = arena.generate_curriculum(
            n_tasks=n_train,
            start_difficulty=1,
            end_difficulty=5
        )
    else:
        print("Using random difficulty")
        train_tasks = arena.generate_batch(n=n_train, mix=True, difficulty_range=(1, 5))

    print(f"✓ Generated {len(train_tasks)} train tasks, {len(eval_tasks)} eval tasks")

    # Run experiments
    all_results = {}

    # 1. Baseline LoRA
    print("\n" + "="*70)
    print("METHOD 1: Baseline LoRA (No Orthogonality, No Replay)")
    print("="*70)
    agent1 = LoRALearner(
        model_name=model_name,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        use_fp16=False,
    )
    results1 = run_experiment(
        model_name, "Baseline_LoRA", agent1, train_tasks, eval_tasks,
        eval_interval, checkpoint_dir
    )
    all_results["baseline_lora"] = results1

    # 2. O-LoRA (No Replay)
    print("\n" + "="*70)
    print("METHOD 2: O-LoRA (Orthogonal, No Replay)")
    print("="*70)
    agent2 = OrthogonalLoRALearner(
        model_name=model_name,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        use_replay=False,
        orthogonal_reg=0.01,
        use_fp16=False,
    )
    results2 = run_experiment(
        model_name, "O-LoRA_No_Replay", agent2, train_tasks, eval_tasks,
        eval_interval, checkpoint_dir
    )
    all_results["olora_no_replay"] = results2

    # 3. O-LoRA + Replay
    print("\n" + "="*70)
    print("METHOD 3: O-LoRA + Experience Replay")
    print("="*70)
    agent3 = OrthogonalLoRALearner(
        model_name=model_name,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        use_replay=True,
        replay_buffer_size=300,
        replay_sample_size=5,
        orthogonal_reg=0.01,
        use_fp16=False,
    )
    results3 = run_experiment(
        model_name, "O-LoRA_With_Replay", agent3, train_tasks, eval_tasks,
        eval_interval, checkpoint_dir
    )
    all_results["olora_with_replay"] = results3

    # Save comparison
    comparison = {
        "model": model_name,
        "config": {
            "n_train": n_train,
            "n_eval": n_eval,
            "use_curriculum": use_curriculum,
            "seed": seed,
        },
        "results": all_results,
        "summary": {
            method: {
                "final_accuracy": res["final_accuracy"],
                "improvement": res["improvement"],
                "time_seconds": res["time_seconds"],
            }
            for method, res in all_results.items()
        }
    }

    with open(f"{checkpoint_dir}/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Final Acc':<12} {'Improvement':<12} {'Time (s)':<10}")
    print("-" * 70)

    for method, res in all_results.items():
        print(f"{res['method']:<30} {res['final_accuracy']:>10.1%}  {res['improvement']:>10.1%}  {res['time_seconds']:>10.1f}")

    print("\n✓ Comparison complete! Results saved to advanced_checkpoints/")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Advanced Continuous Learning Experiments")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-tasks", type=int, default=100)
    parser.add_argument("--eval-tasks", type=int, default=40)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum")
    parser.add_argument("--checkpoint-dir", type=str, default="./advanced_checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_comparison_experiments(
        model_name=args.model,
        n_train=args.train_tasks,
        n_eval=args.eval_tasks,
        eval_interval=args.eval_interval,
        use_curriculum=args.curriculum,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
