"""
Comprehensive Analysis and Visualization of Advanced Experiments

Compares:
1. Baseline LoRA
2. O-LoRA (Orthogonal LoRA)
3. O-LoRA + Experience Replay

Plus analysis by:
- Task difficulty
- Task type
- Learning progression
- Forgetting analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_comparison_results(checkpoint_dir="./advanced_checkpoints"):
    """Load comparison results"""
    with open(f"{checkpoint_dir}/comparison_results.json") as f:
        return json.load(f)


def create_comprehensive_plots(results, save_dir="./advanced_checkpoints"):
    """Create comprehensive comparison plots"""

    fig = plt.figure(figsize=(20, 12))

    methods_data = results["results"]
    methods = list(methods_data.keys())
    method_names = [methods_data[m]["method"] for m in methods]

    # Plot 1: Final Accuracy Comparison
    ax1 = plt.subplot(3, 4, 1)
    final_accs = [methods_data[m]["final_accuracy"] * 100 for m in methods]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax1.bar(range(len(methods)), final_accs, color=colors, alpha=0.8)
    ax1.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    for i, (bar, val) in enumerate(zip(bars, final_accs)):
        ax1.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Plot 2: Improvement Over Baseline
    ax2 = plt.subplot(3, 4, 2)
    improvements = [methods_data[m]["improvement"] * 100 for m in methods]
    bars = ax2.barh(range(len(methods)), improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Learning Improvement', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax2.text(val + 1, i, f'+{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    # Plot 3: Time Efficiency
    ax3 = plt.subplot(3, 4, 3)
    times = [methods_data[m]["time_seconds"] for m in methods]
    bars = ax3.bar(range(len(methods)), times, color=colors, alpha=0.8)
    ax3.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax3.set_title('Training Efficiency', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Efficiency Score (Accuracy / Time)
    ax4 = plt.subplot(3, 4, 4)
    efficiency = [final_accs[i] / times[i] * 10 for i in range(len(methods))]
    bars = ax4.bar(range(len(methods)), efficiency, color=colors, alpha=0.8)
    ax4.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
    ax4.set_title('Efficiency (Acc/Time)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5-7: Learning Curves (Accuracy over time)
    for idx, method in enumerate(methods):
        ax = plt.subplot(3, 4, 5 + idx)
        eval_results = methods_data[method]["eval_results"]

        if eval_results:
            episodes = [r["episode"] for r in eval_results]
            accuracies = [r["accuracy"] * 100 for r in eval_results]

            ax.plot(episodes, accuracies, marker='o', linewidth=2.5,
                   markersize=7, color=colors[idx], label=methods_data[method]["method"])
            ax.axhline(y=methods_data[method]["initial_accuracy"] * 100,
                      color='red', linestyle='--', linewidth=2, alpha=0.7, label='Initial')

            ax.fill_between(episodes, methods_data[method]["initial_accuracy"] * 100,
                           accuracies, alpha=0.3, color=colors[idx])

            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(f'{methods_data[method]["method"]}\nLearning Curve',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])

    # Plot 8: Per-Difficulty Performance
    ax8 = plt.subplot(3, 4, 8)
    difficulties = [1, 2, 3, 4, 5]
    width = 0.25

    for idx, method in enumerate(methods):
        per_diff = methods_data[method]["per_difficulty"]
        accs = []
        for diff in difficulties:
            diff_data = per_diff.get(str(diff), {"correct": 0, "total": 1})
            acc = diff_data["correct"] / diff_data["total"] * 100 if diff_data["total"] > 0 else 0
            accs.append(acc)

        x = np.arange(len(difficulties)) + idx * width
        ax8.bar(x, accs, width, label=methods_data[method]["method"],
               color=colors[idx], alpha=0.8)

    ax8.set_xlabel('Difficulty Level', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax8.set_title('Performance by Difficulty', fontsize=11, fontweight='bold')
    ax8.set_xticks(np.arange(len(difficulties)) + width)
    ax8.set_xticklabels(difficulties)
    ax8.legend(fontsize=8)
    ax8.grid(axis='y', alpha=0.3)
    ax8.set_ylim([0, 100])

    # Plot 9: Training Loss Curves
    ax9 = plt.subplot(3, 4, 9)
    for idx, method in enumerate(methods):
        training_results = methods_data[method]["training_results"]

        episodes = [r["episode"] for r in training_results]
        losses = [r["loss"] for r in training_results if not np.isnan(r["loss"])]
        valid_episodes = [r["episode"] for r in training_results if not np.isnan(r["loss"])]

        if len(losses) > 10:
            # Smooth
            window = 10
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smoothed_eps = valid_episodes[window-1:]
            ax9.plot(smoothed_eps, smoothed, linewidth=2, label=methods_data[method]["method"],
                    color=colors[idx])

    ax9.set_xlabel('Episode', fontsize=10)
    ax9.set_ylabel('Training Loss', fontsize=10)
    ax9.set_title('Training Loss Comparison', fontsize=11, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # Plot 10: Instant Accuracy (Before→After Learning)
    ax10 = plt.subplot(3, 4, 10)
    for idx, method in enumerate(methods):
        training_results = methods_data[method]["training_results"]

        instant_acc = [r["is_correct"] for r in training_results]
        # Moving average
        if len(instant_acc) > 10:
            window = 10
            smoothed = np.convolve(instant_acc, np.ones(window)/window, mode='valid')
            episodes = list(range(window-1, len(instant_acc)))
            ax10.plot(episodes, smoothed * 100, linewidth=2,
                     label=methods_data[method]["method"], color=colors[idx])

    ax10.set_xlabel('Episode', fontsize=10)
    ax10.set_ylabel('Instant Accuracy (%)', fontsize=10)
    ax10.set_title('Instant Learning Performance', fontsize=11, fontweight='bold')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    ax10.set_ylim([0, 100])

    # Plot 11: Per-Task-Type Performance
    ax11 = plt.subplot(3, 4, 11)

    # Get all task types
    all_task_types = set()
    for method in methods:
        all_task_types.update(methods_data[method]["per_type"].keys())

    task_types = sorted(list(all_task_types))
    n_types = len(task_types)
    width = 0.25

    for idx, method in enumerate(methods):
        per_type = methods_data[method]["per_type"]
        accs = []
        for tt in task_types:
            if tt in per_type and per_type[tt]["total"] > 0:
                acc = per_type[tt]["correct"] / per_type[tt]["total"] * 100
            else:
                acc = 0
            accs.append(acc)

        x = np.arange(n_types) + idx * width
        ax11.bar(x, accs, width, label=methods_data[method]["method"],
                color=colors[idx], alpha=0.8)

    ax11.set_xlabel('Task Type', fontsize=10, fontweight='bold')
    ax11.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax11.set_title('Performance by Task Type', fontsize=11, fontweight='bold')
    ax11.set_xticks(np.arange(n_types) + width)
    ax11.set_xticklabels([tt.replace('_', '\n') for tt in task_types], fontsize=7, rotation=45, ha='right')
    ax11.legend(fontsize=8)
    ax11.grid(axis='y', alpha=0.3)
    ax11.set_ylim([0, 100])

    # Plot 12: Summary Table (Text)
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    summary_text = "COMPARISON SUMMARY\n" + "="*40 + "\n\n"
    for method in methods:
        data = methods_data[method]
        summary_text += f"{data['method']}:\n"
        summary_text += f"  Final Acc: {data['final_accuracy']*100:.1f}%\n"
        summary_text += f"  Improvement: +{data['improvement']*100:.1f}%\n"
        summary_text += f"  Time: {data['time_seconds']:.1f}s\n\n"

    ax12.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax12.transAxes)

    plt.suptitle('Advanced Continuous Learning: Comprehensive Comparison',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    plt.savefig(f'{save_dir}/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Comprehensive analysis saved to {save_dir}/comprehensive_analysis.png")
    plt.close()


def generate_report(results, save_dir="./advanced_checkpoints"):
    """Generate markdown report"""

    report = f"""# Advanced Continuous Learning - Experimental Results

**Model:** {results['model']}
**Date:** 2025-10-23
**Configuration:**
- Train tasks: {results['config']['n_train']}
- Eval tasks: {results['config']['n_eval']}
- Curriculum learning: {results['config']['use_curriculum']}

## Research-Based Improvements

### Methods Compared

1. **Baseline LoRA** - Standard LoRA without enhancements
2. **O-LoRA** - Orthogonal LoRA (EMNLP 2023) to reduce interference
3. **O-LoRA + Replay** - O-LoRA with experience replay buffer

### Key Research Papers Implemented

- **O-LoRA:** Orthogonal Subspace Learning for Language Model Continual Learning (EMNLP 2023)
- **CURLoRA:** Stable LLM Continual Fine-Tuning (2024)
- **Experience Replay:** Inspired by continual learning literature

## Results Summary

| Method | Final Accuracy | Improvement | Time (s) | Efficiency |
|--------|---------------|-------------|----------|------------|
"""

    methods_data = results["results"]
    for method_key, method_data in methods_data.items():
        final_acc = method_data["final_accuracy"] * 100
        improvement = method_data["improvement"] * 100
        time_s = method_data["time_seconds"]
        efficiency = final_acc / time_s * 10

        report += f"| {method_data['method']} | {final_acc:.1f}% | +{improvement:.1f}% | {time_s:.1f}s | {efficiency:.2f} |\n"

    report += "\n## Detailed Analysis\n\n"

    for method_key, method_data in methods_data.items():
        report += f"### {method_data['method']}\n\n"
        report += f"- **Final Accuracy:** {method_data['final_accuracy']*100:.2f}%\n"
        report += f"- **Initial Accuracy:** {method_data['initial_accuracy']*100:.2f}%\n"
        report += f"- **Improvement:** +{method_data['improvement']*100:.2f}%\n"
        report += f"- **Training Time:** {method_data['time_seconds']:.2f}s\n"
        report += f"- **Speed:** {method_data['config']['n_train']/method_data['time_seconds']:.2f} episodes/s\n\n"

        # Per-difficulty breakdown
        report += "**Performance by Difficulty:**\n\n"
        per_diff = method_data["per_difficulty"]
        for diff in range(1, 6):
            diff_data = per_diff.get(str(diff), {"correct": 0, "total": 0})
            if diff_data["total"] > 0:
                acc = diff_data["correct"] / diff_data["total"] * 100
                report += f"- Difficulty {diff}: {acc:.1f}% ({diff_data['correct']}/{diff_data['total']})\n"

        report += "\n**Performance by Task Type:**\n\n"
        per_type = method_data["per_type"]
        for task_type, data in sorted(per_type.items()):
            if data["total"] > 0:
                acc = data["correct"] / data["total"] * 100
                report += f"- {task_type}: {acc:.1f}% ({data['correct']}/{data['total']})\n"

        report += "\n---\n\n"

    # Save report
    with open(f"{save_dir}/ADVANCED_RESULTS.md", "w") as f:
        f.write(report)

    print(f"✓ Report saved to {save_dir}/ADVANCED_RESULTS.md")


def main():
    print("Loading results...")
    results = load_comparison_results()

    print("\nCreating visualizations...")
    create_comprehensive_plots(results)

    print("\nGenerating report...")
    generate_report(results)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
