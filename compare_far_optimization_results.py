"""
Compare FAR Optimization Results

Compares results before and after threshold optimization:
- Before: threshold=0.85, FAR=39.5%, ZDR=95.2% (100 episodes)
- After: threshold=0.78, FAR=?, ZDR=? (new results)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_results():
    """Compare before and after optimization results."""

    print("=" * 80)
    print("FAR OPTIMIZATION: BEFORE vs AFTER COMPARISON")
    print("=" * 80)

    # Load current results (this will be the new results)
    try:
        after_results = load_results('multi_episode_results.json')
        after_episodes = after_results['metadata']['n_episodes']
    except FileNotFoundError:
        print("\n❌ No new results found (multi_episode_results.json)")
        print("   Run evaluation first with optimized threshold.")
        return

    # BEFORE results (from previous 100-episode run)
    before = {
        'threshold': 0.85,
        'episodes': 100,
        'base_far': 0.2117,
        'base_zdr': 0.5871,
        'ttt_far': 0.3950,
        'ttt_zdr': 0.9519,
        'ttt_accuracy': 0.7257,
        'ttt_precision': 0.5703,
        'ttt_recall': 0.9428
    }

    # AFTER results (from new run)
    after = {
        'threshold': 0.78,
        'episodes': after_episodes,
        'base_far': after_results['base_model']['false_alarm_rate']['mean'],
        'base_zdr': after_results['base_model']['zero_day_detection_rate']['mean'],
        'ttt_far': after_results['ttt_model']['false_alarm_rate']['mean'],
        'ttt_zdr': after_results['ttt_model']['zero_day_detection_rate']['mean'],
        'ttt_accuracy': after_results['ttt_model']['accuracy']['mean'],
        'ttt_precision': after_results['ttt_model']['precision']['mean'],
        'ttt_recall': after_results['ttt_model']['recall']['mean']
    }

    print(f"\n📊 BEFORE OPTIMIZATION (threshold={before['threshold']}, {before['episodes']} episodes):")
    print(f"   TTT FAR: {before['ttt_far']:.1%}")
    print(f"   TTT ZDR: {before['ttt_zdr']:.1%}")
    print(f"   TTT Accuracy: {before['ttt_accuracy']:.1%}")
    print(f"   TTT Precision: {before['ttt_precision']:.1%}")
    print(f"   TTT Recall: {before['ttt_recall']:.1%}")

    print(f"\n📊 AFTER OPTIMIZATION (threshold={after['threshold']}, {after['episodes']} episodes):")
    print(f"   TTT FAR: {after['ttt_far']:.1%}")
    print(f"   TTT ZDR: {after['ttt_zdr']:.1%}")
    print(f"   TTT Accuracy: {after['ttt_accuracy']:.1%}")
    print(f"   TTT Precision: {after['ttt_precision']:.1%}")
    print(f"   TTT Recall: {after['ttt_recall']:.1%}")

    print(f"\n📈 IMPROVEMENTS:")
    far_change = (after['ttt_far'] - before['ttt_far']) * 100
    zdr_change = (after['ttt_zdr'] - before['ttt_zdr']) * 100
    acc_change = (after['ttt_accuracy'] - before['ttt_accuracy']) * 100
    prec_change = (after['ttt_precision'] - before['ttt_precision']) * 100
    rec_change = (after['ttt_recall'] - before['ttt_recall']) * 100

    print(f"   FAR:       {far_change:+.1f}% {'✅' if far_change < 0 else '❌'}")
    print(f"   ZDR:       {zdr_change:+.1f}% {'✅' if zdr_change >= -5 else '⚠️'}")
    print(f"   Accuracy:  {acc_change:+.1f}%")
    print(f"   Precision: {prec_change:+.1f}%")
    print(f"   Recall:    {rec_change:+.1f}%")

    # Evaluate success
    print(f"\n🎯 OPTIMIZATION SUCCESS CRITERIA:")
    far_target_met = after['ttt_far'] <= 0.20
    zdr_target_met = after['ttt_zdr'] >= 0.90

    print(f"   FAR ≤ 20%: {after['ttt_far']:.1%} {'✅ PASS' if far_target_met else '❌ FAIL'}")
    print(f"   ZDR ≥ 90%: {after['ttt_zdr']:.1%} {'✅ PASS' if zdr_target_met else '❌ FAIL'}")

    if far_target_met and zdr_target_met:
        print(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
        print(f"   Both targets achieved with threshold={after['threshold']}")
    elif far_target_met:
        print(f"\n⚠️ PARTIAL SUCCESS:")
        print(f"   FAR target met, but ZDR below 90%")
        print(f"   Consider lowering threshold slightly (e.g., 0.76-0.77)")
    elif zdr_target_met:
        print(f"\n⚠️ PARTIAL SUCCESS:")
        print(f"   ZDR target met, but FAR above 20%")
        print(f"   Consider additional strategies: ensemble, FAR penalty tuning")
    else:
        print(f"\n❌ OPTIMIZATION UNSUCCESSFUL:")
        print(f"   Neither target met. May need ensemble approach.")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: FAR comparison
    ax = axes[0, 0]
    categories = ['Before\n(0.85)', 'After\n(0.78)']
    far_values = [before['ttt_far'] * 100, after['ttt_far'] * 100]
    colors = ['#e74c3c', '#27ae60' if after['ttt_far'] <= 0.20 else '#f39c12']
    bars = ax.bar(categories, far_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=20, color='green', linestyle='--', linewidth=2, label='Target (20%)')
    ax.set_ylabel('False Alarm Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('FAR Reduction', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, far_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Plot 2: ZDR comparison
    ax = axes[0, 1]
    zdr_values = [before['ttt_zdr'] * 100, after['ttt_zdr'] * 100]
    colors = ['#3498db', '#27ae60' if after['ttt_zdr'] >= 0.90 else '#f39c12']
    bars = ax.bar(categories, zdr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_ylabel('Zero-Day Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('ZDR Maintenance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(85, 100)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, zdr_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Plot 3: Multi-metric comparison
    ax = axes[1, 0]
    metrics = ['Accuracy', 'Precision', 'Recall']
    before_vals = [before['ttt_accuracy']*100, before['ttt_precision']*100, before['ttt_recall']*100]
    after_vals = [after['ttt_accuracy']*100, after['ttt_precision']*100, after['ttt_recall']*100]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, before_vals, width, label='Before (0.85)', color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, after_vals, width, label='After (0.78)', color='#27ae60', alpha=0.7)

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 4: Trade-off visualization
    ax = axes[1, 1]

    # Plot both points on FAR-ZDR space
    ax.scatter([before['ttt_far']*100], [before['ttt_zdr']*100],
              s=300, color='#e74c3c', marker='o', edgecolors='black', linewidth=2,
              label='Before (0.85)', zorder=5)
    ax.scatter([after['ttt_far']*100], [after['ttt_zdr']*100],
              s=300, color='#27ae60', marker='*', edgecolors='black', linewidth=2,
              label='After (0.78)', zorder=5)

    # Draw arrow showing improvement
    ax.annotate('', xy=(after['ttt_far']*100, after['ttt_zdr']*100),
               xytext=(before['ttt_far']*100, before['ttt_zdr']*100),
               arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))

    # Target zone
    ax.axvline(x=20, color='green', linestyle='--', alpha=0.5, label='FAR Target')
    ax.axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='ZDR Target')
    ax.fill_between([0, 20], 90, 100, alpha=0.1, color='green', label='Target Zone')

    ax.set_xlabel('False Alarm Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Zero-Day Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('FAR-ZDR Trade-off Space', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(85, 100)

    plt.suptitle('FAR Optimization Results: Before vs After',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path('performance_plots/far_optimization_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved: {output_path}")

    # Save comparison summary
    summary = {
        'before': before,
        'after': after,
        'improvements': {
            'far_change_pct': far_change,
            'zdr_change_pct': zdr_change,
            'accuracy_change_pct': acc_change,
            'precision_change_pct': prec_change,
            'recall_change_pct': rec_change
        },
        'targets_met': {
            'far_target': far_target_met,
            'zdr_target': zdr_target_met,
            'both_targets': far_target_met and zdr_target_met
        }
    }

    with open('far_optimization_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Comparison summary saved: far_optimization_comparison.json")
    print("=" * 80)

if __name__ == "__main__":
    compare_results()
