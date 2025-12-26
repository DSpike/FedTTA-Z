"""
Analyze the impact of threshold change from 0.85 to 0.78

Compares:
- Baseline (Dec 25): threshold=0.85, attack=unknown (55,900 samples)
- New (Dec 26): threshold=0.78, attack=Exploits (55,400 samples)
"""

import json
from pathlib import Path

def load_results(filepath):
    """Load results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def print_comparison():
    """Print detailed comparison."""

    # Load files
    baseline_file = "multi_episode_results_before_optimization.json"
    new_file = "multi_episode_results/exploits_100_episodes_phase1.json"

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 80)

    if not Path(baseline_file).exists():
        print(f"❌ Baseline file not found: {baseline_file}")
        return

    if not Path(new_file).exists():
        print(f"❌ New results file not found: {new_file}")
        return

    baseline = load_results(baseline_file)
    new = load_results(new_file)

    # Print metadata
    print("\n📊 METADATA COMPARISON")
    print("-" * 80)
    print(f"Baseline:")
    print(f"  Date: {baseline['metadata']['evaluated_at']}")
    print(f"  Total samples: {baseline['metadata']['total_samples']}")
    print(f"  Zero-day samples: {baseline['metadata']['total_zero_day_samples']}")
    print(f"  Episodes: {baseline['metadata']['n_episodes']}")

    print(f"\nNew (Threshold 0.78):")
    print(f"  Date: {new['metadata']['evaluated_at']}")
    print(f"  Total samples: {new['metadata']['total_samples']}")
    print(f"  Zero-day samples: {new['metadata']['total_zero_day_samples']}")
    print(f"  Episodes: {new['metadata']['n_episodes']}")

    # Key metrics comparison
    print("\n📈 KEY METRICS COMPARISON")
    print("=" * 80)

    metrics = [
        ("TTT FAR", "ttt_model", "false_alarm_rate"),
        ("TTT ZDR", "ttt_model", "zero_day_detection_rate"),
        ("TTT Accuracy", "ttt_model", "accuracy"),
        ("TTT Precision", "ttt_model", "precision"),
        ("TTT Recall", "ttt_model", "recall"),
        ("TTT F1", "ttt_model", "f1_score"),
        ("Base FAR", "base_model", "false_alarm_rate"),
        ("Base ZDR", "base_model", "zero_day_detection_rate"),
    ]

    print(f"{'Metric':<20} {'Baseline (0.85)':<20} {'New (0.78)':<20} {'Change':<15} {'Status'}")
    print("-" * 95)

    for name, model, metric in metrics:
        baseline_val = baseline[model][metric]["mean"]
        new_val = new[model][metric]["mean"]
        change = new_val - baseline_val
        change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0

        # Determine status
        if "FAR" in name:
            # Lower FAR is better
            status = "✅ BETTER" if change < 0 else "❌ WORSE"
        else:
            # Higher is better for other metrics
            status = "✅ BETTER" if change > 0 else "❌ WORSE"

        print(f"{name:<20} {baseline_val*100:>6.2f}% ± {baseline[model][metric]['ci_95']*100:.2f}%   "
              f"{new_val*100:>6.2f}% ± {new[model][metric]['ci_95']*100:.2f}%   "
              f"{change*100:>+6.2f}%      {status}")

    # Analysis
    print("\n🔍 ANALYSIS")
    print("=" * 80)

    baseline_far = baseline['ttt_model']['false_alarm_rate']['mean']
    new_far = new['ttt_model']['false_alarm_rate']['mean']
    baseline_zdr = baseline['ttt_model']['zero_day_detection_rate']['mean']
    new_zdr = new['ttt_model']['zero_day_detection_rate']['mean']

    far_change = new_far - baseline_far
    zdr_change = new_zdr - baseline_zdr

    print(f"\n1. FALSE ALARM RATE (FAR):")
    print(f"   Baseline (threshold=0.85): {baseline_far*100:.2f}%")
    print(f"   New (threshold=0.78):      {new_far*100:.2f}%")
    print(f"   Change:                    {far_change*100:+.2f} percentage points")

    if far_change > 0:
        print(f"   ❌ FAR INCREASED - This is unexpected!")
        print(f"      Lower threshold should reduce FAR (fewer attacks predicted)")
    else:
        print(f"   ✅ FAR DECREASED as expected")

    print(f"\n2. ZERO-DAY DETECTION RATE (ZDR):")
    print(f"   Baseline (threshold=0.85): {baseline_zdr*100:.2f}%")
    print(f"   New (threshold=0.78):      {new_zdr*100:.2f}%")
    print(f"   Change:                    {zdr_change*100:+.2f} percentage points")

    if zdr_change > 0:
        print(f"   ✅ ZDR IMPROVED")
    else:
        print(f"   ❌ ZDR DECREASED")

    # Check if datasets are the same
    sample_diff = abs(baseline['metadata']['total_samples'] - new['metadata']['total_samples'])
    if sample_diff > 100:
        print(f"\n⚠️ WARNING: Sample count difference ({sample_diff}) suggests different datasets!")
        print(f"   This may be comparing different attack types or random seeds.")
        print(f"   Results may not be directly comparable.")

    # Conclusion
    print("\n📝 CONCLUSION")
    print("=" * 80)

    if far_change < -0.10:  # FAR reduced by >10 percentage points
        print("✅ SUCCESS: FAR significantly reduced")
    elif far_change < 0:
        print("⚠️ PARTIAL SUCCESS: FAR slightly reduced")
    else:
        print("❌ FAILURE: FAR increased instead of decreasing")
        print("\nPossible reasons:")
        print("1. Different random seeds produced different test sets")
        print("2. Different zero-day attack types")
        print("3. Threshold applied incorrectly")
        print("4. Need to re-run baseline with same attack type for fair comparison")

if __name__ == "__main__":
    print_comparison()
