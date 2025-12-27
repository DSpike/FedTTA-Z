"""
Create a combined performance table for ALL zero-day attacks
Shows aggregated metrics across all attack types
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_attack_results(attack_name: str) -> Dict:
    """Load 100-episode results for an attack type."""
    file_path = Path(f"multi_episode_results/{attack_name.lower()}_100_episodes_phase1.json")

    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        return json.load(f)

def aggregate_metrics(results_list: List[Dict], metric_path: List[str]) -> Tuple[float, float]:
    """
    Aggregate a metric across multiple attack types.

    Args:
        results_list: List of result dictionaries
        metric_path: Path to metric, e.g., ['ttt_model', 'zero_day_detection_rate', 'mean']

    Returns:
        (weighted_mean, pooled_std)
    """
    values = []
    stds = []
    weights = []  # Sample sizes for weighting

    for results in results_list:
        try:
            # Navigate through nested dict
            current = results
            for key in metric_path[:-1]:
                current = current[key]

            # Get mean and std
            mean = current[metric_path[-1]]
            std = current.get('std', 0.0)

            # Weight by dataset size
            weight = results['metadata']['total_samples']

            values.append(mean)
            stds.append(std)
            weights.append(weight)
        except (KeyError, TypeError):
            continue

    if not values:
        return 0.0, 0.0

    # Weighted mean
    total_weight = sum(weights)
    weighted_mean = sum(v * w for v, w in zip(values, weights)) / total_weight

    # Pooled standard deviation (weighted)
    pooled_var = sum(w * (s**2) for w, s in zip(weights, stds)) / total_weight
    pooled_std = np.sqrt(pooled_var)

    return weighted_mean, pooled_std

def create_combined_table():
    """Create combined performance table for all attacks."""

    # Load ALL attack results
    attack_types = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"]

    results_list = []
    attack_names = []
    excluded_attacks = []

    for attack in attack_types:
        data = load_attack_results(attack)
        if data:
            # Include all attacks, but note which ones are small
            results_list.append(data)
            attack_names.append(attack)
            if data['metadata']['total_samples'] < 1000:
                excluded_attacks.append(f"{attack} ({data['metadata']['total_samples']} samples - very small)")
        else:
            excluded_attacks.append(f"{attack} (no data)")

    if not results_list:
        print("❌ No valid results found!")
        return

    print("=" * 80)
    print("COMBINED ZERO-DAY ATTACK PERFORMANCE TABLE (100 Episodes)")
    print("=" * 80)
    print(f"\nIncluded Attack Types: {', '.join(attack_names)}")

    if excluded_attacks:
        print(f"\n⚠️  Note - Small datasets (statistically limited):")
        for exc in excluded_attacks:
            print(f"   - {exc}")

    total_samples = sum(r['metadata']['total_samples'] for r in results_list)
    total_episodes = sum(r['metadata']['n_episodes'] for r in results_list)

    print(f"\nTotal Episodes: {total_episodes}")
    print(f"Total Samples: {total_samples:,}")
    print()
    print("=" * 80)

    # Define metrics to aggregate
    metrics = [
        ("Zero-Day Detection Rate (%)", ['zero_day_detection_rate']),
        ("False Alarm Rate (%)", ['false_alarm_rate']),
        ("F1-Score (%)", ['f1_score']),
        ("Overall Accuracy (%)", ['accuracy']),
        ("Precision (%)", ['precision']),
        ("Recall (%)", ['recall']),
        ("ROC AUC", ['roc_auc']),
        ("AUC-PR", ['auc_pr']),
    ]

    # Print header
    print(f"{'Metric':<30} {'Base Model':<20} {'TTT Model':<20} {'Improvement'}")
    print("-" * 80)

    for metric_name, metric_key in metrics:
        # Get base model values
        base_mean, base_std = aggregate_metrics(
            results_list,
            ['base_model'] + metric_key + ['mean']
        )

        # Get TTT model values
        ttt_mean, ttt_std = aggregate_metrics(
            results_list,
            ['ttt_model'] + metric_key + ['mean']
        )

        # Skip if metric not available
        if base_mean == 0.0 and ttt_mean == 0.0:
            continue

        # Calculate improvement
        improvement = ttt_mean - base_mean

        # Format based on metric type
        if "AUC" in metric_name:
            # AUC metrics (0-1 scale)
            base_str = f"{base_mean:.4f} ± {base_std:.4f}"
            ttt_str = f"{ttt_mean:.4f} ± {ttt_std:.4f}"
            imp_str = f"{improvement:+.4f}"
        else:
            # Percentage metrics
            base_str = f"{base_mean*100:6.2f} ± {base_std*100:.2f}"
            ttt_str = f"{ttt_mean*100:6.2f} ± {ttt_std*100:.2f}"
            imp_str = f"{improvement*100:+6.2f}"

        print(f"{metric_name:<30} {base_str:<20} {ttt_str:<20} {imp_str}")

    print("=" * 80)
    print()

    # Per-attack breakdown
    print("=" * 80)
    print("PER-ATTACK BREAKDOWN")
    print("=" * 80)
    print()

    for attack, results in zip(attack_names, results_list):
        print(f"📊 {attack.upper()}")
        print("-" * 80)

        meta = results['metadata']
        base = results['base_model']
        ttt = results['ttt_model']

        print(f"Dataset Size: {meta['total_samples']:,} samples ({meta['total_zero_day_samples']:,} zero-day)")
        print()

        # Key metrics
        print(f"{'Metric':<30} {'Base':<15} {'TTT':<15} {'Change'}")
        print("-" * 70)

        metrics_to_show = [
            ("ZDR (%)", 'zero_day_detection_rate'),
            ("FAR (%)", 'false_alarm_rate'),
            ("Accuracy (%)", 'accuracy'),
            ("Precision (%)", 'precision'),
            ("Recall (%)", 'recall'),
            ("F1-Score (%)", 'f1_score'),
            ("ROC-AUC", 'roc_auc'),
            ("AUC-PR", 'auc_pr'),
        ]

        for metric_name, key in metrics_to_show:
            if key in base and key in ttt:
                # Check if AUC metric (0-1 scale) or percentage metric
                if "AUC" in metric_name:
                    base_val = base[key]['mean']
                    ttt_val = ttt[key]['mean']
                    change = ttt_val - base_val

                    status = "✅" if (change > 0 and "FAR" not in metric_name) or (change < 0 and "FAR" in metric_name) else "❌"
                    print(f"{metric_name:<30} {base_val:>7.4f}         {ttt_val:>7.4f}         {change:+7.4f} {status}")
                else:
                    base_val = base[key]['mean'] * 100
                    ttt_val = ttt[key]['mean'] * 100
                    change = ttt_val - base_val

                    status = "✅" if (change > 0 and "FAR" not in metric_name) or (change < 0 and "FAR" in metric_name) else "❌"
                    print(f"{metric_name:<30} {base_val:>6.2f}%        {ttt_val:>6.2f}%        {change:+6.2f}% {status}")

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Calculate overall statistics
    zdr_base, _ = aggregate_metrics(results_list, ['base_model', 'zero_day_detection_rate', 'mean'])
    zdr_ttt, _ = aggregate_metrics(results_list, ['ttt_model', 'zero_day_detection_rate', 'mean'])
    far_base, _ = aggregate_metrics(results_list, ['base_model', 'false_alarm_rate', 'mean'])
    far_ttt, _ = aggregate_metrics(results_list, ['ttt_model', 'false_alarm_rate', 'mean'])

    print(f"Average ZDR Improvement: {(zdr_ttt - zdr_base)*100:+.2f}% ({zdr_base*100:.2f}% → {zdr_ttt*100:.2f}%)")
    print(f"Average FAR Change: {(far_ttt - far_base)*100:+.2f}% ({far_base*100:.2f}% → {far_ttt*100:.2f}%)")
    print()

    # Count how many attacks meet publication criteria
    print("Publication-Ready Attack Types (ZDR>90%, FAR<40%, Acc>75%, F1>80%):")
    print("-" * 80)

    excellent_count = 0
    for attack, results in zip(attack_names, results_list):
        ttt = results['ttt_model']
        zdr = ttt['zero_day_detection_rate']['mean'] * 100
        far = ttt['false_alarm_rate']['mean'] * 100
        acc = ttt['accuracy']['mean'] * 100
        f1 = ttt['f1_score']['mean'] * 100

        meets_criteria = zdr > 90 and far < 40 and acc > 75 and f1 > 80

        if meets_criteria:
            excellent_count += 1
            status = "✅ PUBLICATION READY"
        else:
            status = "❌ Needs improvement"

        # Add warning for small datasets
        dataset_size = results['metadata']['total_samples']
        size_note = " ⚠️ (small dataset)" if dataset_size < 1000 else ""

        print(f"  {attack:<15} {status:<25} (ZDR={zdr:.2f}%, FAR={far:.2f}%){size_note}")

    print(f"\nTotal: {excellent_count}/{len(results_list)} attack types meet all criteria")
    print()
    print("=" * 80)

if __name__ == "__main__":
    create_combined_table()
