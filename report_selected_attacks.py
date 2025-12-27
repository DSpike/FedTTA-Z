#!/usr/bin/env python3
"""
Report ZDR and Recall Results for Selected Zero-Day Attacks

Reports performance for: Reconnaissance, Backdoor, DoS, Analysis, and Fuzzers
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np
from scipy import stats

def load_attack_results(attack_name: str) -> Dict:
    """Load 100-episode results for an attack type."""
    file_path = Path(f"multi_episode_results/{attack_name.lower()}_100_episodes_phase1.json")

    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        return json.load(f)

def format_metric(value: float, ci: float, is_percentage: bool = True) -> str:
    """Format metric with confidence interval."""
    if is_percentage:
        return f"{value*100:.2f} ± {ci*100:.2f}"
    else:
        return f"{value:.4f} ± {ci:.4f}"

def extract_per_episode_values(data: Dict, metric_name: str) -> tuple:
    """Extract per-episode values for base and TTT models for a given metric."""
    base_values = []
    ttt_values = []

    for episode in data['per_episode_results']:
        base_val = episode['base_model'].get(metric_name)
        ttt_val = episode['ttt_model'].get(metric_name)

        if base_val is not None and ttt_val is not None:
            base_values.append(base_val)
            ttt_values.append(ttt_val)

    return base_values, ttt_values

def calculate_p_value(base_values: list, ttt_values: list) -> float:
    """Calculate paired t-test p-value for base vs TTT comparison."""
    if len(base_values) == 0 or len(ttt_values) == 0:
        return None

    if len(base_values) != len(ttt_values):
        return None

    # Paired t-test (two-tailed)
    t_stat, p_value = stats.ttest_rel(ttt_values, base_values)
    return p_value

def get_significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"  # not significant

def main():
    # Selected attacks for reporting
    selected_attacks = ["Reconnaissance", "Backdoor", "DoS", "Analysis", "Fuzzers"]

    print("=" * 100)
    print("ZERO-DAY DETECTION PERFORMANCE REPORT")
    print("Selected Attack Scenarios: Reconnaissance, Backdoor, DoS, Analysis, Fuzzers")
    print("=" * 100)
    print()

    # Summary table
    print("=" * 170)
    print("SUMMARY TABLE - ZDR AND RECALL PERFORMANCE (100 Episodes)")
    print("=" * 170)
    print()
    print(f"{'Attack Type':<20} {'Dataset Size':<15} {'Base ZDR (%)':<20} {'TTT ZDR (%)':<20} {'ZDR Δ':<15} "
          f"{'Base Recall (%)':<20} {'TTT Recall (%)':<20} {'Recall Δ':<15}")
    print("-" * 170)

    total_samples = 0
    total_zd_samples = 0

    for attack in selected_attacks:
        data = load_attack_results(attack)
        if not data:
            print(f"{attack:<20} {'NO DATA':<15} {'-':<20} {'-':<20} {'-':<15} {'-':<20} {'-':<20} {'-':<15}")
            continue

        # Extract metrics
        base_zdr = data['base_model']['zero_day_detection_rate']
        ttt_zdr = data['ttt_model']['zero_day_detection_rate']
        zdr_improvement = ttt_zdr['mean'] - base_zdr['mean']

        # Extract recall if available
        base_recall = data['base_model'].get('recall', None)
        ttt_recall = data['ttt_model'].get('recall', None)

        dataset_size = data['metadata']['total_samples']
        zd_samples = data['metadata']['total_zero_day_samples']

        total_samples += dataset_size
        total_zd_samples += zd_samples

        # Calculate p-values from per-episode results
        base_zdr_values, ttt_zdr_values = extract_per_episode_values(data, 'zero_day_detection_rate')
        zdr_p_value = calculate_p_value(base_zdr_values, ttt_zdr_values)
        zdr_sig = get_significance_marker(zdr_p_value)

        # Format recall strings with p-value
        if base_recall and ttt_recall:
            base_recall_str = f"{base_recall['mean']*100:>5.2f} ± {base_recall['ci_95']*100:>4.2f}"
            ttt_recall_str = f"{ttt_recall['mean']*100:>5.2f} ± {ttt_recall['ci_95']*100:>4.2f}"
            recall_improvement = ttt_recall['mean'] - base_recall['mean']

            # Calculate recall p-value
            base_recall_values, ttt_recall_values = extract_per_episode_values(data, 'recall')
            recall_p_value = calculate_p_value(base_recall_values, ttt_recall_values)
            recall_sig = get_significance_marker(recall_p_value)

            recall_imp_str = f"{recall_improvement*100:>+6.2f}% {recall_sig}"
        else:
            base_recall_str = "N/A"
            ttt_recall_str = "N/A"
            recall_imp_str = "N/A"

        print(f"{attack:<20} {dataset_size:>6,} samples   "
              f"{base_zdr['mean']*100:>5.2f} ± {base_zdr['ci_95']*100:>4.2f}      "
              f"{ttt_zdr['mean']*100:>5.2f} ± {ttt_zdr['ci_95']*100:>4.2f}      "
              f"{zdr_improvement*100:>+6.2f}% {zdr_sig:<6} "
              f"{base_recall_str:<20} "
              f"{ttt_recall_str:<20} "
              f"{recall_imp_str:<15}")

    print("-" * 170)
    print(f"{'TOTAL':<20} {total_samples:>6,} samples   ({total_zd_samples:,} zero-day samples)")
    print("=" * 170)
    print()
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print()

    # Detailed breakdown
    print("=" * 100)
    print("DETAILED BREAKDOWN - ZDR AND RECALL WITH ALL METRICS")
    print("=" * 100)
    print()

    for attack in selected_attacks:
        data = load_attack_results(attack)
        if not data:
            continue

        print(f"\n{'='*100}")
        print(f"📊 {attack.upper()}")
        print(f"{'='*100}")
        print(f"Dataset: {data['metadata']['total_samples']:,} total samples "
              f"({data['metadata']['total_zero_day_samples']:,} zero-day, "
              f"{data['metadata']['total_non_zero_day_samples']:,} non-zero-day)")
        print(f"Episodes: {data['metadata']['n_episodes']}")
        print()

        # Base model metrics
        base = data['base_model']
        print("BASE MODEL PERFORMANCE:")
        print("-" * 100)
        print(f"  Zero-Day Detection Rate (ZDR):  {base['zero_day_detection_rate']['mean']*100:>6.2f}% ± {base['zero_day_detection_rate']['ci_95']*100:>5.2f}%")
        if 'recall' in base:
            print(f"  Recall:                         {base['recall']['mean']*100:>6.2f}% ± {base['recall']['ci_95']*100:>5.2f}%")
        print(f"  False Alarm Rate (FAR):         {base['false_alarm_rate']['mean']*100:>6.2f}% ± {base['false_alarm_rate']['ci_95']*100:>5.2f}%")
        print(f"  Accuracy:                       {base['accuracy']['mean']*100:>6.2f}% ± {base['accuracy']['ci_95']*100:>5.2f}%")
        if 'precision' in base:
            print(f"  Precision:                      {base['precision']['mean']*100:>6.2f}% ± {base['precision']['ci_95']*100:>5.2f}%")
        print(f"  F1-Score:                       {base['f1_score']['mean']*100:>6.2f}% ± {base['f1_score']['ci_95']*100:>5.2f}%")
        if 'roc_auc' in base:
            print(f"  ROC-AUC:                        {base['roc_auc']['mean']:.4f} ± {base['roc_auc']['ci_95']:.4f}")
        if 'auc_pr' in base:
            print(f"  AUC-PR:                         {base['auc_pr']['mean']:.4f} ± {base['auc_pr']['ci_95']:.4f}")
        print()

        # TTT model metrics
        ttt = data['ttt_model']
        print("TTT-ENHANCED MODEL PERFORMANCE:")
        print("-" * 100)
        print(f"  Zero-Day Detection Rate (ZDR):  {ttt['zero_day_detection_rate']['mean']*100:>6.2f}% ± {ttt['zero_day_detection_rate']['ci_95']*100:>5.2f}%")
        if 'recall' in ttt:
            print(f"  Recall:                         {ttt['recall']['mean']*100:>6.2f}% ± {ttt['recall']['ci_95']*100:>5.2f}%")
        print(f"  False Alarm Rate (FAR):         {ttt['false_alarm_rate']['mean']*100:>6.2f}% ± {ttt['false_alarm_rate']['ci_95']*100:>5.2f}%")
        print(f"  Accuracy:                       {ttt['accuracy']['mean']*100:>6.2f}% ± {ttt['accuracy']['ci_95']*100:>5.2f}%")
        if 'precision' in ttt:
            print(f"  Precision:                      {ttt['precision']['mean']*100:>6.2f}% ± {ttt['precision']['ci_95']*100:>5.2f}%")
        print(f"  F1-Score:                       {ttt['f1_score']['mean']*100:>6.2f}% ± {ttt['f1_score']['ci_95']*100:>5.2f}%")
        if 'roc_auc' in ttt:
            print(f"  ROC-AUC:                        {ttt['roc_auc']['mean']:.4f} ± {ttt['roc_auc']['ci_95']:.4f}")
        if 'auc_pr' in ttt:
            print(f"  AUC-PR:                         {ttt['auc_pr']['mean']:.4f} ± {ttt['auc_pr']['ci_95']:.4f}")
        print()

        # Improvements with p-values
        print("TTT IMPROVEMENT (Statistical Significance):")
        print("-" * 100)

        # ZDR improvement
        base_zdr_vals, ttt_zdr_vals = extract_per_episode_values(data, 'zero_day_detection_rate')
        zdr_p = calculate_p_value(base_zdr_vals, ttt_zdr_vals)
        zdr_sig_marker = get_significance_marker(zdr_p)
        zdr_imp = (ttt['zero_day_detection_rate']['mean'] - base['zero_day_detection_rate']['mean']) * 100
        p_str_zdr = f"p={zdr_p:.2e}" if zdr_p else "p=N/A"
        print(f"  ZDR Improvement:                {zdr_imp:>+7.2f} percentage points  ({p_str_zdr}, {zdr_sig_marker})")

        # Recall improvement
        if 'recall' in ttt and 'recall' in base:
            base_recall_vals, ttt_recall_vals = extract_per_episode_values(data, 'recall')
            recall_p = calculate_p_value(base_recall_vals, ttt_recall_vals)
            recall_sig_marker = get_significance_marker(recall_p)
            recall_imp = (ttt['recall']['mean'] - base['recall']['mean']) * 100
            p_str_recall = f"p={recall_p:.2e}" if recall_p else "p=N/A"
            print(f"  Recall Improvement:             {recall_imp:>+7.2f} percentage points  ({p_str_recall}, {recall_sig_marker})")

        # FAR change
        base_far_vals, ttt_far_vals = extract_per_episode_values(data, 'false_alarm_rate')
        far_p = calculate_p_value(base_far_vals, ttt_far_vals)
        far_sig_marker = get_significance_marker(far_p)
        far_change = (ttt['false_alarm_rate']['mean'] - base['false_alarm_rate']['mean']) * 100
        p_str_far = f"p={far_p:.2e}" if far_p else "p=N/A"
        print(f"  FAR Change:                     {far_change:>+7.2f} percentage points  ({p_str_far}, {far_sig_marker})")

        # Accuracy improvement
        base_acc_vals, ttt_acc_vals = extract_per_episode_values(data, 'accuracy')
        acc_p = calculate_p_value(base_acc_vals, ttt_acc_vals)
        acc_sig_marker = get_significance_marker(acc_p)
        acc_imp = (ttt['accuracy']['mean'] - base['accuracy']['mean']) * 100
        p_str_acc = f"p={acc_p:.2e}" if acc_p else "p=N/A"
        print(f"  Accuracy Improvement:           {acc_imp:>+7.2f} percentage points  ({p_str_acc}, {acc_sig_marker})")

        # F1-Score improvement
        base_f1_vals, ttt_f1_vals = extract_per_episode_values(data, 'f1_score')
        f1_p = calculate_p_value(base_f1_vals, ttt_f1_vals)
        f1_sig_marker = get_significance_marker(f1_p)
        f1_imp = (ttt['f1_score']['mean'] - base['f1_score']['mean']) * 100
        p_str_f1 = f"p={f1_p:.2e}" if f1_p else "p=N/A"
        print(f"  F1-Score Improvement:           {f1_imp:>+7.2f} percentage points  ({p_str_f1}, {f1_sig_marker})")
        print()
        print("Note: *** = p<0.001 (highly significant), ** = p<0.01 (very significant), * = p<0.05 (significant), ns = not significant")
        print()

    # Publication readiness assessment
    print()
    print("=" * 100)
    print("PUBLICATION READINESS ASSESSMENT")
    print("=" * 100)
    print()
    print("Criteria: ZDR ≥ 90%, FAR ≤ 40%, Accuracy ≥ 75%, F1-Score ≥ 80%")
    print()
    print(f"{'Attack Type':<20} {'ZDR':<12} {'FAR':<12} {'Accuracy':<12} {'F1-Score':<12} {'Status':<25}")
    print("-" * 100)

    for attack in selected_attacks:
        data = load_attack_results(attack)
        if not data:
            continue

        ttt = data['ttt_model']
        zdr = ttt['zero_day_detection_rate']['mean'] * 100
        far = ttt['false_alarm_rate']['mean'] * 100
        acc = ttt['accuracy']['mean'] * 100
        f1 = ttt['f1_score']['mean'] * 100

        # Check criteria
        zdr_pass = zdr >= 90
        far_pass = far <= 40
        acc_pass = acc >= 75
        f1_pass = f1 >= 80

        all_pass = zdr_pass and far_pass and acc_pass and f1_pass

        status = "✅ PUBLICATION READY" if all_pass else "❌ Needs improvement"

        # Show which criteria failed
        issues = []
        if not zdr_pass:
            issues.append(f"ZDR={zdr:.1f}%")
        if not far_pass:
            issues.append(f"FAR={far:.1f}%")
        if not acc_pass:
            issues.append(f"Acc={acc:.1f}%")
        if not f1_pass:
            issues.append(f"F1={f1:.1f}%")

        if issues:
            status += f" ({', '.join(issues)})"

        print(f"{attack:<20} {zdr:>5.2f}%      {far:>5.2f}%      {acc:>5.2f}%      {f1:>5.2f}%      {status}")

    print("=" * 100)
    print()

if __name__ == "__main__":
    main()
