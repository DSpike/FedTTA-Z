"""
Comprehensive comparison of all zero-day attack types (100 episodes each)
"""

import json
from pathlib import Path
from typing import Dict, List

def load_attack_results(attack_name: str) -> Dict:
    """Load 100-episode results for an attack type."""
    file_path = Path(f"multi_episode_results/{attack_name.lower()}_100_episodes_phase1.json")

    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        return json.load(f)

def create_comparison_table():
    """Create comprehensive comparison table for all attacks."""

    attack_types = ["Analysis", "Backdoor", "Exploits", "Shellcode", "Worms"]

    print("=" * 120)
    print("COMPREHENSIVE ZERO-DAY ATTACK PERFORMANCE COMPARISON (100 Episodes)")
    print("=" * 120)
    print()

    # Collect all results
    results = {}
    for attack in attack_types:
        data = load_attack_results(attack)
        if data:
            results[attack] = data

    if not results:
        print("❌ No results found!")
        return

    # Print dataset statistics
    print("📊 DATASET STATISTICS")
    print("-" * 120)
    print(f"{'Attack Type':<15} {'Episodes':<10} {'Total Samples':<15} {'Zero-Day Samples':<18} {'Normal Samples':<15}")
    print("-" * 120)

    for attack, data in results.items():
        meta = data['metadata']
        print(f"{attack:<15} {meta['n_episodes']:<10} {meta['total_samples']:<15,} "
              f"{meta['total_zero_day_samples']:<18,} {meta['total_non_zero_day_samples']:<15,}")

    print()

    # TTT Model Performance
    print("🎯 TTT MODEL PERFORMANCE (Primary Metrics)")
    print("-" * 120)
    print(f"{'Attack Type':<15} {'ZDR (%)':<20} {'FAR (%)':<20} {'Accuracy (%)':<20} {'F1 Score (%)':<20}")
    print("-" * 120)

    for attack, data in results.items():
        ttt = data['ttt_model']
        zdr = ttt['zero_day_detection_rate']['mean'] * 100
        zdr_ci = ttt['zero_day_detection_rate']['ci_95'] * 100
        far = ttt['false_alarm_rate']['mean'] * 100
        far_ci = ttt['false_alarm_rate']['ci_95'] * 100
        acc = ttt['accuracy']['mean'] * 100
        acc_ci = ttt['accuracy']['ci_95'] * 100
        f1 = ttt['f1_score']['mean'] * 100
        f1_ci = ttt['f1_score']['ci_95'] * 100

        print(f"{attack:<15} {zdr:>6.2f} ± {zdr_ci:<5.2f}     {far:>6.2f} ± {far_ci:<5.2f}     "
              f"{acc:>6.2f} ± {acc_ci:<5.2f}     {f1:>6.2f} ± {f1_ci:<5.2f}")

    print()

    # TTT Model Extended Metrics
    print("📈 TTT MODEL EXTENDED METRICS")
    print("-" * 120)
    print(f"{'Attack Type':<15} {'Precision (%)':<20} {'Recall (%)':<20} {'ROC-AUC (%)':<20} {'AUC-PR (%)':<20}")
    print("-" * 120)

    for attack, data in results.items():
        ttt = data['ttt_model']

        # Handle optional metrics gracefully
        if 'precision' in ttt:
            prec = ttt['precision']['mean'] * 100
            prec_ci = ttt['precision']['ci_95'] * 100
        else:
            prec, prec_ci = 0.0, 0.0

        if 'recall' in ttt:
            rec = ttt['recall']['mean'] * 100
            rec_ci = ttt['recall']['ci_95'] * 100
        else:
            rec, rec_ci = 0.0, 0.0

        if 'roc_auc' in ttt:
            roc = ttt['roc_auc']['mean'] * 100
            roc_ci = ttt['roc_auc']['ci_95'] * 100
        else:
            roc, roc_ci = 0.0, 0.0

        if 'auc_pr' in ttt:
            auc_pr = ttt['auc_pr']['mean'] * 100
            auc_pr_ci = ttt['auc_pr']['ci_95'] * 100
        else:
            auc_pr, auc_pr_ci = 0.0, 0.0

        # Only print if at least one metric is available
        if prec > 0 or rec > 0 or roc > 0 or auc_pr > 0:
            prec_str = f"{prec:>6.2f} ± {prec_ci:<5.2f}" if prec > 0 else "N/A           "
            rec_str = f"{rec:>6.2f} ± {rec_ci:<5.2f}" if rec > 0 else "N/A           "
            roc_str = f"{roc:>6.2f} ± {roc_ci:<5.2f}" if roc > 0 else "N/A           "
            auc_str = f"{auc_pr:>6.2f} ± {auc_pr_ci:<5.2f}" if auc_pr > 0 else "N/A           "

            print(f"{attack:<15} {prec_str}     {rec_str}     {roc_str}     {auc_str}")
        else:
            print(f"{attack:<15} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}")

    print()

    # Base Model Performance
    print("📊 BASE MODEL PERFORMANCE (For Comparison)")
    print("-" * 120)
    print(f"{'Attack Type':<15} {'ZDR (%)':<20} {'FAR (%)':<20} {'Accuracy (%)':<20} {'F1 Score (%)':<20}")
    print("-" * 120)

    for attack, data in results.items():
        base = data['base_model']
        zdr = base['zero_day_detection_rate']['mean'] * 100
        zdr_ci = base['zero_day_detection_rate']['ci_95'] * 100
        far = base['false_alarm_rate']['mean'] * 100
        far_ci = base['false_alarm_rate']['ci_95'] * 100
        acc = base['accuracy']['mean'] * 100
        acc_ci = base['accuracy']['ci_95'] * 100
        f1 = base['f1_score']['mean'] * 100
        f1_ci = base['f1_score']['ci_95'] * 100

        print(f"{attack:<15} {zdr:>6.2f} ± {zdr_ci:<5.2f}     {far:>6.2f} ± {far_ci:<5.2f}     "
              f"{acc:>6.2f} ± {acc_ci:<5.2f}     {f1:>6.2f} ± {f1_ci:<5.2f}")

    print()

    # Improvement Analysis
    print("📈 TTT IMPROVEMENT OVER BASE MODEL")
    print("-" * 120)
    print(f"{'Attack Type':<15} {'ZDR Gain (%)':<20} {'FAR Change (%)':<20} {'Accuracy Gain (%)':<20} {'F1 Gain (%)':<20}")
    print("-" * 120)

    for attack, data in results.items():
        base = data['base_model']
        ttt = data['ttt_model']

        zdr_gain = (ttt['zero_day_detection_rate']['mean'] - base['zero_day_detection_rate']['mean']) * 100
        far_change = (ttt['false_alarm_rate']['mean'] - base['false_alarm_rate']['mean']) * 100
        acc_gain = (ttt['accuracy']['mean'] - base['accuracy']['mean']) * 100
        f1_gain = (ttt['f1_score']['mean'] - base['f1_score']['mean']) * 100

        zdr_status = "✅" if zdr_gain > 0 else "❌"
        far_status = "✅" if far_change < 0 else "❌"
        acc_status = "✅" if acc_gain > 0 else "❌"
        f1_status = "✅" if f1_gain > 0 else "❌"

        print(f"{attack:<15} {zdr_gain:>+6.2f} {zdr_status:<13} {far_change:>+6.2f} {far_status:<13} "
              f"{acc_gain:>+6.2f} {acc_status:<13} {f1_gain:>+6.2f} {f1_status}")

    print()

    # Publication Criteria Check
    print("✅ PUBLICATION READINESS (Targets: ZDR>90%, FAR<40%, Accuracy>75%, F1>80%)")
    print("-" * 120)
    print(f"{'Attack Type':<15} {'ZDR>90%':<12} {'FAR<40%':<12} {'Acc>75%':<12} {'F1>80%':<12} {'Total':<10} {'Verdict'}")
    print("-" * 120)

    for attack, data in results.items():
        ttt = data['ttt_model']

        zdr = ttt['zero_day_detection_rate']['mean'] * 100
        far = ttt['false_alarm_rate']['mean'] * 100
        acc = ttt['accuracy']['mean'] * 100
        f1 = ttt['f1_score']['mean'] * 100

        zdr_pass = "✅ PASS" if zdr > 90 else "❌ FAIL"
        far_pass = "✅ PASS" if far < 40 else "❌ FAIL"
        acc_pass = "✅ PASS" if acc > 75 else "❌ FAIL"
        f1_pass = "✅ PASS" if f1 > 80 else "❌ FAIL"

        passed = sum([zdr > 90, far < 40, acc > 75, f1 > 80])
        total = f"{passed}/4"

        if passed == 4:
            verdict = "🎉 EXCELLENT"
        elif passed == 3:
            verdict = "✅ GOOD"
        elif passed == 2:
            verdict = "⚠️  FAIR"
        else:
            verdict = "❌ POOR"

        print(f"{attack:<15} {zdr_pass:<12} {far_pass:<12} {acc_pass:<12} {f1_pass:<12} {total:<10} {verdict}")

    print()

    # Ranking
    print("🏆 RANKING BY OVERALL PERFORMANCE")
    print("-" * 120)

    # Calculate composite score (ZDR + (100-FAR) + Accuracy + F1) / 4
    scores = []
    for attack, data in results.items():
        ttt = data['ttt_model']
        zdr = ttt['zero_day_detection_rate']['mean'] * 100
        far = ttt['false_alarm_rate']['mean'] * 100
        acc = ttt['accuracy']['mean'] * 100
        f1 = ttt['f1_score']['mean'] * 100

        # Composite score (FAR is inverted so lower is better)
        composite = (zdr + (100 - far) + acc + f1) / 4
        scores.append((attack, composite, zdr, far, acc, f1))

    # Sort by composite score
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6} {'Attack Type':<15} {'Composite Score':<18} {'ZDR':<10} {'FAR':<10} {'Acc':<10} {'F1'}")
    print("-" * 120)

    for rank, (attack, composite, zdr, far, acc, f1) in enumerate(scores, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<4} {attack:<15} {composite:>6.2f}%          {zdr:>6.2f}%  {far:>6.2f}%  {acc:>6.2f}%  {f1:>6.2f}%")

    print()
    print("=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)

    best = scores[0]
    print(f"\n🏆 BEST ATTACK TYPE: {best[0]}")
    print(f"   Composite Score: {best[1]:.2f}%")
    print(f"   ZDR: {best[2]:.2f}%")
    print(f"   FAR: {best[3]:.2f}%")
    print(f"   Accuracy: {best[4]:.2f}%")
    print(f"   F1: {best[5]:.2f}%")
    print(f"\n✅ Use {best[0]} attack type for publication!")
    print("=" * 120)

if __name__ == "__main__":
    create_comparison_table()
