"""
Comprehensive Zero-Day Evaluation Script

This script automatically runs leave-one-attack-out evaluation for all 9 UNSW-NB15 attack types.
It modifies config_loader.py for each attack, runs main.py, and aggregates results.

Usage:
    python run_comprehensive_evaluation.py

Output:
    - Individual results: zdr_results_{attack_name}.json
    - Comprehensive summary: COMPREHENSIVE_ZDR_RESULTS.json
    - Markdown report: COMPREHENSIVE_ZDR_RESULTS.md
"""

import subprocess
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import sys


def backup_config():
    """Backup original config_loader.py"""
    print("Backing up config_loader.py...")
    shutil.copy('config_loader.py', 'config_loader.py.backup')
    print("✓ Backup created: config_loader.py.backup")


def restore_config():
    """Restore original config_loader.py"""
    print("\nRestoring original config_loader.py...")
    shutil.copy('config_loader.py.backup', 'config_loader.py')
    print("✓ Config restored")


def update_zero_day_attack(attack_name):
    """Update zero_day_attack in config_loader.py"""
    print(f"\nUpdating config for zero-day attack: {attack_name}")

    with open('config_loader.py', 'r') as f:
        config_content = f.read()

    # Find and replace the zero_day_attack line in UNSW section
    # Looking for: 'zero_day_attack': "DoS",
    import re
    pattern = r"('zero_day_attack':\s*['\"])[^'\"]+(['\"])"
    replacement = rf"\g<1>{attack_name}\g<2>"

    new_content = re.sub(pattern, replacement, config_content)

    with open('config_loader.py', 'w') as f:
        f.write(new_content)

    print(f"✓ Config updated: zero_day_attack = '{attack_name}'")


def delete_saved_test_sets():
    """Delete saved test sets to force regeneration"""
    test_set_dir = Path('saved_test_sets')
    if test_set_dir.exists():
        pkl_files = list(test_set_dir.glob('*.pkl'))
        if pkl_files:
            print(f"\nDeleting {len(pkl_files)} saved test set files...")
            for f in pkl_files:
                f.unlink()
            print("✓ Saved test sets deleted")
    else:
        print("\nNo saved test sets directory found (will be created)")


def run_experiment(attack_name):
    """Run main.py for the specified zero-day attack"""
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {attack_name} as Zero-Day")
    print(f"{'='*70}\n")

    # Run main.py
    try:
        result = subprocess.run(
            ['python', 'main.py'],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode != 0:
            print(f"⚠️  Warning: main.py exited with code {result.returncode}")
            print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False

        print("✓ Experiment completed successfully")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ Error: Experiment timed out after 2 hours")
        return False
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False


def extract_results(attack_name):
    """Extract results from performance_metrics_.json"""
    print(f"\nExtracting results for {attack_name}...")

    try:
        with open('performance_plots/performance_metrics_.json', 'r') as f:
            metrics = json.load(f)

        eval_results = metrics['evaluation_results']
        base_model = eval_results['base_model']
        adapted_model = eval_results['adapted_model']

        results = {
            'attack_name': attack_name,
            'timestamp': datetime.now().isoformat(),

            # Base Model
            'base_accuracy': base_model['accuracy'],
            'base_precision': base_model['precision'],
            'base_recall': base_model['recall'],
            'base_f1_score': base_model['f1_score'],
            'base_zdr': base_model['zero_day_detection_rate'],
            'base_far': base_model['far'],
            'base_confusion_matrix': base_model['confusion_matrix'],

            # TTT Adapted Model
            'ttt_accuracy': adapted_model['accuracy'],
            'ttt_precision': adapted_model['precision'],
            'ttt_recall': adapted_model['recall'],
            'ttt_f1_score': adapted_model['f1_score'],
            'ttt_zdr': adapted_model['zero_day_detection_rate'],
            'ttt_far': adapted_model['far'],
            'ttt_confusion_matrix': adapted_model['confusion_matrix'],

            # Improvements
            'accuracy_improvement': adapted_model['accuracy'] - base_model['accuracy'],
            'zdr_improvement': adapted_model['zero_day_detection_rate'] - base_model['zero_day_detection_rate'],
            'far_improvement': base_model['far'] - adapted_model['far'],  # Reduction is good

            # Sample counts
            'zero_day_samples': base_model.get('zero_day_only', {}).get('num_samples', 0),
            'non_zero_day_samples': base_model.get('non_zero_day', {}).get('num_samples', 0),
            'total_samples': base_model['confusion_matrix'][0][0] + base_model['confusion_matrix'][0][1] +
                           base_model['confusion_matrix'][1][0] + base_model['confusion_matrix'][1][1],
        }

        # Save individual result
        result_file = f'zdr_results_{attack_name}.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved: {result_file}")
        print(f"  Base ZDR: {results['base_zdr']:.2%}")
        print(f"  TTT ZDR:  {results['ttt_zdr']:.2%}")
        print(f"  Improvement: +{results['zdr_improvement']:.2%}")

        return results

    except Exception as e:
        print(f"❌ Error extracting results: {e}")
        return None


def generate_summary_report(all_results):
    """Generate comprehensive summary report"""
    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE SUMMARY")
    print(f"{'='*70}\n")

    # Calculate averages
    valid_results = [r for r in all_results.values() if r is not None]

    if not valid_results:
        print("❌ No valid results to summarize")
        return

    avg_base_zdr = sum(r['base_zdr'] for r in valid_results) / len(valid_results)
    avg_ttt_zdr = sum(r['ttt_zdr'] for r in valid_results) / len(valid_results)
    avg_improvement = sum(r['zdr_improvement'] for r in valid_results) / len(valid_results)

    avg_base_accuracy = sum(r['base_accuracy'] for r in valid_results) / len(valid_results)
    avg_ttt_accuracy = sum(r['ttt_accuracy'] for r in valid_results) / len(valid_results)

    avg_base_far = sum(r['base_far'] for r in valid_results) / len(valid_results)
    avg_ttt_far = sum(r['ttt_far'] for r in valid_results) / len(valid_results)

    # Save JSON summary
    summary = {
        'metadata': {
            'total_experiments': len(all_results),
            'successful_experiments': len(valid_results),
            'failed_experiments': len(all_results) - len(valid_results),
            'generated_at': datetime.now().isoformat(),
        },
        'averages': {
            'base_zdr': avg_base_zdr,
            'ttt_zdr': avg_ttt_zdr,
            'zdr_improvement': avg_improvement,
            'base_accuracy': avg_base_accuracy,
            'ttt_accuracy': avg_ttt_accuracy,
            'base_far': avg_base_far,
            'ttt_far': avg_ttt_far,
        },
        'per_attack_results': all_results
    }

    with open('COMPREHENSIVE_ZDR_RESULTS.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ JSON summary saved: COMPREHENSIVE_ZDR_RESULTS.json")

    # Generate markdown report
    generate_markdown_report(summary)

    # Print summary to console
    print(f"\n{'='*70}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nExperiments Completed: {len(valid_results)}/{len(all_results)}")
    print(f"\nAverage Performance:")
    print(f"  Base Model ZDR:        {avg_base_zdr:.2%}")
    print(f"  TTT Adapted Model ZDR: {avg_ttt_zdr:.2%}")
    print(f"  Average Improvement:   +{avg_improvement:.2%}")
    print(f"\n  Base Model Accuracy:   {avg_base_accuracy:.2%}")
    print(f"  TTT Model Accuracy:    {avg_ttt_accuracy:.2%}")
    print(f"\n  Base Model FAR:        {avg_base_far:.2%}")
    print(f"  TTT Model FAR:         {avg_ttt_far:.2%}")

    print(f"\n{'='*70}")
    print("Per-Attack Results:")
    print(f"{'='*70}")
    print(f"{'Attack':<20s} {'Base ZDR':>10s} {'TTT ZDR':>10s} {'Improvement':>12s} {'Samples':>8s}")
    print('-' * 70)

    for attack_name, result in sorted(all_results.items()):
        if result:
            print(f"{attack_name:<20s} {result['base_zdr']:>9.2%} {result['ttt_zdr']:>9.2%} "
                  f"{result['zdr_improvement']:>11.2%} {result['zero_day_samples']:>8d}")
        else:
            print(f"{attack_name:<20s} {'FAILED':>9s} {'FAILED':>9s} {'FAILED':>12s} {'N/A':>8s}")

    print('=' * 70)


def generate_markdown_report(summary):
    """Generate markdown report"""

    report = f"""# Comprehensive Zero-Day Detection Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: UNSW-NB15
**Evaluation Method**: Leave-One-Attack-Out (9 attack types)

---

## Executive Summary

**Experiments Completed**: {summary['metadata']['successful_experiments']}/{summary['metadata']['total_experiments']}

### Average Performance Across All Attack Types

| Metric | Base Model | TTT Adapted | Improvement |
|--------|-----------|-------------|-------------|
| **Zero-Day Detection Rate** | {summary['averages']['base_zdr']:.2%} | {summary['averages']['ttt_zdr']:.2%} | **+{summary['averages']['zdr_improvement']:.2%}** |
| **Accuracy** | {summary['averages']['base_accuracy']:.2%} | {summary['averages']['ttt_accuracy']:.2%} | +{summary['averages']['ttt_accuracy'] - summary['averages']['base_accuracy']:.2%} |
| **False Alarm Rate** | {summary['averages']['base_far']:.2%} | {summary['averages']['ttt_far']:.2%} | {summary['averages']['base_far'] - summary['averages']['ttt_far']:.2%} |

---

## Per-Attack Type Results

### Zero-Day Detection Rate by Attack Type

| Attack Type | Base ZDR | TTT ZDR | Improvement | Zero-Day Samples | Status |
|-------------|----------|---------|-------------|------------------|--------|
"""

    for attack_name, result in sorted(summary['per_attack_results'].items()):
        if result:
            status = '✅' if result['ttt_zdr'] >= 0.90 else '⚠️' if result['ttt_zdr'] >= 0.80 else '❌'
            report += f"| {attack_name} | {result['base_zdr']:.2%} | {result['ttt_zdr']:.2%} | +{result['zdr_improvement']:.2%} | {result['zero_day_samples']} | {status} |\n"
        else:
            report += f"| {attack_name} | N/A | N/A | N/A | N/A | ❌ FAILED |\n"

    report += f"""
**Legend**: ✅ Excellent (≥90%), ⚠️ Good (80-89%), ❌ Needs Improvement (<80%)

---

## Detailed Performance Metrics

### Accuracy by Attack Type

| Attack Type | Base Accuracy | TTT Accuracy | Improvement |
|-------------|--------------|--------------|-------------|
"""

    for attack_name, result in sorted(summary['per_attack_results'].items()):
        if result:
            report += f"| {attack_name} | {result['base_accuracy']:.2%} | {result['ttt_accuracy']:.2%} | +{result['accuracy_improvement']:.2%} |\n"

    report += f"""

### False Alarm Rate by Attack Type

| Attack Type | Base FAR | TTT FAR | Reduction |
|-------------|----------|---------|-----------|
"""

    for attack_name, result in sorted(summary['per_attack_results'].items()):
        if result:
            report += f"| {attack_name} | {result['base_far']:.2%} | {result['ttt_far']:.2%} | {result['far_improvement']:.2%} |\n"

    report += f"""

---

## Key Findings

### Best Performing Attack Types (Highest TTT ZDR)

"""

    valid_results = [(name, res) for name, res in summary['per_attack_results'].items() if res]
    sorted_by_zdr = sorted(valid_results, key=lambda x: x[1]['ttt_zdr'], reverse=True)

    for i, (attack_name, result) in enumerate(sorted_by_zdr[:3], 1):
        report += f"{i}. **{attack_name}**: {result['ttt_zdr']:.2%} ZDR (+{result['zdr_improvement']:.2%} improvement)\n"

    report += f"""

### Worst Performing Attack Types (Lowest TTT ZDR)

"""

    for i, (attack_name, result) in enumerate(sorted_by_zdr[-3:][::-1], 1):
        report += f"{i}. **{attack_name}**: {result['ttt_zdr']:.2%} ZDR (+{result['zdr_improvement']:.2%} improvement)\n"

    report += f"""

### Largest TTT Improvements

"""

    sorted_by_improvement = sorted(valid_results, key=lambda x: x[1]['zdr_improvement'], reverse=True)

    for i, (attack_name, result) in enumerate(sorted_by_improvement[:3], 1):
        report += f"{i}. **{attack_name}**: +{result['zdr_improvement']:.2%} (Base: {result['base_zdr']:.2%} → TTT: {result['ttt_zdr']:.2%})\n"

    report += f"""

---

## Analysis

### Overall Assessment

Average TTT ZDR: **{summary['averages']['ttt_zdr']:.2%}**

"""

    avg_zdr = summary['averages']['ttt_zdr']

    if avg_zdr >= 0.90:
        report += """
**Status**: ✅ **EXCELLENT** - Strong paper with competitive results

Your approach achieves ≥90% average ZDR across all attack types, which is competitive with state-of-the-art methods. This demonstrates that your Test-Time Training approach generalizes well across diverse zero-day attack scenarios.

**Recommendation**: Proceed with Phase 2 (improve base model architecture) to close the accuracy gap, then publish at a top-tier venue (ICLR, INFOCOM, or security conference).
"""
    elif avg_zdr >= 0.85:
        report += """
**Status**: ⚠️ **GOOD** - Publishable with improvements

Your approach achieves 85-90% average ZDR, which is good but needs improvement to compete with SOTA (98-100%). The TTT mechanism shows promise, but architectural enhancements are critical.

**Recommendation**: Proceed with Phase 2 (improve base model) as a high priority. Target: 90%+ ZDR before publication. Consider machine learning conferences (AAAI, ICML workshop) or journals.
"""
    else:
        report += """
**Status**: ❌ **NEEDS SIGNIFICANT IMPROVEMENT**

Your approach achieves <85% average ZDR, which is significantly below SOTA (98-100%). This suggests fundamental issues with either the architecture or the approach.

**Recommendation**: Re-evaluate the base model architecture before proceeding. Consider:
1. Hybrid approach (Random Forest + Neural Network)
2. Replace TCN with Transformer
3. Extensive feature engineering
4. Analyze failure cases to understand root causes
"""

    report += f"""

### Attack Type Characteristics

Based on the results, attack types can be categorized as:

**Easy to Detect** (≥90% ZDR):
"""

    easy_attacks = [name for name, res in valid_results if res['ttt_zdr'] >= 0.90]
    if easy_attacks:
        for attack in easy_attacks:
            report += f"- {attack}\n"
    else:
        report += "- None\n"

    report += """
**Moderate Difficulty** (80-89% ZDR):
"""

    medium_attacks = [name for name, res in valid_results if 0.80 <= res['ttt_zdr'] < 0.90]
    if medium_attacks:
        for attack in medium_attacks:
            report += f"- {attack}\n"
    else:
        report += "- None\n"

    report += """
**Hard to Detect** (<80% ZDR):
"""

    hard_attacks = [name for name, res in valid_results if res['ttt_zdr'] < 0.80]
    if hard_attacks:
        for attack in hard_attacks:
            report += f"- {attack}\n"
    else:
        report += "- None\n"

    report += """

---

## Next Steps

Based on these comprehensive results:

### If Average ZDR ≥ 90%
1. ✅ Your approach is competitive - proceed with confidence
2. Improve base model to close accuracy gap (currently below SOTA)
3. Optimize TTT hyperparameters to push ZDR to 95%+
4. Write paper targeting top-tier venue (ICLR, INFOCOM, S&P)

### If Average ZDR 85-90%
1. ⚠️ Your approach shows promise but needs improvement
2. Priority: Improve base model architecture (Phase 2)
3. Analyze failure cases for hard-to-detect attack types
4. Target: 90%+ average ZDR before submission
5. Consider machine learning conferences or journals

### If Average ZDR < 85%
1. ❌ Fundamental issues need addressing
2. Re-evaluate base model architecture completely
3. Consider hybrid approach (tree-based + neural)
4. Extensive feature engineering required
5. May need to reconsider overall approach

---

## Conclusion

This comprehensive evaluation across all 9 UNSW-NB15 attack types provides a complete picture of your Test-Time Training approach's effectiveness for zero-day detection.

**Key Takeaway**: {
'Your TTT mechanism is highly effective and generalizes well across diverse attack types.' if avg_zdr >= 0.90
else 'Your TTT mechanism shows promise but requires architectural improvements for competitive performance.' if avg_zdr >= 0.85
else 'Significant architectural changes are needed to achieve competitive zero-day detection performance.'
}
"""

    with open('COMPREHENSIVE_ZDR_RESULTS.md', 'w') as f:
        f.write(report)

    print("✓ Markdown report saved: COMPREHENSIVE_ZDR_RESULTS.md")


def main():
    """Main execution function"""

    # List of UNSW-NB15 attack types (excluding Normal)
    zero_day_attacks = [
        'Fuzzers',
        'Analysis',
        'Backdoor',
        'DoS',
        'Exploits',
        'Generic',
        'Reconnaissance',
        'Shellcode',
        'Worms'
    ]

    print(f"\n{'='*70}")
    print("COMPREHENSIVE ZERO-DAY EVALUATION")
    print(f"{'='*70}")
    print(f"\nThis script will run {len(zero_day_attacks)} separate experiments.")
    print(f"Each experiment uses one attack type as zero-day (held out from training).")
    print(f"\nEstimated time: {len(zero_day_attacks)} × 4-6 hours = 40-60 hours")
    print(f"\nAttack types to evaluate:")
    for i, attack in enumerate(zero_day_attacks, 1):
        print(f"  {i}. {attack}")

    print(f"\n{'='*70}")
    response = input("\nProceed with comprehensive evaluation? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("Evaluation cancelled.")
        return

    # Backup original config
    backup_config()

    # Store all results
    all_results = {}

    try:
        for i, attack_name in enumerate(zero_day_attacks, 1):
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {i}/{len(zero_day_attacks)}: {attack_name}")
            print(f"{'='*70}")

            # Update config
            update_zero_day_attack(attack_name)

            # Delete saved test sets
            delete_saved_test_sets()

            # Run experiment
            success = run_experiment(attack_name)

            if success:
                # Extract results
                results = extract_results(attack_name)
                all_results[attack_name] = results
            else:
                all_results[attack_name] = None
                print(f"⚠️  Skipping {attack_name} due to experiment failure")

            print(f"\nProgress: {i}/{len(zero_day_attacks)} experiments completed")

        # Generate comprehensive summary
        generate_summary_report(all_results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        print("Generating partial results...")
        generate_summary_report(all_results)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original config
        restore_config()

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - COMPREHENSIVE_ZDR_RESULTS.json (machine-readable)")
    print("  - COMPREHENSIVE_ZDR_RESULTS.md (human-readable report)")
    print("  - zdr_results_{attack_name}.json (individual results)")
    print("\nOriginal configuration restored.")


if __name__ == '__main__':
    main()
