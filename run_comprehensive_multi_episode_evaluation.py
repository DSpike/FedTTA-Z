"""
Comprehensive Multi-Episode Evaluation for All Attack Types

Runs multi-episode evaluation for all 9 UNSW-NB15 attack types in sequence.

Usage:
    python run_comprehensive_multi_episode_evaluation.py

Optional arguments:
    --episodes 10           Number of episodes per attack (default: 10)
    --episode-size 800      Target episode size (default: 800)
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

from config_loader import get_dataset_config
from multi_episode_evaluation import MultiEpisodeEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_config():
    """Backup config_loader.py before modifications."""
    config_path = Path('config_loader.py')
    backup_path = Path('config_loader.py.multi_episode_backup')

    if not backup_path.exists():
        shutil.copy(config_path, backup_path)
        logger.info(f"✅ Backed up config_loader.py to {backup_path}")


def restore_config():
    """Restore original config_loader.py."""
    config_path = Path('config_loader.py')
    backup_path = Path('config_loader.py.multi_episode_backup')

    if backup_path.exists():
        shutil.copy(backup_path, config_path)
        logger.info(f"✅ Restored config_loader.py from backup")


def update_zero_day_attack(attack_name):
    """
    Update zero_day_attack in config_loader.py.

    Args:
        attack_name: Name of attack to set as zero-day
    """
    config_path = Path('config_loader.py')

    with open(config_path, 'r') as f:
        content = f.read()

    # Replace zero_day_attack line
    import re
    pattern = r"('zero_day_attack':\s*['\"])[^'\"]+(['\"])"
    replacement = rf"\g<1>{attack_name}\g<2>"
    new_content = re.sub(pattern, replacement, content)

    with open(config_path, 'w') as f:
        f.write(new_content)

    logger.info(f"✅ Updated zero_day_attack to: {attack_name}")


def delete_saved_test_sets():
    """Delete saved test sets to force regeneration."""
    test_set_dir = Path('saved_test_sets')

    if test_set_dir.exists():
        pkl_files = list(test_set_dir.glob('*.pkl'))
        if pkl_files:
            for f in pkl_files:
                f.unlink()
            logger.info(f"✅ Deleted {len(pkl_files)} saved test sets")


def generate_summary_report(all_results, output_dir):
    """
    Generate comprehensive summary report.

    Args:
        all_results: Dictionary of results for all attacks
        output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Calculate overall statistics
    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if not valid_results:
        logger.error("❌ No valid results to summarize")
        return

    n_attacks = len(valid_results)

    # Extract metrics
    base_zdr_means = [r['base_model']['zero_day_detection_rate']['mean'] for r in valid_results.values()]
    ttt_zdr_means = [r['ttt_model']['zero_day_detection_rate']['mean'] for r in valid_results.values()]
    improvements = [r['improvement']['zero_day_detection_rate']['mean'] for r in valid_results.values()]

    base_acc_means = [r['base_model']['accuracy']['mean'] for r in valid_results.values()]
    ttt_acc_means = [r['ttt_model']['accuracy']['mean'] for r in valid_results.values()]

    base_far_means = [r['base_model']['false_alarm_rate']['mean'] for r in valid_results.values()]
    ttt_far_means = [r['ttt_model']['false_alarm_rate']['mean'] for r in valid_results.values()]

    base_f1_means = [r['base_model'].get('f1_score', {}).get('mean', 0.0) for r in valid_results.values()]
    ttt_f1_means = [r['ttt_model'].get('f1_score', {}).get('mean', 0.0) for r in valid_results.values()]

    # Check if ensemble results exist
    has_ensemble = 'ensemble_model' in list(valid_results.values())[0]
    if has_ensemble:
        ensemble_zdr_means = [r['ensemble_model']['zero_day_detection_rate']['mean'] for r in valid_results.values()]
        ensemble_acc_means = [r['ensemble_model']['accuracy']['mean'] for r in valid_results.values()]
        ensemble_far_means = [r['ensemble_model']['false_alarm_rate']['mean'] for r in valid_results.values()]
        ensemble_f1_means = [r['ensemble_model'].get('f1_score', {}).get('mean', 0.0) for r in valid_results.values()]
        ensemble_improvements = [r['ensemble_improvement']['zero_day_detection_rate']['mean'] for r in valid_results.values()]

    # Overall statistics
    summary = {
        'metadata': {
            'total_attacks_evaluated': n_attacks,
            'attacks_failed': len(all_results) - n_attacks,
            'episodes_per_attack': valid_results[list(valid_results.keys())[0]]['metadata']['n_episodes'],
            'generated_at': datetime.now().isoformat(),
        },

        'overall_statistics': {
            'base_model': {
                'zdr_mean': float(np.mean(base_zdr_means)),
                'zdr_std': float(np.std(base_zdr_means)),
                'accuracy_mean': float(np.mean(base_acc_means)),
                'f1_mean': float(np.mean(base_f1_means)),
                'far_mean': float(np.mean(base_far_means)),
            },
            'ttt_model': {
                'zdr_mean': float(np.mean(ttt_zdr_means)),
                'zdr_std': float(np.std(ttt_zdr_means)),
                'accuracy_mean': float(np.mean(ttt_acc_means)),
                'f1_mean': float(np.mean(ttt_f1_means)),
                'far_mean': float(np.mean(ttt_far_means)),
            },
            'improvement': {
                'zdr_mean': float(np.mean(improvements)),
                'zdr_std': float(np.std(improvements)),
            }
        },

        'per_attack_results': all_results
    }

    # Add ensemble statistics if available
    if has_ensemble:
        summary['overall_statistics']['ensemble_model'] = {
            'zdr_mean': float(np.mean(ensemble_zdr_means)),
            'zdr_std': float(np.std(ensemble_zdr_means)),
            'accuracy_mean': float(np.mean(ensemble_acc_means)),
            'f1_mean': float(np.mean(ensemble_f1_means)),
            'far_mean': float(np.mean(ensemble_far_means)),
        }
        summary['overall_statistics']['ensemble_improvement'] = {
            'zdr_mean': float(np.mean(ensemble_improvements)),
            'zdr_std': float(np.std(ensemble_improvements)),
        }

    # Save JSON
    json_path = output_dir / 'comprehensive_multi_episode_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Saved JSON results to: {json_path}")

    # Generate markdown report
    generate_markdown_report(summary, output_dir)


def generate_markdown_report(summary, output_dir):
    """Generate human-readable markdown report."""
    md_path = output_dir / 'comprehensive_multi_episode_results.md'

    report = f"""# Comprehensive Multi-Episode Zero-Day Detection Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: UNSW-NB15
**Evaluation Method**: Leave-One-Attack-Out with Multi-Episode Evaluation
**Episodes per Attack**: {summary['metadata']['episodes_per_attack']}

---

## Executive Summary

**Attacks Evaluated**: {summary['metadata']['total_attacks_evaluated']}/{summary['metadata']['total_attacks_evaluated'] + summary['metadata']['attacks_failed']}

### Overall Performance (Average Across All Attacks)

| Metric | Base Model | TTT Model | Improvement |
|--------|-----------|-----------|-------------|
| **Zero-Day Detection Rate** | {summary['overall_statistics']['base_model']['zdr_mean']:.2%} ± {summary['overall_statistics']['base_model']['zdr_std']:.2%} | {summary['overall_statistics']['ttt_model']['zdr_mean']:.2%} ± {summary['overall_statistics']['ttt_model']['zdr_std']:.2%} | **+{summary['overall_statistics']['improvement']['zdr_mean']:.2%}** |
| **Accuracy** | {summary['overall_statistics']['base_model']['accuracy_mean']:.2%} | {summary['overall_statistics']['ttt_model']['accuracy_mean']:.2%} | +{summary['overall_statistics']['ttt_model']['accuracy_mean'] - summary['overall_statistics']['base_model']['accuracy_mean']:.2%} |
| **F1-Score** | {summary['overall_statistics']['base_model'].get('f1_mean', 0.0):.2%} | {summary['overall_statistics']['ttt_model'].get('f1_mean', 0.0):.2%} | +{summary['overall_statistics']['ttt_model'].get('f1_mean', 0.0) - summary['overall_statistics']['base_model'].get('f1_mean', 0.0):.2%} |
| **False Alarm Rate** | {summary['overall_statistics']['base_model']['far_mean']:.2%} | {summary['overall_statistics']['ttt_model']['far_mean']:.2%} | {summary['overall_statistics']['base_model']['far_mean'] - summary['overall_statistics']['ttt_model']['far_mean']:.2%} |

---

## Per-Attack Results (with Confidence Intervals)

### Zero-Day Detection Rate

| Attack Type | Base ZDR (Mean ± 95% CI) | TTT ZDR (Mean ± 95% CI) | Improvement | Episodes | Total Samples |
|-------------|--------------------------|-------------------------|-------------|----------|---------------|
"""

    # Sort by TTT ZDR (descending)
    sorted_attacks = sorted(
        [(k, v) for k, v in summary['per_attack_results'].items() if v is not None],
        key=lambda x: x[1]['ttt_model']['zero_day_detection_rate']['mean'],
        reverse=True
    )

    for attack_name, result in sorted_attacks:
        base_zdr = result['base_model']['zero_day_detection_rate']
        ttt_zdr = result['ttt_model']['zero_day_detection_rate']
        improvement = result['improvement']['zero_day_detection_rate']
        n_episodes = result['metadata']['n_episodes']
        total_samples = result['metadata']['total_samples']

        status = '✅' if ttt_zdr['mean'] >= 0.90 else '⚠️' if ttt_zdr['mean'] >= 0.80 else '❌'

        report += f"| {attack_name} | {base_zdr['mean']:.2%} ± {base_zdr['ci_95']:.2%} | {ttt_zdr['mean']:.2%} ± {ttt_zdr['ci_95']:.2%} | +{improvement['mean']:.2%} | {n_episodes} | {total_samples} {status} |\n"

    report += f"""
**Legend**: ✅ Excellent (≥90%), ⚠️ Good (80-89%), ❌ Needs Improvement (<80%)

---

## Detailed Performance Breakdown

### Accuracy by Attack Type

| Attack Type | Base Accuracy (Mean ± 95% CI) | TTT Accuracy (Mean ± 95% CI) | Improvement |
|-------------|-------------------------------|------------------------------|-------------|
"""

    for attack_name, result in sorted_attacks:
        base_acc = result['base_model']['accuracy']
        ttt_acc = result['ttt_model']['accuracy']
        improvement = result['improvement']['accuracy']

        report += f"| {attack_name} | {base_acc['mean']:.2%} ± {base_acc['ci_95']:.2%} | {ttt_acc['mean']:.2%} ± {ttt_acc['ci_95']:.2%} | +{improvement['mean']:.2%} |\n"

    report += f"""

### F1-Score by Attack Type

| Attack Type | Base F1 (Mean ± 95% CI) | TTT F1 (Mean ± 95% CI) | Improvement |
|-------------|-------------------------|------------------------|-------------|
"""

    for attack_name, result in sorted_attacks:
        base_f1 = result['base_model'].get('f1_score', {'mean': 0, 'ci_95': 0})
        ttt_f1 = result['ttt_model'].get('f1_score', {'mean': 0, 'ci_95': 0})

        report += f"| {attack_name} | {base_f1['mean']:.2%} ± {base_f1['ci_95']:.2%} | {ttt_f1['mean']:.2%} ± {ttt_f1['ci_95']:.2%} | +{ttt_f1['mean'] - base_f1['mean']:.2%} |\n"

    report += f"""

### False Alarm Rate by Attack Type

| Attack Type | Base FAR (Mean ± 95% CI) | TTT FAR (Mean ± 95% CI) | Reduction |
|-------------|--------------------------|-------------------------|-----------|
"""

    for attack_name, result in sorted_attacks:
        base_far = result['base_model']['false_alarm_rate']
        ttt_far = result['ttt_model']['false_alarm_rate']

        report += f"| {attack_name} | {base_far['mean']:.2%} ± {base_far['ci_95']:.2%} | {ttt_far['mean']:.2%} ± {ttt_far['ci_95']:.2%} | {base_far['mean'] - ttt_far['mean']:.2%} |\n"

    report += f"""

---

## Key Findings

### Best Performing Attack Types (Highest TTT ZDR)

"""

    for i, (attack_name, result) in enumerate(sorted_attacks[:3], 1):
        zdr = result['ttt_model']['zero_day_detection_rate']
        improvement = result['improvement']['zero_day_detection_rate']
        report += f"{i}. **{attack_name}**: {zdr['mean']:.2%} ± {zdr['ci_95']:.2%} (95% CI, +{improvement['mean']:.2%} improvement)\n"

    report += f"""

### Largest TTT Improvements

"""

    sorted_by_improvement = sorted(
        sorted_attacks,
        key=lambda x: x[1]['improvement']['zero_day_detection_rate']['mean'],
        reverse=True
    )

    for i, (attack_name, result) in enumerate(sorted_by_improvement[:3], 1):
        improvement = result['improvement']['zero_day_detection_rate']
        base_zdr = result['base_model']['zero_day_detection_rate']
        ttt_zdr = result['ttt_model']['zero_day_detection_rate']
        report += f"{i}. **{attack_name}**: +{improvement['mean']:.2%} ± {improvement['ci_95']:.2%} (Base: {base_zdr['mean']:.2%} → TTT: {ttt_zdr['mean']:.2%})\n"

    report += f"""

---

## Statistical Reliability

### Confidence Intervals

All results reported with **95% confidence intervals** computed across {summary['metadata']['episodes_per_attack']} independent evaluation episodes per attack type.

**Interpretation**:
- Mean ± CI indicates the range where the true performance lies with 95% probability
- Smaller CI = more reliable estimate
- CI width decreases with more episodes (current: {summary['metadata']['episodes_per_attack']} episodes)

### Sample Coverage

Total samples evaluated across all attacks and episodes:

"""

    total_zero_day = sum(r['metadata']['total_zero_day_samples'] for r in summary['per_attack_results'].values() if r)
    total_non_zero_day = sum(r['metadata']['total_non_zero_day_samples'] for r in summary['per_attack_results'].values() if r)
    total_all = sum(r['metadata']['total_samples'] for r in summary['per_attack_results'].values() if r)

    report += f"""
- **Total test samples**: {total_all:,}
- **Zero-day samples**: {total_zero_day:,}
- **Non zero-day samples**: {total_non_zero_day:,}

This provides **statistically robust evaluation** compared to single-episode evaluation.

---

## Conclusion

### Overall Assessment

Average TTT ZDR: **{summary['overall_statistics']['ttt_model']['zdr_mean']:.2%}**

"""

    avg_zdr = summary['overall_statistics']['ttt_model']['zdr_mean']

    if avg_zdr >= 0.90:
        report += """
**Status**: ✅ **EXCELLENT** - Strong publication-ready results

Your Test-Time Training approach achieves ≥90% average ZDR across all attack types with robust confidence intervals. This demonstrates strong generalization and is competitive with state-of-the-art methods.

**Recommendation**: Proceed with publication targeting top-tier conferences (ICLR, INFOCOM) or journals. Emphasize the multi-episode evaluation methodology and confidence intervals.
"""
    elif avg_zdr >= 0.85:
        report += """
**Status**: ⚠️ **GOOD** - Publishable with minor improvements

Your approach achieves 85-90% average ZDR, which is good but below SOTA (98-100%). The multi-episode evaluation provides robust statistics for publication.

**Recommendation**: Consider architectural improvements (feature engineering, hybrid models) to push ZDR to 90%+. Current results are publishable at workshops or journals.
"""
    else:
        report += """
**Status**: ❌ **NEEDS IMPROVEMENT**

Average ZDR <85% suggests fundamental issues that need addressing before publication.

**Recommendation**: Focus on improving base model architecture and revisit TTT mechanism before comprehensive evaluation.
"""

    report += """

### Key Strengths

1. ✅ **Multi-episode evaluation** provides statistically robust results
2. ✅ **Confidence intervals** demonstrate reliability
3. ✅ **Comprehensive coverage** across all 9 attack types
4. ✅ **Aligns with meta-learning philosophy** (multiple test episodes)

### Next Steps

Based on these results, recommended next actions are documented in `IMMEDIATE_ACTION_PLAN.md` and `FINAL_VERDICT_AND_ANALYSIS.md`.
"""

    with open(md_path, 'w') as f:
        f.write(report)

    logger.info(f"✅ Saved markdown report to: {md_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive Multi-Episode Evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes per attack (default: 10)')
    parser.add_argument('--episode-size', type=int, default=800,
                       help='Target episode size (default: 800)')
    parser.add_argument('--output-dir', type=str, default='multi_episode_results',
                       help='Output directory (default: multi_episode_results)')

    args = parser.parse_args()

    # UNSW-NB15 attack types (excluding Normal)
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

    logger.info(f"\n{'='*70}")
    logger.info("COMPREHENSIVE MULTI-EPISODE EVALUATION")
    logger.info(f"{'='*70}")
    logger.info(f"\nAttacks to evaluate: {len(zero_day_attacks)}")
    logger.info(f"Episodes per attack: {args.episodes}")
    logger.info(f"Episode size target: {args.episode_size}")
    logger.info(f"Total episodes: {len(zero_day_attacks) * args.episodes}")
    logger.info(f"\nEstimated time: {len(zero_day_attacks)} attacks × {args.episodes} episodes × 15-20 min/episode")
    logger.info(f"                ≈ {len(zero_day_attacks) * args.episodes * 17.5 / 60:.1f} hours")

    response = input("\nProceed with comprehensive evaluation? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Evaluation cancelled.")
        return 0

    # Backup config
    backup_config()

    all_results = {}

    try:
        for i, attack_name in enumerate(zero_day_attacks, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"ATTACK {i}/{len(zero_day_attacks)}: {attack_name}")
            logger.info(f"{'='*70}\n")

            # Update config
            update_zero_day_attack(attack_name)

            # Delete saved test sets
            delete_saved_test_sets()

            # Run multi-episode evaluation
            try:
                config = get_dataset_config('UNSW')  # Use UNSW dataset
                evaluator = MultiEpisodeEvaluator(
                    config=config,
                    n_episodes=args.episodes,
                    episode_size_target=args.episode_size
                )

                results = evaluator.run_evaluation()
                all_results[attack_name] = results

                # Save individual result
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                individual_path = output_dir / f'multi_episode_{attack_name}.json'
                with open(individual_path, 'w') as f:
                    json.dump(results, f, indent=2)

                logger.info(f"✅ {attack_name} completed successfully")

            except Exception as e:
                logger.error(f"❌ {attack_name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[attack_name] = None

            logger.info(f"\nProgress: {i}/{len(zero_day_attacks)} attacks completed")

        # Generate comprehensive summary
        logger.info("\n📊 Generating comprehensive summary...")
        generate_summary_report(all_results, args.output_dir)

        logger.info(f"\n{'='*70}")
        logger.info("COMPREHENSIVE EVALUATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"\nResults saved to: {args.output_dir}/")
        logger.info(f"  - comprehensive_multi_episode_results.json")
        logger.info(f"  - comprehensive_multi_episode_results.md")
        logger.info(f"  - multi_episode_{{attack}}.json (per attack)")

    finally:
        # Restore original config
        restore_config()

    return 0


if __name__ == '__main__':
    exit(main())
