#!/usr/bin/env python3
"""
Analyze per-attack-type performance to identify which specific known attacks are failing.
This will help pinpoint exactly what's causing the inverted performance pattern.
"""

import json
import numpy as np
import logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_per_attack_performance():
    """Detailed analysis of performance per attack type"""

    logger.info("="*80)
    logger.info("PER-ATTACK-TYPE PERFORMANCE ANALYSIS")
    logger.info("="*80)

    # Check if we have saved test sets with attack categories
    saved_test_dir = Path("saved_test_sets")
    if not saved_test_dir.exists():
        logger.error(f"❌ No saved_test_sets directory found")
        logger.info(f"   Cannot analyze per-attack-type performance without saved test sets")
        return

    # Find most recent test set
    test_sets = sorted(saved_test_dir.glob("*_test_set_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not test_sets:
        logger.error(f"❌ No test sets found in {saved_test_dir}")
        return

    test_set_file = test_sets[0]
    logger.info(f"\n📂 Loading test set: {test_set_file.name}")

    import pickle
    with open(test_set_file, 'rb') as f:
        test_set = pickle.load(f)

    # Extract data
    attack_cats = test_set.get('attack_categories', [])
    y_test = test_set.get('y_test')  # Binary labels
    y_test_multiclass = test_set.get('y_test_multiclass')  # Multiclass labels

    if not attack_cats:
        logger.error(f"❌ No attack_categories in test set")
        return

    logger.info(f"   Test set size: {len(attack_cats)} samples")

    # Load predictions from metrics
    metrics_file = Path("performance_plots/performance_metrics_.json")
    if not metrics_file.exists():
        logger.warning(f"⚠️  No metrics file found - will only show test set composition")
        predictions = None
    else:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Try to extract predictions (may not be saved)
        predictions = None
        logger.info(f"   Note: Predictions not stored in metrics - showing test set composition only")

    # Load config to get zero-day attack
    from config_loader import get_dataset_config
    config = get_dataset_config()
    zero_day_attack = config.zero_day_attack

    logger.info(f"\n🎯 Zero-day attack: {zero_day_attack}")

    # Analyze test set composition
    logger.info("\n" + "="*80)
    logger.info("TEST SET COMPOSITION BY ATTACK TYPE")
    logger.info("="*80)

    attack_counts = Counter(attack_cats)
    total_samples = len(attack_cats)

    # Separate into Normal, Known Attacks, Zero-day
    normal_count = attack_counts.get('Normal', 0)
    zd_count = attack_counts.get(zero_day_attack, 0)

    logger.info(f"\n📊 Overall Distribution:")
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Normal: {normal_count} ({100*normal_count/total_samples:.1f}%)")

    known_attacks = {}
    for attack, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
        if attack != 'Normal' and attack != zero_day_attack:
            known_attacks[attack] = count

    known_total = sum(known_attacks.values())
    logger.info(f"   Known attacks (total): {known_total} ({100*known_total/total_samples:.1f}%)")
    for attack, count in sorted(known_attacks.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_samples
        logger.info(f"      {attack}: {count} ({pct:.1f}%)")

    logger.info(f"   Zero-day ({zero_day_attack}): {zd_count} ({100*zd_count/total_samples:.1f}%)")

    # Check if we have binary labels to analyze
    if y_test is not None:
        y_test_np = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else np.array(y_test)

        logger.info("\n" + "="*80)
        logger.info("BINARY LABEL ANALYSIS")
        logger.info("="*80)

        # Check binary labels for each attack type
        logger.info(f"\n🔍 Binary Label Verification:")

        for attack_type in sorted(set(attack_cats)):
            attack_mask = np.array([cat == attack_type for cat in attack_cats])
            attack_binary_labels = y_test_np[attack_mask]

            if len(attack_binary_labels) > 0:
                label_counts = Counter(attack_binary_labels)
                label_0 = label_counts.get(0, 0)
                label_1 = label_counts.get(1, 0)

                is_zd = " ← ZERO-DAY" if attack_type == zero_day_attack else ""
                logger.info(f"   {attack_type}: {len(attack_binary_labels)} samples{is_zd}")
                logger.info(f"      Binary label 0 (Normal): {label_0} ({100*label_0/len(attack_binary_labels):.1f}%)")
                logger.info(f"      Binary label 1 (Attack): {label_1} ({100*label_1/len(attack_binary_labels):.1f}%)")

                if attack_type != 'Normal' and label_0 > 0:
                    logger.warning(f"      ⚠️  WARNING: Attack type '{attack_type}' has {label_0} samples with binary label 0!")

    # Analyze training data composition
    logger.info("\n" + "="*80)
    logger.info("TRAINING DATA CHECK")
    logger.info("="*80)

    logger.info(f"\n📊 To verify training/test distribution match:")
    logger.info(f"   1. Check if training data excluded {zero_day_attack} (zero-day)")
    logger.info(f"   2. Check if training data has similar proportions of known attacks:")

    if known_attacks:
        logger.info(f"\n   Known attacks in test set:")
        for attack, count in sorted(known_attacks.items(), key=lambda x: -x[1]):
            pct = 100 * count / known_total if known_total > 0 else 0
            logger.info(f"      {attack}: {pct:.1f}% of known attacks")

    # Hypothesis about why zero-day performs better
    logger.info("\n" + "="*80)
    logger.info("HYPOTHESIS: Why Zero-Day Performs Better")
    logger.info("="*80)

    logger.info(f"\n🔍 Possible Explanations:")

    logger.info(f"\n   1. FEATURE SIMILARITY:")
    logger.info(f"      - {zero_day_attack} may have SIMPLER or MORE DISTINCTIVE patterns")
    logger.info(f"      - Known attacks may have OVERLAPPING features with Normal traffic")
    logger.info(f"      - Model's general attack detector works better on {zero_day_attack}")

    logger.info(f"\n   2. TRAINING SET IMBALANCE:")
    logger.info(f"      - If known attack types were OVERSAMPLED in training")
    logger.info(f"      - Model may have MEMORIZED specific patterns")
    logger.info(f"      - Slight variations in test set → poor performance")
    logger.info(f"      - {zero_day_attack} is completely different → general features work")

    logger.info(f"\n   3. ZERO-DAY CHARACTERISTICS:")
    logger.info(f"      - {zero_day_attack} samples: {zd_count}")
    logger.info(f"      - {zero_day_attack} may have:")
    logger.info(f"         • Higher attack intensity (easier to detect)")
    logger.info(f"         • More anomalous features (clearer deviation from normal)")
    logger.info(f"         • Less noise (cleaner attack signatures)")

    logger.info(f"\n   4. KNOWN ATTACK ISSUES:")
    logger.info(f"      - Known attacks total: {known_total}")
    logger.info(f"      - Known attacks may have:")
    logger.info(f"         • More subtle attack patterns")
    logger.info(f"         • Higher similarity to normal traffic")
    logger.info(f"         • Greater intra-attack-type variability")

    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("ACTIONABLE NEXT STEPS")
    logger.info("="*80)

    logger.info(f"\n🔧 To diagnose root cause:")

    logger.info(f"\n   1. CHECK FEATURE DISTRIBUTIONS:")
    logger.info(f"      Run: Analyze feature statistics per attack type")
    logger.info(f"      Compare: {zero_day_attack} vs each known attack vs Normal")
    logger.info(f"      Look for: Feature overlap, distinctiveness, variance")

    logger.info(f"\n   2. CHECK TRAINING DATA:")
    logger.info(f"      Verify: Was {zero_day_attack} completely excluded from training?")
    logger.info(f"      Check: Training set proportions of known attacks")
    logger.info(f"      Compare: Training vs test distribution")

    logger.info(f"\n   3. ANALYZE PREDICTION SCORES:")
    logger.info(f"      Extract: Model prediction probabilities per sample")
    logger.info(f"      Plot: Prediction distribution for each attack type")
    logger.info(f"      Identify: Systematic biases in confidence")

    logger.info(f"\n   4. TEST ALTERNATIVE SPLITS:")
    logger.info(f"      Try: Different zero-day attack types")
    logger.info(f"      Check: If pattern persists or is specific to {zero_day_attack}")
    logger.info(f"      Run: Multiple trials with different random seeds")

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    analyze_per_attack_performance()
