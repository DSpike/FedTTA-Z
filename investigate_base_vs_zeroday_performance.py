#!/usr/bin/env python3
"""
Investigate why base model performs worse on normal+known attacks but better on zero-day attacks.

This counterintuitive pattern suggests:
1. Overfitting to training attack types
2. Poor generalization to similar but slightly different attack patterns
3. Good generalization to very different attack patterns (zero-day)
4. Possible issues with confidence calibration
5. Training set composition issues
"""

import json
import numpy as np
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_performance_discrepancy():
    """Analyze why base model has inverted performance pattern"""

    logger.info("="*80)
    logger.info("INVESTIGATING BASE MODEL vs ZERO-DAY PERFORMANCE DISCREPANCY")
    logger.info("="*80)

    # Load performance metrics
    metrics_file = Path("performance_plots/performance_metrics_.json")
    if not metrics_file.exists():
        logger.error(f"❌ Metrics file not found: {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Extract relevant data
    eval_results = metrics.get('evaluation_results', {})
    base_model = eval_results.get('base_model', {})
    ttt_model = eval_results.get('adapted_model', {})

    if not base_model:
        logger.error("❌ No base_model results found in metrics")
        return

    logger.info("\n" + "="*80)
    logger.info("1. OVERALL PERFORMANCE COMPARISON")
    logger.info("="*80)

    # Overall metrics
    base_cm = base_model.get('confusion_matrix', [[0,0],[0,0]])
    ttt_cm = ttt_model.get('confusion_matrix', [[0,0],[0,0]]) if ttt_model else [[0,0],[0,0]]

    logger.info("\n📊 BASE MODEL - Overall Performance:")
    logger.info(f"   Confusion Matrix: {base_cm}")
    if len(base_cm) == 2 and len(base_cm[1]) == 2:
        tn, fp = base_cm[0]
        fn, tp = base_cm[1]
        total_normal = tn + fp
        total_attack = fn + tp

        accuracy = (tn + tp) / (total_normal + total_attack) if (total_normal + total_attack) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / total_normal if total_normal > 0 else 0

        logger.info(f"   Total samples: {total_normal + total_attack}")
        logger.info(f"   Normal samples: {total_normal}")
        logger.info(f"   Attack samples: {total_attack} (Known + Zero-day)")
        logger.info(f"")
        logger.info(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        logger.info(f"   Accuracy: {accuracy:.2%}")
        logger.info(f"   Precision: {precision:.2%}")
        logger.info(f"   Recall (Attack Detection): {recall:.2%}")
        logger.info(f"   F1 Score: {f1:.2%}")
        logger.info(f"   FAR (False Alarm Rate): {far:.2%}")

    # Zero-day only performance
    base_zd = base_model.get('zero_day_only', {})
    if base_zd:
        logger.info("\n📊 BASE MODEL - Zero-Day Only Performance:")
        zd_cm = base_zd.get('confusion_matrix', [[0,0],[0,0]])
        zd_samples = base_zd.get('num_samples', 0)
        zdr = base_zd.get('zero_day_detection_rate', 0)

        logger.info(f"   Confusion Matrix: {zd_cm}")
        logger.info(f"   Total zero-day samples: {zd_samples}")

        if len(zd_cm) == 2 and len(zd_cm[1]) == 2:
            zd_tn, zd_fp = zd_cm[0]
            zd_fn, zd_tp = zd_cm[1]

            zd_accuracy = (zd_tn + zd_tp) / zd_samples if zd_samples > 0 else 0
            zd_precision = zd_tp / (zd_tp + zd_fp) if (zd_tp + zd_fp) > 0 else 0
            zd_recall = zd_tp / (zd_tp + zd_fn) if (zd_tp + zd_fn) > 0 else 0

            logger.info(f"   TN={zd_tn}, FP={zd_fp}, FN={zd_fn}, TP={zd_tp}")
            logger.info(f"   Accuracy: {zd_accuracy:.2%}")
            logger.info(f"   Precision: {zd_precision:.2%}")
            logger.info(f"   Zero-Day Detection Rate: {zdr:.2%}")

    # Calculate known attack performance (Overall - Zero-day)
    logger.info("\n" + "="*80)
    logger.info("2. KNOWN ATTACK vs ZERO-DAY BREAKDOWN")
    logger.info("="*80)

    if base_zd and len(base_cm) == 2 and len(zd_cm) == 2:
        # Known attacks = Total attacks - Zero-day attacks
        total_attacks = base_cm[1][0] + base_cm[1][1]  # FN + TP
        zd_attacks = zd_cm[1][0] + zd_cm[1][1] if (zd_cm[1][0] + zd_cm[1][1]) > 0 else zd_samples
        known_attacks = total_attacks - zd_attacks

        # Known attack TP/FN
        total_tp = base_cm[1][1]
        total_fn = base_cm[1][0]
        zd_tp = zd_cm[1][1]
        zd_fn = zd_cm[1][0]
        known_tp = total_tp - zd_tp
        known_fn = total_fn - zd_fn

        known_detection_rate = known_tp / (known_tp + known_fn) if (known_tp + known_fn) > 0 else 0

        logger.info(f"\n📊 Attack Composition:")
        logger.info(f"   Total attack samples: {total_attacks}")
        logger.info(f"   Known attacks: {known_attacks} ({100*known_attacks/total_attacks if total_attacks > 0 else 0:.1f}%)")
        logger.info(f"   Zero-day attacks: {zd_attacks} ({100*zd_attacks/total_attacks if total_attacks > 0 else 0:.1f}%)")

        logger.info(f"\n📊 Known Attack Performance:")
        logger.info(f"   TP={known_tp}, FN={known_fn}")
        logger.info(f"   Detection Rate: {known_detection_rate:.2%}")

        logger.info(f"\n📊 Zero-Day Attack Performance:")
        logger.info(f"   TP={zd_tp}, FN={zd_fn}")
        logger.info(f"   Detection Rate: {zdr:.2%}")

        # KEY INSIGHT
        logger.info("\n" + "="*80)
        logger.info("3. KEY PERFORMANCE DISCREPANCY")
        logger.info("="*80)

        if zdr > known_detection_rate:
            diff = zdr - known_detection_rate
            logger.info(f"\n⚠️  INVERTED PERFORMANCE PATTERN DETECTED!")
            logger.info(f"   Known Attack Detection: {known_detection_rate:.2%}")
            logger.info(f"   Zero-Day Detection: {zdr:.2%}")
            logger.info(f"   Difference: {diff:+.2%} (Zero-day BETTER by {diff*100:.1f} percentage points)")

            logger.info(f"\n🔍 POSSIBLE CAUSES:")
            logger.info(f"   1. OVERFITTING to Training Attack Patterns:")
            logger.info(f"      - Model learned very specific features of training attacks")
            logger.info(f"      - Test set 'known' attacks may differ slightly from training")
            logger.info(f"      - Zero-day attacks are so different that general patterns work better")

            logger.info(f"\n   2. DATASET DISTRIBUTION MISMATCH:")
            logger.info(f"      - Training set may not represent test set 'known' attacks well")
            logger.info(f"      - Check if known attack types in test differ from training")

            logger.info(f"\n   3. CONFIDENCE CALIBRATION ISSUES:")
            logger.info(f"      - Model may be overconfident on known attacks → wrong predictions")
            logger.info(f"      - Model may be more cautious on zero-day → correct by accident")

            logger.info(f"\n   4. FEATURE DISTRIBUTION SHIFT:")
            logger.info(f"      - Known attacks in test may have different feature distributions")
            logger.info(f"      - Zero-day attacks happen to match general attack patterns better")

            logger.info(f"\n   5. TRAINING SET COMPOSITION:")
            logger.info(f"      - Check if zero-day attack type was underrepresented in training")
            logger.info(f"      - Model may have learned more robust features due to imbalance")
        else:
            logger.info(f"\n✅ EXPECTED PATTERN:")
            logger.info(f"   Known Attack Detection: {known_detection_rate:.2%}")
            logger.info(f"   Zero-Day Detection: {zdr:.2%}")
            logger.info(f"   Difference: {zdr - known_detection_rate:+.2%}")
            logger.info(f"   Known attacks detected better (as expected)")

    # Analyze prediction confidence
    logger.info("\n" + "="*80)
    logger.info("4. PREDICTION CONFIDENCE ANALYSIS")
    logger.info("="*80)

    # Check if we have probability distributions
    if 'roc_curve' in base_model:
        roc = base_model['roc_curve']
        fpr = roc.get('fpr', [])
        tpr = roc.get('tpr', [])

        if fpr and tpr:
            # Find operating point (closest to top-left corner)
            distances = [(f**2 + (1-t)**2, f, t) for f, t in zip(fpr, tpr)]
            distances.sort()
            _, best_fpr, best_tpr = distances[0]

            logger.info(f"\n📊 ROC Curve Analysis:")
            logger.info(f"   Best Operating Point: FPR={best_fpr:.3f}, TPR={best_tpr:.3f}")
            logger.info(f"   AUC: {base_model.get('roc_auc', 'N/A')}")

    # Training data analysis
    logger.info("\n" + "="*80)
    logger.info("5. TRAINING DATA COMPOSITION CHECK")
    logger.info("="*80)

    training_history = metrics.get('training_history')
    if training_history:
        logger.info("\n📊 Training History Available:")
        if isinstance(training_history, list) and len(training_history) > 0:
            final_epoch = training_history[-1]
            logger.info(f"   Final epoch: {final_epoch.get('epoch', 'N/A')}")
            logger.info(f"   Final train loss: {final_epoch.get('train_loss', 'N/A')}")
            logger.info(f"   Final val loss: {final_epoch.get('val_loss', 'N/A')}")

            # Check for overfitting
            train_loss = final_epoch.get('train_loss', 0)
            val_loss = final_epoch.get('val_loss', 0)
            if val_loss > train_loss * 1.2:
                logger.info(f"\n⚠️  OVERFITTING DETECTED:")
                logger.info(f"   Validation loss ({val_loss:.4f}) significantly higher than training loss ({train_loss:.4f})")
                logger.info(f"   This suggests the model memorized training data rather than learning general patterns")

    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("6. RECOMMENDATIONS")
    logger.info("="*80)

    if base_zd and zdr > known_detection_rate:
        logger.info("\n🔧 To fix inverted performance pattern:")
        logger.info("\n   A. DATA-LEVEL FIXES:")
        logger.info("      1. Verify training/test split:")
        logger.info("         - Check if 'known' attacks in test are same types as training")
        logger.info("         - Ensure stratified sampling preserves attack type distribution")
        logger.info("      ")
        logger.info("      2. Increase training data diversity:")
        logger.info("         - Add more variations of each attack type")
        logger.info("         - Use data augmentation for attack samples")
        logger.info("      ")
        logger.info("      3. Check for data leakage:")
        logger.info("         - Ensure test 'known' attacks aren't contaminated")
        logger.info("         - Verify zero-day attack was properly excluded from training")

        logger.info("\n   B. MODEL-LEVEL FIXES:")
        logger.info("      1. Regularization:")
        logger.info("         - Increase dropout rate")
        logger.info("         - Add L2 regularization")
        logger.info("         - Use early stopping")
        logger.info("      ")
        logger.info("      2. Architecture changes:")
        logger.info("         - Reduce model capacity (fewer parameters)")
        logger.info("         - Use simpler model that learns more general features")
        logger.info("      ")
        logger.info("      3. Training procedure:")
        logger.info("         - Use cross-validation to check consistency")
        logger.info("         - Train on more diverse attack types")
        logger.info("         - Balance training set better")

        logger.info("\n   C. EVALUATION-LEVEL CHECKS:")
        logger.info("      1. Verify label correctness:")
        logger.info("         - Check if 'known' attack labels are correct in test set")
        logger.info("         - Verify zero-day mask identifies correct samples")
        logger.info("      ")
        logger.info("      2. Analyze per-attack-type performance:")
        logger.info("         - Break down known attack detection by specific attack type")
        logger.info("         - Identify which known attacks are failing")
        logger.info("      ")
        logger.info("      3. Check prediction distributions:")
        logger.info("         - Plot prediction scores for known vs zero-day")
        logger.info("         - Look for systematic bias in confidence scores")

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    analyze_performance_discrepancy()
