"""
Investigate why TTT improves zero-day detection but hurts overall performance.
This script analyzes the optimization results to understand the trade-off.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_optimization_results():
    """Analyze optimization results to understand TTT performance trade-offs."""
    
    # Load best hyperparameters
    best_params_path = Path("best_hyperparameters.json")
    if not best_params_path.exists():
        logger.error("❌ best_hyperparameters.json not found. Run optimization first.")
        return
    
    with open(best_params_path, 'r') as f:
        best_data = json.load(f)
    
    best_trial = best_data.get('best_trial_number', -1)
    best_value = best_data.get('best_value', 0.0)
    best_params = best_data.get('best_params', {})
    best_attrs = best_data.get('best_user_attrs', {})
    
    logger.info("="*80)
    logger.info("📊 ANALYSIS: TTT Performance Trade-off Investigation")
    logger.info("="*80)
    
    logger.info(f"\n🎯 Best Trial: {best_trial} (ZDR: {best_value:.4f} = {best_value*100:.2f}%)")
    
    logger.info("\n📈 Performance Metrics:")
    logger.info(f"  Base Model:")
    logger.info(f"    Accuracy: {best_attrs.get('base_accuracy', 0):.4f}")
    logger.info(f"    F1-Score: {best_attrs.get('base_f1', 0):.4f}")
    logger.info(f"    AUC-PR:   {best_attrs.get('base_auc_pr', 0):.4f}")
    logger.info(f"    ZDR:      {best_attrs.get('base_zdr', 0):.4f} ({best_attrs.get('base_zdr', 0)*100:.2f}%)")
    
    logger.info(f"\n  TTT Model:")
    logger.info(f"    Accuracy: {best_attrs.get('ttt_accuracy', 0):.4f}")
    logger.info(f"    F1-Score: {best_attrs.get('ttt_f1', 0):.4f}")
    logger.info(f"    AUC-PR:   {best_attrs.get('ttt_auc_pr', 0):.4f}")
    logger.info(f"    ZDR:      {best_attrs.get('ttt_zdr', 0):.4f} ({best_attrs.get('ttt_zdr', 0)*100:.2f}%)")
    
    logger.info(f"\n  Changes (TTT - Base):")
    acc_diff = best_attrs.get('ttt_accuracy', 0) - best_attrs.get('base_accuracy', 0)
    f1_diff = best_attrs.get('ttt_f1', 0) - best_attrs.get('base_f1', 0)
    auc_pr_diff = best_attrs.get('ttt_auc_pr', 0) - best_attrs.get('base_auc_pr', 0)
    zdr_diff = best_attrs.get('ttt_zdr', 0) - best_attrs.get('base_zdr', 0)
    
    logger.info(f"    Accuracy: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
    logger.info(f"    F1-Score: {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
    logger.info(f"    AUC-PR:   {auc_pr_diff:+.4f} ({auc_pr_diff*100:+.2f}%)")
    logger.info(f"    ZDR:      {zdr_diff:+.4f} ({zdr_diff*100:+.2f}%)")
    
    logger.info("\n🔍 Key Finding:")
    logger.info("="*80)
    if acc_diff < 0 or f1_diff < 0:
        logger.warning("⚠️  TTT HURTS overall performance (Accuracy/F1 decreased)")
        logger.info("   However, ZDR improved significantly (+{:.2f}%)".format(zdr_diff*100))
        logger.info("\n   💡 Hypothesis:")
        logger.info("   1. Pure TENT (entropy minimization) adapts to test distribution")
        logger.info("   2. Test set has 20-30% zero-day samples")
        logger.info("   3. Entropy minimization makes predictions more confident for ALL samples")
        logger.info("   4. This helps zero-day detection (model learns to identify them)")
        logger.info("   5. BUT may hurt non-zero-day performance (overfitting to test distribution)")
    else:
        logger.info("✅ TTT improves both zero-day AND overall performance")
    
    logger.info("\n⚙️  TTT Configuration (Best Trial):")
    logger.info(f"   use_pseudo_labels: {best_params.get('use_pseudo_labels', False)}")
    logger.info(f"   entropy_weight: {best_params.get('entropy_weight', 0):.4f}")
    logger.info(f"   ttt_lr: {best_params.get('ttt_lr', 0):.6f}")
    logger.info(f"   ttt_base_steps: {best_params.get('ttt_base_steps', 0)}")
    logger.info(f"   ttt_batch_size: {best_params.get('ttt_batch_size', 0)}")
    
    logger.info("\n💡 Root Cause Analysis:")
    logger.info("="*80)
    logger.info("When use_pseudo_labels=False, TTT uses ONLY entropy minimization:")
    logger.info("  • Entropy minimization encourages confident predictions")
    logger.info("  • It adapts to the test distribution (20-30% zero-day, 70-80% non-zero-day)")
    logger.info("  • The model may become overconfident in wrong predictions for non-zero-day samples")
    logger.info("  • This causes accuracy/F1 to decrease while ZDR improves")
    
    logger.info("\n🔧 Potential Solutions:")
    logger.info("="*80)
    logger.info("1. Enable pseudo-labels (use_pseudo_labels=True)")
    logger.info("   • Provides supervision signal from confident predictions")
    logger.info("   • May help prevent overfitting to wrong predictions")
    logger.info("")
    logger.info("2. Use weighted entropy minimization")
    logger.info("   • Weight entropy loss differently for zero-day vs non-zero-day")
    logger.info("   • Only minimize entropy for samples the model is confident about")
    logger.info("")
    logger.info("3. Regularization during TTT")
    logger.info("   • Add L2 penalty to prevent large parameter changes")
    logger.info("   • Use early stopping to prevent overfitting")
    logger.info("")
    logger.info("4. Separate adaptation strategies")
    logger.info("   • Different adaptation for zero-day vs non-zero-day samples")
    logger.info("   • But this requires knowing which samples are zero-day (not realistic)")
    
    logger.info("\n📊 Conclusion:")
    logger.info("="*80)
    logger.info("Pure TENT (entropy-only) improves zero-day detection by adapting to the")
    logger.info("test distribution, but this adaptation can hurt performance on non-zero-day")
    logger.info("samples if the model becomes overconfident in incorrect predictions.")
    logger.info("")
    logger.info("The trade-off depends on:")
    logger.info("  • Zero-day detection priority (high for this use case)")
    logger.info("  • Acceptable degradation in non-zero-day performance")
    logger.info("  • Test distribution characteristics")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    analyze_optimization_results()










