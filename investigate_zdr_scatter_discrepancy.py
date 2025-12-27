#!/usr/bin/env python3
"""
Investigate why scatter plot shows good separation but ZDR is zero
"""
import logging
import json
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_discrepancy():
    """Investigate the discrepancy between scatter plot and ZDR"""
    
    # Load TTT adaptation data
    results_dir = Path("results")
    ttt_adaptation_file = None
    
    # Find the most recent results directory
    if results_dir.exists():
        result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        if result_dirs:
            latest_dir = result_dirs[0]
            ttt_adaptation_file = latest_dir / "ttt_adaptation_data.json"
    
    if ttt_adaptation_file is None or not ttt_adaptation_file.exists():
        logger.error("❌ TTT adaptation data file not found!")
        logger.error(f"   Expected: {ttt_adaptation_file}")
        return
    
    logger.info(f"📂 Loading TTT adaptation data from: {ttt_adaptation_file}")
    
    with open(ttt_adaptation_file, 'r') as f:
        ttt_data = json.load(f)
    
    # Get attack vs normal data (scatter plot data)
    attack_vs_normal_data = ttt_data.get('attack_vs_normal_data', [])
    
    if not attack_vs_normal_data:
        logger.error("❌ No attack_vs_normal_data found in TTT adaptation data!")
        return
    
    # Get the last step (end of adaptation)
    end_data = attack_vs_normal_data[-1]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 SCATTER PLOT ANALYSIS (End of TTT Adaptation)")
    logger.info(f"{'='*80}")
    
    attack_probs = np.array(end_data.get('attack_probs', []))
    binary_labels = np.array(end_data.get('binary_labels', []))
    
    logger.info(f"   Total samples in scatter plot: {len(attack_probs)}")
    logger.info(f"   Normal samples (label=0): {(binary_labels == 0).sum()}")
    logger.info(f"   Attack samples (label=1): {(binary_labels == 1).sum()}")
    
    # Calculate separation statistics
    normal_probs = attack_probs[binary_labels == 0]
    attack_probs_only = attack_probs[binary_labels == 1]
    
    if len(normal_probs) > 0 and len(attack_probs_only) > 0:
        normal_mean = normal_probs.mean()
        attack_mean = attack_probs_only.mean()
        separation = attack_mean - normal_mean
        
        logger.info(f"\n   📊 Separation Statistics:")
        logger.info(f"      Normal Mean Attack Probability: {normal_mean:.4f} (std: {normal_probs.std():.4f})")
        logger.info(f"      Attack Mean Attack Probability: {attack_mean:.4f} (std: {attack_probs_only.std():.4f})")
        logger.info(f"      Separation (Attack - Normal): {separation:.4f}")
        
        if separation > 0.5:
            logger.info(f"      ✅ EXCELLENT separation - TTT successfully distinguishes attacks from normal")
        elif separation > 0.3:
            logger.info(f"      ✅ GOOD separation - TTT provides reasonable separation")
        elif separation > 0.1:
            logger.info(f"      ⚠️  MODERATE separation - TTT provides some separation")
        else:
            logger.info(f"      ❌ POOR separation - TTT struggles to distinguish")
    
    # Check if zero_day_mask is available
    zero_day_mask = ttt_data.get('zero_day_mask', None)
    
    if zero_day_mask is None:
        logger.warning(f"\n   ⚠️  No zero_day_mask found in TTT adaptation data!")
        logger.warning(f"      Cannot determine which samples are zero-day vs known attacks")
        logger.warning(f"      This is why ZDR might be zero - we can't identify zero-day samples!")
        return
    
    zero_day_mask = np.array(zero_day_mask)
    
    logger.info(f"\n   🎯 ZERO-DAY ANALYSIS:")
    logger.info(f"      Total samples: {len(zero_day_mask)}")
    logger.info(f"      Zero-day samples: {zero_day_mask.sum()} ({100*zero_day_mask.sum()/len(zero_day_mask):.1f}%)")
    logger.info(f"      Non-zero-day samples: {(~zero_day_mask).sum()} ({100*(~zero_day_mask).sum()/len(zero_day_mask):.1f}%)")
    
    # Check if lengths match
    if len(zero_day_mask) != len(attack_probs):
        logger.error(f"\n   ❌ CRITICAL MISMATCH:")
        logger.error(f"      zero_day_mask length: {len(zero_day_mask)}")
        logger.error(f"      attack_probs length: {len(attack_probs)}")
        logger.error(f"      These should match! This is likely the root cause of ZDR=0")
        return
    
    # Analyze zero-day samples specifically
    zero_day_probs = attack_probs[zero_day_mask]
    zero_day_binary_labels = binary_labels[zero_day_mask]
    
    known_attack_probs = attack_probs[~zero_day_mask & (binary_labels == 1)]
    known_attack_binary_labels = binary_labels[~zero_day_mask & (binary_labels == 1)]
    
    logger.info(f"\n   📊 ZERO-DAY vs KNOWN ATTACK COMPARISON:")
    logger.info(f"      Zero-day samples: {len(zero_day_probs)}")
    logger.info(f"         Mean attack probability: {zero_day_probs.mean():.4f}")
    logger.info(f"         Binary labels: Normal={(zero_day_binary_labels == 0).sum()}, Attack={(zero_day_binary_labels == 1).sum()}")
    
    logger.info(f"      Known attack samples: {len(known_attack_probs)}")
    if len(known_attack_probs) > 0:
        logger.info(f"         Mean attack probability: {known_attack_probs.mean():.4f}")
        logger.info(f"         Binary labels: Normal={(known_attack_binary_labels == 0).sum()}, Attack={(known_attack_binary_labels == 1).sum()}")
    
    # Calculate what ZDR would be based on scatter plot data
    # ZDR = TP / (TP + FN) where TP = zero-day predicted as attack, FN = zero-day predicted as normal
    zero_day_tp = ((zero_day_probs > 0.5) & (zero_day_binary_labels == 1)).sum()
    zero_day_fn = ((zero_day_probs <= 0.5) & (zero_day_binary_labels == 1)).sum()
    
    # Also check using binary_labels directly (what the model actually predicted)
    zero_day_predictions = (zero_day_probs > 0.5).astype(int)
    zero_day_tp_from_pred = ((zero_day_predictions == 1) & (zero_day_binary_labels == 1)).sum()
    zero_day_fn_from_pred = ((zero_day_predictions == 0) & (zero_day_binary_labels == 1)).sum()
    
    logger.info(f"\n   🎯 ZDR CALCULATION FROM SCATTER PLOT DATA:")
    logger.info(f"      Zero-day TP (detected): {zero_day_tp_from_pred}")
    logger.info(f"      Zero-day FN (missed): {zero_day_fn_from_pred}")
    
    if (zero_day_tp_from_pred + zero_day_fn_from_pred) > 0:
        zdr_from_scatter = zero_day_tp_from_pred / (zero_day_tp_from_pred + zero_day_fn_from_pred)
        logger.info(f"      ZDR (from scatter plot): {zdr_from_scatter:.4f}")
        
        if zdr_from_scatter == 0.0:
            logger.error(f"\n   ❌ ROOT CAUSE IDENTIFIED:")
            logger.error(f"      ZDR is zero because ALL zero-day samples are predicted as Normal!")
            logger.error(f"      Zero-day mean attack probability: {zero_day_probs.mean():.4f}")
            logger.error(f"      This is below threshold 0.5, so all zero-day samples are classified as Normal")
            logger.error(f"      Even though TTT shows good separation for ALL attacks (known + zero-day),")
            logger.error(f"      it fails specifically for zero-day attacks!")
        else:
            logger.info(f"      ✅ ZDR is non-zero from scatter plot data")
            logger.warning(f"      ⚠️  But ZDR in performance plot is zero - there's a mismatch!")
            logger.warning(f"      This suggests the evaluation uses different predictions/thresholds")
    else:
        logger.error(f"      ❌ Cannot calculate ZDR: No zero-day attack samples found!")
    
    # Compare with known attacks
    if len(known_attack_probs) > 0:
        known_attack_predictions = (known_attack_probs > 0.5).astype(int)
        known_attack_tp = ((known_attack_predictions == 1) & (known_attack_binary_labels == 1)).sum()
        known_attack_fn = ((known_attack_predictions == 0) & (known_attack_binary_labels == 1)).sum()
        
        logger.info(f"\n   📊 KNOWN ATTACK DETECTION (for comparison):")
        logger.info(f"      Known attack TP: {known_attack_tp}")
        logger.info(f"      Known attack FN: {known_attack_fn}")
        if (known_attack_tp + known_attack_fn) > 0:
            known_attack_recall = known_attack_tp / (known_attack_tp + known_attack_fn)
            logger.info(f"      Known attack recall: {known_attack_recall:.4f}")
            
            logger.info(f"\n   🔍 KEY INSIGHT:")
            logger.info(f"      Known attack recall: {known_attack_recall:.4f}")
            logger.info(f"      Zero-day recall (ZDR): {zdr_from_scatter:.4f}")
            logger.info(f"      Difference: {known_attack_recall - zdr_from_scatter:.4f}")
            
            if known_attack_recall > zdr_from_scatter:
                logger.warning(f"      ⚠️  TTT is better at detecting KNOWN attacks than ZERO-DAY attacks!")
                logger.warning(f"      This explains why scatter plot shows good separation (includes known attacks)")
                logger.warning(f"      but ZDR is zero (only zero-day attacks)")

if __name__ == "__main__":
    investigate_discrepancy()



