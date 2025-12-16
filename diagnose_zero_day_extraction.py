"""
Diagnostic script to check why zero-day samples are not being extracted from test set.
This will check:
1. Test set composition (what labels are present)
2. Zero-day attack configuration
3. Multiclass label mapping
4. Sequence alignment
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_zero_day_extraction():
    """Diagnose why zero-day samples are not being extracted"""
    
    logger.info("="*80)
    logger.info("ZERO-DAY EXTRACTION DIAGNOSTIC")
    logger.info("="*80)
    
    # 1. Check configuration
    try:
        from config import SystemConfig
        config = SystemConfig()
        logger.info("\n📋 CONFIGURATION:")
        logger.info(f"   Zero-day attack: '{config.zero_day_attack}'")
        logger.info(f"   Zero-day attack label: {config.zero_day_attack_label}")
        logger.info(f"   Dataset type: {getattr(config, 'dataset_type', 'N/A')}")
        logger.info(f"   Data path: {config.data_path}")
        logger.info(f"   Test path: {config.test_path}")
        
        # Check attack types mapping
        if hasattr(config, 'attack_types') and config.attack_types:
            logger.info(f"\n   Attack types mapping:")
            for attack_name, attack_label in config.attack_types.items():
                marker = " ⭐ ZERO-DAY" if attack_name == config.zero_day_attack else ""
                logger.info(f"      {attack_name}: {attack_label}{marker}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return
    
    # 2. Check if preprocessed data exists
    logger.info("\n📊 CHECKING PREPROCESSED DATA:")
    
    try:
        # Try to load saved test set
        saved_test_set_path = Path("saved_test_sets") / f"test_set_{config.zero_day_attack}.npz"
        if saved_test_set_path.exists():
            logger.info(f"   ✅ Found saved test set: {saved_test_set_path}")
            saved_data = np.load(saved_test_set_path, allow_pickle=True)
            
            # Check saved test set composition
            if 'y_test_multiclass' in saved_data:
                y_test_multiclass_saved = saved_data['y_test_multiclass']
                unique_labels_saved = np.unique(y_test_multiclass_saved)
                label_counts_saved = np.bincount(y_test_multiclass_saved.astype(int))
                
                logger.info(f"   Saved test set multiclass labels:")
                logger.info(f"      Total samples: {len(y_test_multiclass_saved)}")
                logger.info(f"      Unique labels: {unique_labels_saved}")
                logger.info(f"      Label distribution:")
                for label in unique_labels_saved:
                    count = label_counts_saved[int(label)]
                    pct = 100 * count / len(y_test_multiclass_saved)
                    marker = " ⭐ ZERO-DAY" if label == config.zero_day_attack_label else ""
                    logger.info(f"         Label {label}: {count} samples ({pct:.1f}%){marker}")
                
                # Check if zero-day label exists
                if config.zero_day_attack_label in unique_labels_saved:
                    zero_day_count_saved = label_counts_saved[config.zero_day_attack_label]
                    logger.info(f"   ✅ Zero-day label {config.zero_day_attack_label} found in saved test set: {zero_day_count_saved} samples")
                else:
                    logger.error(f"   ❌ Zero-day label {config.zero_day_attack_label} NOT found in saved test set!")
                    logger.error(f"      Available labels: {unique_labels_saved.tolist()}")
                    logger.error(f"      This is the root cause - zero-day samples are missing from saved test set!")
            else:
                logger.warning(f"   ⚠️  Saved test set does not contain 'y_test_multiclass'")
        else:
            logger.info(f"   ⚠️  No saved test set found at {saved_test_set_path}")
            logger.info(f"      Test set will be created during preprocessing")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not check saved test set: {e}")
    
    # 3. Check if main.py has been run and check logs
    logger.info("\n🔍 DIAGNOSTIC RECOMMENDATIONS:")
    logger.info("   1. Check the logs above for 'POST-FILTERING VERIFICATION' section")
    logger.info("      - Look for 'Zero-day sequences in filtered multiclass: X'")
    logger.info("      - If X == 0, zero-day samples were filtered out during post-sequence filtering")
    logger.info("")
    logger.info("   2. Check the logs for 'DETAILED ZERO-DAY DIAGNOSTIC' section")
    logger.info("      - Look for 'Unique labels in multiclass sequence: [...]'")
    logger.info("      - Check if zero-day label is in the list")
    logger.info("      - If not, check label mapping in config.attack_types")
    logger.info("")
    logger.info("   3. Common issues:")
    logger.info("      a) Size mismatch: multiclass sequence length != test tensor length")
    logger.info("      b) Label mismatch: zero-day attack label not in multiclass sequence")
    logger.info("      c) Filtering issue: zero-day samples filtered out during post-sequence filtering")
    logger.info("      d) Sequence creation: zero-day samples lost during sequence creation")
    logger.info("")
    logger.info("   4. To fix:")
    logger.info("      - If saved test set has no zero-day samples: delete saved_test_sets/ folder and rerun")
    logger.info("      - If label mismatch: check config.zero_day_attack_label matches actual label in data")
    logger.info("      - If size mismatch: check sequence creation logic in preprocessor")
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    diagnose_zero_day_extraction()









