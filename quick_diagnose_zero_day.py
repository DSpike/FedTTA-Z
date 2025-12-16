"""
Quick diagnostic script to identify why zero-day metrics are zero.
Checks configuration, saved test sets, and identifies root causes.
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

def quick_diagnose():
    """Quick diagnostic to find why zero-day metrics are zero"""
    
    print("="*80)
    print("QUICK ZERO-DAY DIAGNOSTIC")
    print("="*80)
    
    # 1. Check configuration
    print("\n[1/5] CHECKING CONFIGURATION...")
    try:
        from config import SystemConfig
        config = SystemConfig()
        print(f"   ✅ Config loaded")
        print(f"   Zero-day attack: '{config.zero_day_attack}'")
        print(f"   Zero-day attack label: {config.zero_day_attack_label}")
        print(f"   Dataset: {getattr(config, 'dataset_type', 'N/A')}")
        print(f"   Sequence length: {config.sequence_length}")
        print(f"   Sequence stride: {config.sequence_stride}")
        
        # Verify attack types mapping
        if hasattr(config, 'attack_types') and config.zero_day_attack in config.attack_types:
            mapped_label = config.attack_types[config.zero_day_attack]
            if mapped_label == config.zero_day_attack_label:
                print(f"   ✅ Attack type mapping correct: {config.zero_day_attack} → {mapped_label}")
            else:
                print(f"   ❌ MISMATCH: Attack type mapping says {mapped_label} but config.zero_day_attack_label = {config.zero_day_attack_label}")
        else:
            print(f"   ⚠️  Zero-day attack '{config.zero_day_attack}' not in attack_types mapping")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return
    
    # 2. Check saved test sets
    print("\n[2/5] CHECKING SAVED TEST SETS...")
    saved_test_sets_dir = Path("saved_test_sets")
    if saved_test_sets_dir.exists():
        test_set_files = list(saved_test_sets_dir.glob("test_set_*.npz"))
        print(f"   Found {len(test_set_files)} saved test set(s)")
        
        # Check the one matching current zero-day attack
        matching_test_set = saved_test_sets_dir / f"test_set_{config.zero_day_attack}.npz"
        if matching_test_set.exists():
            print(f"   ✅ Found matching test set: {matching_test_set.name}")
            try:
                data = np.load(matching_test_set, allow_pickle=True)
                
                # Check if it has multiclass labels
                if 'y_test_multiclass' in data:
                    y_test_multiclass = data['y_test_multiclass']
                    unique_labels = np.unique(y_test_multiclass)
                    label_counts = np.bincount(y_test_multiclass.astype(int))
                    
                    print(f"   Test set size: {len(y_test_multiclass)} samples")
                    print(f"   Unique labels: {unique_labels}")
                    
                    # Check for zero-day label
                    if config.zero_day_attack_label in unique_labels:
                        zero_day_count = label_counts[config.zero_day_attack_label]
                        zero_day_pct = 100 * zero_day_count / len(y_test_multiclass)
                        print(f"   ✅ Zero-day samples (label {config.zero_day_attack_label}): {zero_day_count} ({zero_day_pct:.1f}%)")
                    else:
                        print(f"   ❌ PROBLEM: Zero-day label {config.zero_day_attack_label} NOT found in saved test set!")
                        print(f"      Available labels: {unique_labels.tolist()}")
                        print(f"      This is likely the root cause!")
                else:
                    print(f"   ⚠️  Saved test set does not contain 'y_test_multiclass'")
                    
                # Check other keys
                print(f"   Available keys: {list(data.keys())}")
                
            except Exception as e:
                print(f"   ❌ Failed to load saved test set: {e}")
        else:
            print(f"   ⚠️  No matching test set found for '{config.zero_day_attack}'")
            print(f"      Expected: {matching_test_set.name}")
            if test_set_files:
                print(f"      Found files: {[f.name for f in test_set_files]}")
    else:
        print(f"   ⚠️  No saved_test_sets directory found")
        print(f"      Test sets will be created during preprocessing")
    
    # 3. Check sequence mapping logic
    print("\n[3/5] ANALYZING SEQUENCE MAPPING LOGIC...")
    print(f"   Sequence mapping uses: last timestep only (idx = seq_idx * stride + (length - 1))")
    print(f"   This means: Zero-day samples must be at the LAST timestep of a sequence to be mapped")
    print(f"   ⚠️  If zero-day samples are in the middle of sequences, they won't be mapped!")
    
    # 4. Check if we can simulate the issue
    print("\n[4/5] CHECKING PREPROCESSED DATA STRUCTURE...")
    try:
        # Check if main.py has been run and has preprocessed data
        # This is a simplified check - actual data would be in memory during runtime
        print(f"   Preprocessed data check: Run main.py to see actual data")
        print(f"   Look for these diagnostic messages in console output:")
        print(f"      - '🔍 SEQUENCE MAPPING DIAGNOSTIC'")
        print(f"      - '🔍 POST-FILTERING VERIFICATION'")
        print(f"      - '🔍 DETAILED ZERO-DAY DIAGNOSTIC'")
    except Exception as e:
        print(f"   ⚠️  Could not check preprocessed data: {e}")
    
    # 5. Root cause analysis and recommendations
    print("\n[5/5] ROOT CAUSE ANALYSIS & RECOMMENDATIONS...")
    print("\n   Possible root causes:")
    print("   1. ❌ Zero-day label mismatch:")
    print(f"      - Config says: {config.zero_day_attack} → label {config.zero_day_attack_label}")
    print(f"      - But data might have different label")
    print(f"      → FIX: Check actual labels in preprocessed data")
    print("")
    print("   2. ❌ Sequence mapping issue:")
    print("      - Zero-day samples not at last timestep of sequences")
    print("      - Only last timestep is used for multiclass label mapping")
    print("      → FIX: Check all timesteps in sequence, not just last one")
    print("")
    print("   3. ❌ Post-sequence filtering removed all zero-day:")
    print("      - Zero-day sequences filtered out during composition adjustment")
    print("      - Available zero-day count was 0 before filtering")
    print("      → FIX: Ensure zero-day samples are preserved during filtering")
    print("")
    print("   4. ❌ Size mismatch between multiclass labels and sequences:")
    print("      - y_test_multiclass_seq length != X_test_seq length")
    print("      - Causes fallback to original attack_cat (may be incorrect)")
    print("      → FIX: Ensure all sequences have corresponding multiclass labels")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\n   💡 NEXT STEPS:")
    print("   1. Run main.py and check console output for diagnostic messages")
    print("   2. Look for 'SEQUENCE MAPPING DIAGNOSTIC' - does it show zero-day sequences?")
    print("   3. Look for 'POST-FILTERING VERIFICATION' - are zero-day sequences preserved?")
    print("   4. Check for ERROR messages about zero-day samples")
    print("   5. If saved test set has zero-day samples but evaluation doesn't, the issue is in sequence mapping")
    print("\n")

if __name__ == "__main__":
    quick_diagnose()









