"""
Diagnose Test Set Composition Issue
Check why test set doesn't have the intended composition
"""

import pickle
import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("TEST SET COMPOSITION DIAGNOSIS")
print("=" * 60)

# Check saved test sets
test_set_dir = Path("saved_test_sets")
if test_set_dir.exists():
    test_set_files = list(test_set_dir.glob("test_set*.pkl"))
    print(f"\n📦 Found {len(test_set_files)} saved test set files")
    
    # Check the best trial test set
    best_trial_path = test_set_dir / "test_set_best_trial.pkl"
    if best_trial_path.exists():
        print(f"\n🔍 Analyzing: {best_trial_path}")
        with open(best_trial_path, 'rb') as f:
            saved_test_set = pickle.load(f)
        
        x_test = saved_test_set.get('X_test')
        y_test = saved_test_set.get('y_test')
        y_test_multiclass = saved_test_set.get('y_test_multiclass')
        test_attack_cat = saved_test_set.get('test_attack_cat')
        test_attack_cat_original = saved_test_set.get('test_attack_cat_original')
        zero_day_attack = saved_test_set.get('zero_day_attack', 'unknown')
        
        print(f"\n📊 Saved Test Set Composition:")
        print(f"   X_test sequences: {len(x_test) if x_test is not None else 'None'}")
        print(f"   y_test (binary): {len(y_test) if y_test is not None else 'None'}")
        print(f"   y_test_multiclass: {len(y_test_multiclass) if y_test_multiclass is not None else 'None'}")
        print(f"   test_attack_cat: {len(test_attack_cat) if test_attack_cat is not None else 'None'}")
        print(f"   test_attack_cat_original: {len(test_attack_cat_original) if test_attack_cat_original is not None else 'None'}")
        print(f"   Zero-day attack: {zero_day_attack}")
        
        # Check size mismatches
        if x_test is not None and y_test_multiclass is not None:
            if len(x_test) != len(y_test_multiclass):
                print(f"\n⚠️ SIZE MISMATCH:")
                print(f"   X_test: {len(x_test)} sequences")
                print(f"   y_test_multiclass: {len(y_test_multiclass)} labels")
                print(f"   Difference: {len(x_test) - len(y_test_multiclass)}")
        
        # Check zero-day composition if multiclass labels exist
        if y_test_multiclass is not None:
            if torch.is_tensor(y_test_multiclass):
                y_multiclass_np = y_test_multiclass.cpu().numpy()
            else:
                y_multiclass_np = np.array(y_test_multiclass)
            
            # Get zero-day label (assuming it's label 5 for Exploits)
            zero_day_label = 5  # This should match config.zero_day_attack_label
            zero_day_mask = (y_multiclass_np == zero_day_label)
            zero_day_count = zero_day_mask.sum()
            total_count = len(y_multiclass_np)
            zero_day_percentage = 100 * zero_day_count / total_count if total_count > 0 else 0
            
            print(f"\n📊 Zero-Day Composition (from y_test_multiclass):")
            print(f"   Zero-day: {zero_day_count}/{total_count} ({zero_day_percentage:.1f}%)")
            print(f"   Non-zero-day: {total_count - zero_day_count}/{total_count} ({100-zero_day_percentage:.1f}%)")
            
            # Check unique labels
            unique_labels, counts = np.unique(y_multiclass_np, return_counts=True)
            print(f"\n   Unique labels: {dict(zip(unique_labels, counts))}")
        
        # Check test_attack_cat_original composition
        if test_attack_cat_original is not None:
            if isinstance(test_attack_cat_original, (list, np.ndarray)):
                attack_cat_np = np.array(test_attack_cat_original)
                unique_attacks, attack_counts = np.unique(attack_cat_np, return_counts=True)
                print(f"\n📊 test_attack_cat_original Composition:")
                print(f"   Total samples: {len(attack_cat_np)}")
                print(f"   Zero-day ({zero_day_attack}): {(attack_cat_np == zero_day_attack).sum()} ({(attack_cat_np == zero_day_attack).sum()/len(attack_cat_np)*100:.1f}%)")
                print(f"   Unique attack types: {dict(zip(unique_attacks, attack_counts))}")
        
        # Check test_attack_cat (sequence-level)
        if test_attack_cat is not None:
            if isinstance(test_attack_cat, (list, np.ndarray)):
                attack_cat_seq = np.array(test_attack_cat)
                print(f"\n📊 test_attack_cat (sequence-level) Composition:")
                print(f"   Total sequences: {len(attack_cat_seq)}")
                zero_day_seq_count = (attack_cat_seq == zero_day_attack).sum()
                print(f"   Zero-day ({zero_day_attack}): {zero_day_seq_count}/{len(attack_cat_seq)} ({100*zero_day_seq_count/len(attack_cat_seq):.1f}%)")
                unique_attacks_seq, attack_counts_seq = np.unique(attack_cat_seq, return_counts=True)
                print(f"   Unique attack types: {dict(zip(unique_attacks_seq, attack_counts_seq))}")
        
        # Check if this matches expected composition
        if y_test_multiclass is not None and x_test is not None:
            if len(x_test) == len(y_test_multiclass):
                print(f"\n✅ Sizes match - should work correctly")
            else:
                print(f"\n❌ SIZE MISMATCH - This will cause fallback to trigger!")
                print(f"   The fallback will use test_attack_cat_original which may be wrong")
    else:
        print(f"\n⚠️ No test_set_best_trial.pkl found")
else:
    print(f"\n⚠️ No saved_test_sets directory found")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)









