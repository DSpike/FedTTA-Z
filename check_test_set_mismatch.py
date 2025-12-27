#!/usr/bin/env python3
"""Check for dataset mismatch between saved test sets and current config"""
import pickle
import os
from pathlib import Path
from config_loader import get_dataset_config

def check_saved_test_sets():
    """Check saved test sets for dataset/zero-day attack mismatch"""
    
    config = get_dataset_config()
    current_dataset = "UNSW" if "UNSW" in config.data_path else "CICIDS2017"
    current_zero_day = config.zero_day_attack
    current_grouping = config.use_category_grouping
    
    print("=" * 80)
    print("DATASET AND TEST SET MISMATCH CHECK")
    print("=" * 80)
    print(f"\n📊 Current Configuration:")
    print(f"   Dataset: {current_dataset}")
    print(f"   Data Path: {config.data_path}")
    print(f"   Zero-Day Attack: {current_zero_day}")
    print(f"   Attack Grouping: {current_grouping}")
    print(f"   Zero-Day Attack Label: {config.zero_day_attack_label}")
    
    # Check saved test sets directory
    saved_test_sets_dir = Path("saved_test_sets")
    if not saved_test_sets_dir.exists():
        print(f"\n✅ No saved_test_sets directory found - no mismatch possible")
        return
    
    # Find all saved test sets
    test_set_files = list(saved_test_sets_dir.glob("*.pkl"))
    
    if not test_set_files:
        print(f"\n✅ No saved test sets found - no mismatch possible")
        return
    
    print(f"\n📦 Found {len(test_set_files)} saved test set files")
    
    # Check the most likely to be used (best_trial or latest)
    priority_files = [
        saved_test_sets_dir / "test_set_best_trial.pkl",
        saved_test_sets_dir / "cicids_test_set_trial_49.pkl",  # Latest CICIDS
        saved_test_sets_dir / "test_set_trial_44.pkl",  # Latest generic
    ]
    
    mismatches_found = []
    
    for test_file in priority_files:
        if test_file.exists():
            print(f"\n🔍 Checking: {test_file.name}")
            try:
                with open(test_file, 'rb') as f:
                    saved_data = pickle.load(f)
                
                saved_zero_day = saved_data.get('zero_day_attack', 'UNKNOWN')
                saved_dataset = "CICIDS2017" if "cicids" in test_file.name.lower() else "UNKNOWN"
                
                # Try to infer dataset from data path if available
                if 'X_test_original' in saved_data:
                    # Check feature count to infer dataset
                    if hasattr(saved_data['X_test_original'], 'shape'):
                        feature_count = saved_data['X_test_original'].shape[1] if len(saved_data['X_test_original'].shape) > 1 else 0
                        if feature_count == 78:
                            saved_dataset = "CICIDS2017"
                        elif feature_count == 43:
                            saved_dataset = "UNSW"
                
                print(f"   Saved Zero-Day Attack: {saved_zero_day}")
                print(f"   Saved Dataset: {saved_dataset}")
                
                # Check for mismatch
                if saved_zero_day != current_zero_day:
                    print(f"   ❌ MISMATCH: Zero-day attack mismatch!")
                    print(f"      Current: {current_zero_day}, Saved: {saved_zero_day}")
                    mismatches_found.append({
                        'file': test_file.name,
                        'type': 'zero_day_attack',
                        'current': current_zero_day,
                        'saved': saved_zero_day
                    })
                
                if saved_dataset != "UNKNOWN" and saved_dataset != current_dataset:
                    print(f"   ❌ MISMATCH: Dataset mismatch!")
                    print(f"      Current: {current_dataset}, Saved: {saved_dataset}")
                    mismatches_found.append({
                        'file': test_file.name,
                        'type': 'dataset',
                        'current': current_dataset,
                        'saved': saved_dataset
                    })
                
                if saved_zero_day == current_zero_day and (saved_dataset == "UNKNOWN" or saved_dataset == current_dataset):
                    print(f"   ✅ No mismatch detected")
                    
            except Exception as e:
                print(f"   ⚠️  Error reading file: {str(e)}")
    
    # Summary
    print("\n" + "=" * 80)
    if mismatches_found:
        print("❌ MISMATCHES FOUND:")
        for mismatch in mismatches_found:
            print(f"   File: {mismatch['file']}")
            print(f"   Type: {mismatch['type']}")
            print(f"   Current: {mismatch['current']}, Saved: {mismatch['saved']}")
        print("\n⚠️  RECOMMENDATION:")
        print("   The system should detect this mismatch and create a new test set.")
        print("   However, to be safe, you may want to:")
        print("   1. Delete or rename saved test sets that don't match")
        print("   2. Or ensure the validation logic in main.py is working correctly")
    else:
        print("✅ NO MISMATCHES DETECTED")
        print("   Saved test sets match current configuration")

if __name__ == "__main__":
    check_saved_test_sets()



