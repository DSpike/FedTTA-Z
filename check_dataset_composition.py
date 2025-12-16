"""
Quick script to check validation and test set composition
"""

import sys
import numpy as np
import torch
from config import SystemConfig
from main import BlockchainFederatedIncentiveSystem
import logging

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)

def check_composition():
    """Check validation and test set composition"""
    
    print("="*80)
    print("DATASET COMPOSITION CHECK")
    print("="*80)
    
    try:
        config = SystemConfig()
        system = BlockchainFederatedIncentiveSystem(config)
        
        # Initialize system properly
        system._initialize_system()
        
        # Preprocess data
        print("\n⏳ Preprocessing data...")
        success = system.preprocess_data()
        
        if not success:
            print("❌ Data preprocessing failed")
            return
        
        preprocessed = system.preprocessed_data
        
        # Check Validation Set
        print("\n" + "="*80)
        print("VALIDATION SET COMPOSITION")
        print("="*80)
        
        if 'y_val' in preprocessed:
            y_val = preprocessed['y_val']
            if torch.is_tensor(y_val):
                y_val_np = y_val.cpu().numpy()
            else:
                y_val_np = np.array(y_val)
            
            total_val = len(y_val_np)
            normal_val = np.sum(y_val_np == 0)
            attack_val = total_val - normal_val
            
            print(f"\n📊 Binary Labels:")
            print(f"   Total samples: {total_val:,}")
            print(f"   Normal (BENIGN): {normal_val:,} ({100*normal_val/total_val:.1f}%)")
            print(f"   Attacks: {attack_val:,} ({100*attack_val/total_val:.1f}%)")
            
            # Check multiclass if available
            if 'y_val_multiclass' in preprocessed:
                y_val_mc = preprocessed['y_val_multiclass']
                if torch.is_tensor(y_val_mc):
                    y_val_mc_np = y_val_mc.cpu().numpy()
                else:
                    y_val_mc_np = np.array(y_val_mc)
                
                unique_labels, counts = np.unique(y_val_mc_np, return_counts=True)
                label_dist = dict(zip(unique_labels, counts))
                
                print(f"\n📊 Multiclass Labels (Attack Types):")
                print(f"   Unique labels: {sorted(unique_labels)}")
                print(f"   Label distribution:")
                for label in sorted(unique_labels):
                    count = label_dist[label]
                    pct = 100 * count / total_val
                    label_name = "BENIGN" if label == 0 else f"Attack_{label}"
                    if label == config.zero_day_attack_label:
                        print(f"      Label {label} ({label_name}) [ZERO-DAY]: {count:,} ({pct:.1f}%) ⭐")
                    else:
                        print(f"      Label {label} ({label_name}): {count:,} ({pct:.1f}%)")
                
                # Check for zero-day (should be 0 in validation)
                zero_day_in_val = label_dist.get(config.zero_day_attack_label, 0)
                if zero_day_in_val > 0:
                    print(f"\n   ⚠️  WARNING: Found {zero_day_in_val} zero-day samples in validation set!")
                    print(f"      Zero-day should be excluded from validation set.")
                else:
                    print(f"\n   ✅ Zero-day correctly excluded from validation set")
        
        # Check Test Set (BEFORE sequences)
        print("\n" + "="*80)
        print("TEST SET COMPOSITION (BEFORE SEQUENCE CREATION)")
        print("="*80)
        
        # Check original test subset (before sequences)
        if 'X_test_original' in preprocessed and 'y_test_original' in preprocessed:
            y_test_orig = preprocessed['y_test_original']
            if torch.is_tensor(y_test_orig):
                y_test_orig_np = y_test_orig.cpu().numpy()
            else:
                y_test_orig_np = np.array(y_test_orig)
            
            total_test = len(y_test_orig_np)
            normal_test = np.sum(y_test_orig_np == 0)
            attack_test = total_test - normal_test
            
            print(f"\n📊 Binary Labels (Original Test Subset):")
            print(f"   Total samples: {total_test:,}")
            print(f"   Normal (BENIGN): {normal_test:,} ({100*normal_test/total_test:.1f}%)")
            print(f"   Attacks: {attack_test:,} ({100*attack_test/total_test:.1f}%)")
            
            # Check multiclass if available
            # Try to get original multiclass labels
            y_test_mc_orig = None
            if 'y_test_multiclass' in preprocessed:
                # This might be sequence-level, check if it matches original size
                y_test_mc_seq = preprocessed['y_test_multiclass']
                if torch.is_tensor(y_test_mc_seq):
                    y_test_mc_seq_np = y_test_mc_seq.cpu().numpy()
                else:
                    y_test_mc_seq_np = np.array(y_test_mc_seq)
                
                # Check if it's sequence-level or original
                if len(y_test_mc_seq_np) == total_test:
                    y_test_mc_orig = y_test_mc_seq_np
                else:
                    print(f"\n   ⚠️  y_test_multiclass is sequence-level ({len(y_test_mc_seq_np)} sequences), not original ({total_test} samples)")
            
            if y_test_mc_orig is not None:
                unique_labels, counts = np.unique(y_test_mc_orig, return_counts=True)
                label_dist = dict(zip(unique_labels, counts))
                
                print(f"\n📊 Multiclass Labels (Original Test Subset):")
                print(f"   Unique labels: {sorted(unique_labels)}")
                print(f"   Label distribution:")
                for label in sorted(unique_labels):
                    count = label_dist[label]
                    pct = 100 * count / total_test
                    label_name = "BENIGN" if label == 0 else f"Attack_{label}"
                    if label == config.zero_day_attack_label:
                        print(f"      Label {label} ({label_name}) [ZERO-DAY]: {count:,} ({pct:.1f}%) ⭐")
                    else:
                        print(f"      Label {label} ({label_name}): {count:,} ({pct:.1f}%)")
                
                # Calculate composition
                zero_day_count = label_dist.get(config.zero_day_attack_label, 0)
                normal_count = label_dist.get(0, 0)
                known_attack_count = total_test - zero_day_count - normal_count
                
                print(f"\n📊 Composition Summary:")
                print(f"   Normal (BENIGN): {normal_count:,} ({100*normal_count/total_test:.1f}%)")
                print(f"   Zero-day ({config.zero_day_attack}): {zero_day_count:,} ({100*zero_day_count/total_test:.1f}%)")
                print(f"   Known attacks: {known_attack_count:,} ({100*known_attack_count/total_test:.1f}%)")
        
        # Check Test Set (AFTER sequences and filtering)
        print("\n" + "="*80)
        print("TEST SET COMPOSITION (AFTER SEQUENCE CREATION & FILTERING)")
        print("="*80)
        
        if 'X_test' in preprocessed and 'y_test' in preprocessed:
            X_test = preprocessed['X_test']
            y_test = preprocessed['y_test']
            
            if torch.is_tensor(y_test):
                y_test_np = y_test.cpu().numpy()
            else:
                y_test_np = np.array(y_test)
            
            total_test_seq = len(y_test_np)
            normal_test_seq = np.sum(y_test_np == 0)
            attack_test_seq = total_test_seq - normal_test_seq
            
            print(f"\n📊 Binary Labels (Sequences):")
            print(f"   Total sequences: {total_test_seq:,}")
            print(f"   Normal (BENIGN): {normal_test_seq:,} ({100*normal_test_seq/total_test_seq:.1f}%)")
            print(f"   Attacks: {attack_test_seq:,} ({100*attack_test_seq/total_test_seq:.1f}%)")
            
            # Check multiclass sequences
            if 'y_test_multiclass' in preprocessed:
                y_test_mc = preprocessed['y_test_multiclass']
                if torch.is_tensor(y_test_mc):
                    y_test_mc_np = y_test_mc.cpu().numpy()
                else:
                    y_test_mc_np = np.array(y_test_mc)
                
                if len(y_test_mc_np) == total_test_seq:
                    unique_labels, counts = np.unique(y_test_mc_np, return_counts=True)
                    label_dist = dict(zip(unique_labels, counts))
                    
                    print(f"\n📊 Multiclass Labels (Sequences):")
                    print(f"   Unique labels: {sorted(unique_labels)}")
                    print(f"   Label distribution:")
                    for label in sorted(unique_labels):
                        count = label_dist[label]
                        pct = 100 * count / total_test_seq
                        label_name = "BENIGN" if label == 0 else f"Attack_{label}"
                        if label == config.zero_day_attack_label:
                            print(f"      Label {label} ({label_name}) [ZERO-DAY]: {count:,} ({pct:.1f}%) ⭐")
                        else:
                            print(f"      Label {label} ({label_name}): {count:,} ({pct:.1f}%)")
                    
                    # Calculate composition
                    zero_day_count = label_dist.get(config.zero_day_attack_label, 0)
                    normal_count = label_dist.get(0, 0)
                    known_attack_count = total_test_seq - zero_day_count - normal_count
                    
                    print(f"\n📊 Composition Summary:")
                    print(f"   Normal (BENIGN): {normal_count:,} ({100*normal_count/total_test_seq:.1f}%)")
                    print(f"   Zero-day ({config.zero_day_attack}): {zero_day_count:,} ({100*zero_day_count/total_test_seq:.1f}%)")
                    print(f"   Known attacks: {known_attack_count:,} ({100*known_attack_count/total_test_seq:.1f}%)")
                    
                    # Target composition
                    print(f"\n🎯 Target Composition:")
                    print(f"   Normal: 60%")
                    print(f"   Known attacks: 30%")
                    print(f"   Zero-day: 10%")
        
        print("\n" + "="*80)
        print("COMPOSITION CHECK COMPLETE")
        print("="*80)
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    check_composition()

