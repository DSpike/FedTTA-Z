#!/usr/bin/env python3
"""
Test the updated preprocessing that rebalances the complete dataset first
"""

import sys
import os
import torch
import numpy as np
from collections import Counter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

def test_rebalanced_complete_dataset():
    """Test the rebalancing on the complete dataset"""
    
    print("=" * 80)
    print("TESTING REBALANCING ON COMPLETE DATASET")
    print("=" * 80)
    
    # Initialize preprocessor and get rebalanced data
    print("\n1. Loading and preprocessing UNSW-NB15 dataset with complete dataset rebalancing...")
    preprocessor = UNSWPreprocessor()
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    # Get the data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    attack_types = data['attack_types']
    
    print(f"\n2. Dataset Statistics After Complete Dataset Rebalancing:")
    print(f"   Training samples: {len(y_train):,}")
    print(f"   Validation samples: {len(y_val):,}")
    print(f"   Test samples: {len(y_test):,}")
    
    # Analyze class distribution after rebalancing
    print(f"\n3. Class Distribution Analysis:")
    
    sets = {
        "Training Set (Rebalanced)": y_train,
        "Validation Set (Rebalanced)": y_val,
        "Test Set (Rebalanced)": y_test
    }
    
    for name, labels in sets.items():
        counts = Counter(labels.numpy())
        total_samples = len(labels)
        print(f"\n   {name}:")
        
        for label_id in sorted(counts.keys()):
            attack_name = attack_types.get(label_id, f"Unknown ({label_id})")
            count = counts[label_id]
            percentage = (count / total_samples) * 100
            print(f"     {attack_name} (Label {label_id}): {count:,} samples ({percentage:.2f}%)")
        
        if len(counts) > 1:
            max_count = max(counts.values())
            min_count = min(counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"     Imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
        else:
            print("     Imbalance ratio: N/A (only one class)")
    
    # Check specifically for minority classes
    print(f"\n4. Minority Classes Check:")
    minority_classes = ['Worms', 'Analysis', 'Backdoor', 'Shellcode']
    
    for name, labels in sets.items():
        counts = Counter(labels.numpy())
        print(f"\n   {name} - Minority Classes:")
        for label_id in sorted(counts.keys()):
            attack_name = attack_types.get(label_id, f"Unknown ({label_id})")
            if attack_name in minority_classes:
                count = counts[label_id]
                percentage = (count / len(labels)) * 100
                print(f"     {attack_name} (Label {label_id}): {count:,} samples ({percentage:.2f}%)")
    
    print("\n" + "="*80)
    print("COMPLETE DATASET REBALANCING TEST COMPLETE")
    print("="*80)
    
    return data

if __name__ == "__main__":
    data = test_rebalanced_complete_dataset()



