#!/usr/bin/env python3
"""
Test the updated preprocessing with data rebalancing
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

def test_rebalancing():
    """Test the data rebalancing functionality"""
    
    print("=" * 80)
    print("TESTING DATA REBALANCING FUNCTIONALITY")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Load and preprocess data with rebalancing
    print("\n1. Loading and preprocessing UNSW-NB15 dataset with rebalancing...")
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    # Get the data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    attack_types = data['attack_types']
    
    print(f"\n2. Dataset Statistics After Rebalancing:")
    print(f"   Training samples: {len(y_train):,}")
    print(f"   Validation samples: {len(y_val):,}")
    print(f"   Test samples: {len(y_test):,}")
    
    # Analyze class distribution after rebalancing
    print(f"\n3. Class Distribution Analysis:")
    
    sets = {
        "Training Set (Rebalanced)": y_train,
        "Validation Set": y_val,
        "Test Set": y_test
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
    
    # Create visualization
    print(f"\n4. Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('UNSW-NB15 Class Distribution After Rebalancing', fontsize=16, fontweight='bold')
    
    for i, (name, labels) in enumerate(sets.items()):
        counts = Counter(labels.numpy())
        sorted_labels = sorted(counts.keys())
        attack_names = [attack_types.get(l, f"Unknown ({l})") for l in sorted_labels]
        sample_counts = [counts[l] for l in sorted_labels]
        
        bars = axes[i].bar(range(len(attack_names)), sample_counts, alpha=0.8)
        axes[i].set_title(name)
        axes[i].set_xlabel('Attack Classes')
        axes[i].set_ylabel('Number of Samples')
        axes[i].set_xticks(range(len(attack_names)))
        axes[i].set_xticklabels(attack_names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, sample_counts):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 100,
                        f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'rebalanced_class_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("REBALANCING TEST COMPLETE")
    print("="*80)
    
    return data

if __name__ == "__main__":
    data = test_rebalancing()



