#!/usr/bin/env python3
"""
Class Distribution Visualization
Analyzes and visualizes the class distribution in the UNSW-NB15 dataset
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

def visualize_class_distribution():
    """Visualize class distribution in the UNSW-NB15 dataset"""
    
    print("=" * 80)
    print("UNSW-NB15 CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing UNSW-NB15 dataset...")
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"   Training data: {X_train.shape[0]} samples")
    print(f"   Validation data: {X_val.shape[0]} samples")
    print(f"   Test data: {X_test.shape[0]} samples")
    
    # Get attack type mapping
    attack_types = data['attack_types']
    print(f"\n   Attack type mapping: {attack_types}")
    
    # Convert tensors to numpy for analysis
    y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
    y_val_np = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val
    y_test_np = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
    
    # Create reverse mapping for labels
    label_to_name = {v: k for k, v in attack_types.items()}
    
    print("\n2. Class Distribution Analysis:")
    
    # Analyze each dataset split
    datasets = {
        'Training': y_train_np,
        'Validation': y_val_np,
        'Test': y_test_np
    }
    
    all_distributions = {}
    
    for split_name, labels in datasets.items():
        print(f"\n   {split_name} Set:")
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        # Convert to percentages
        label_percentages = {label: (count/total_samples)*100 for label, count in label_counts.items()}
        
        # Store for visualization
        all_distributions[split_name] = {
            'counts': label_counts,
            'percentages': label_percentages,
            'total': total_samples
        }
        
        # Print detailed distribution
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = label_percentages[label]
            class_name = label_to_name.get(label, f"Unknown_{label}")
            print(f"     {class_name} (Label {label}): {count:,} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"     Imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
    
    # Create comprehensive visualizations
    print("\n3. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('UNSW-NB15 Dataset Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of sample counts by class
    ax1 = axes[0, 0]
    all_labels = sorted(set(y_train_np) | set(y_val_np) | set(y_test_np))
    class_names = [label_to_name.get(label, f"Unknown_{label}") for label in all_labels]
    
    train_counts = [all_distributions['Training']['counts'].get(label, 0) for label in all_labels]
    val_counts = [all_distributions['Validation']['counts'].get(label, 0) for label in all_labels]
    test_counts = [all_distributions['Test']['counts'].get(label, 0) for label in all_labels]
    
    x = np.arange(len(all_labels))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Training', alpha=0.8)
    ax1.bar(x, val_counts, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('Attack Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Counts by Class and Dataset Split')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pie chart of training set distribution
    ax2 = axes[0, 1]
    train_labels = list(all_distributions['Training']['counts'].keys())
    train_counts_list = list(all_distributions['Training']['counts'].values())
    train_class_names = [label_to_name.get(label, f"Unknown_{label}") for label in train_labels]
    
    # Sort by count for better visualization
    sorted_indices = np.argsort(train_counts_list)[::-1]
    train_labels_sorted = [train_labels[i] for i in sorted_indices]
    train_counts_sorted = [train_counts_list[i] for i in sorted_indices]
    train_class_names_sorted = [train_class_names[i] for i in sorted_indices]
    
    ax2.pie(train_counts_sorted, labels=train_class_names_sorted, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Training Set Class Distribution')
    
    # 3. Log scale bar plot to show imbalance
    ax3 = axes[1, 0]
    ax3.bar(range(len(train_class_names_sorted)), train_counts_sorted, alpha=0.8)
    ax3.set_xlabel('Attack Classes (sorted by count)')
    ax3.set_ylabel('Number of Samples (Log Scale)')
    ax3.set_title('Class Imbalance in Training Set (Log Scale)')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(train_class_names_sorted)))
    ax3.set_xticklabels(train_class_names_sorted, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Imbalance ratio analysis
    ax4 = axes[1, 1]
    imbalance_ratios = []
    for split_name, dist in all_distributions.items():
        counts = list(dist['counts'].values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            ratio = max_count / min_count if min_count > 0 else float('inf')
            imbalance_ratios.append(ratio)
        else:
            imbalance_ratios.append(0)
    
    splits = list(all_distributions.keys())
    bars = ax4.bar(splits, imbalance_ratios, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax4.set_ylabel('Imbalance Ratio (Max/Min)')
    ax4.set_title('Dataset Imbalance by Split')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, imbalance_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.1f}:1', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'class_distribution_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Create a detailed imbalance analysis
    print("\n4. Detailed Imbalance Analysis:")
    
    for split_name, dist in all_distributions.items():
        counts = list(dist['counts'].values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            cv = std_count / mean_count if mean_count > 0 else 0
            
            print(f"\n   {split_name} Set Imbalance Metrics:")
            print(f"     Maximum class size: {max_count:,}")
            print(f"     Minimum class size: {min_count:,}")
            print(f"     Mean class size: {mean_count:.1f}")
            print(f"     Standard deviation: {std_count:.1f}")
            print(f"     Coefficient of variation: {cv:.3f}")
            print(f"     Imbalance ratio: {max_count/min_count:.2f}:1")
            
            # Classify imbalance level
            if max_count/min_count < 2:
                imbalance_level = "Balanced"
            elif max_count/min_count < 10:
                imbalance_level = "Moderately Imbalanced"
            elif max_count/min_count < 100:
                imbalance_level = "Highly Imbalanced"
            else:
                imbalance_level = "Extremely Imbalanced"
            
            print(f"     Imbalance level: {imbalance_level}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return all_distributions

if __name__ == "__main__":
    distributions = visualize_class_distribution()



