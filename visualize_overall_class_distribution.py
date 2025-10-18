#!/usr/bin/env python3
"""
Overall Class Distribution Visualization
Shows the class distribution of the entire UNSW-NB15 dataset
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

def visualize_overall_class_distribution():
    """Visualize overall class distribution of the UNSW-NB15 dataset"""
    
    print("=" * 80)
    print("UNSW-NB15 OVERALL CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Load and preprocess data
    print("\n1. Loading UNSW-NB15 dataset...")
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    # Get the original data before splitting
    print("\n2. Analyzing original dataset...")
    
    # Load the raw data directly
    import pandas as pd
    
    # Load training and test data
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    
    # Combine all data
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"   Total samples: {len(all_data):,}")
    print(f"   Features: {all_data.shape[1]}")
    
    # Get attack type mapping
    attack_types = data['attack_types']
    print(f"\n   Attack type mapping: {attack_types}")
    
    # Analyze class distribution
    label_counts = Counter(all_data['attack_cat'])
    total_samples = len(all_data)
    
    print(f"\n3. Class Distribution Analysis:")
    print(f"   Total samples: {total_samples:,}")
    
    # Convert to percentages
    label_percentages = {label: (count/total_samples)*100 for label, count in label_counts.items()}
    
    # Print detailed distribution
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = label_percentages[label]
        print(f"     {label}: {count:,} samples ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('UNSW-NB15 Dataset Overall Class Distribution', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of sample counts by class
    ax1 = axes[0, 0]
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # Sort by count for better visualization
    sorted_indices = np.argsort(counts)[::-1]
    labels_sorted = [labels[i] for i in sorted_indices]
    counts_sorted = [counts[i] for i in sorted_indices]
    
    bars = ax1.bar(range(len(labels_sorted)), counts_sorted, alpha=0.8)
    ax1.set_xlabel('Attack Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Counts by Class')
    ax1.set_xticks(range(len(labels_sorted)))
    ax1.set_xticklabels(labels_sorted, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts_sorted):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # 2. Pie chart of class distribution
    ax2 = axes[0, 1]
    ax2.pie(counts_sorted, labels=labels_sorted, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Pie Chart)')
    
    # 3. Log scale bar plot to show imbalance
    ax3 = axes[1, 0]
    ax3.bar(range(len(labels_sorted)), counts_sorted, alpha=0.8)
    ax3.set_xlabel('Attack Classes (sorted by count)')
    ax3.set_ylabel('Number of Samples (Log Scale)')
    ax3.set_title('Class Imbalance (Log Scale)')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(labels_sorted)))
    ax3.set_xticklabels(labels_sorted, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Imbalance analysis
    ax4 = axes[1, 1]
    
    # Calculate imbalance metrics
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    cv = std_count / mean_count if mean_count > 0 else 0
    
    # Create a bar chart showing the imbalance
    imbalance_data = {
        'Max Class': max_count,
        'Min Class': min_count,
        'Mean Class': mean_count,
        'Median Class': np.median(counts)
    }
    
    bars = ax4.bar(imbalance_data.keys(), imbalance_data.values(), alpha=0.8, color=['red', 'blue', 'green', 'orange'])
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Class Size Statistics')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, imbalance_data.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'overall_class_distribution_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Detailed imbalance analysis
    print(f"\n5. Detailed Imbalance Analysis:")
    print(f"     Maximum class size: {max_count:,}")
    print(f"     Minimum class size: {min_count:,}")
    print(f"     Mean class size: {mean_count:.1f}")
    print(f"     Median class size: {np.median(counts):.1f}")
    print(f"     Standard deviation: {std_count:.1f}")
    print(f"     Coefficient of variation: {cv:.3f}")
    print(f"     Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Classify imbalance level
    if imbalance_ratio < 2:
        imbalance_level = "Balanced"
    elif imbalance_ratio < 10:
        imbalance_level = "Moderately Imbalanced"
    elif imbalance_ratio < 100:
        imbalance_level = "Highly Imbalanced"
    else:
        imbalance_level = "Extremely Imbalanced"
    
    print(f"     Imbalance level: {imbalance_level}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return label_counts, label_percentages

if __name__ == "__main__":
    counts, percentages = visualize_overall_class_distribution()
