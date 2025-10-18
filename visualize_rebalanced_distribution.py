#!/usr/bin/env python3
"""
Visualize the class distribution after SMOTE and downsampling
Shows the rebalanced dataset from the preprocessing pipeline
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

def visualize_rebalanced_distribution():
    """Visualize the class distribution after SMOTE and downsampling"""
    
    print("=" * 80)
    print("UNSW-NB15 REBALANCED CLASS DISTRIBUTION VISUALIZATION")
    print("=" * 80)
    
    # Initialize preprocessor and get rebalanced data
    print("\n1. Loading UNSW-NB15 dataset with rebalancing...")
    preprocessor = UNSWPreprocessor()
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    # Get the rebalanced training data
    X_train_rebalanced = data['X_train']
    y_train_rebalanced = data['y_train']
    attack_types = data['attack_types']
    
    print(f"   Rebalanced training samples: {len(y_train_rebalanced):,}")
    print(f"   Number of classes: {len(attack_types)}")
    
    # Analyze class distribution after rebalancing
    print(f"\n2. Rebalanced Class Distribution Analysis:")
    counts = Counter(y_train_rebalanced.numpy())
    total_samples = len(y_train_rebalanced)
    
    # Sort by count for better visualization
    sorted_labels = sorted(counts.keys())
    attack_names = [attack_types.get(label, f"Unknown ({label})") for label in sorted_labels]
    sample_counts = [counts[label] for count, label in zip([counts[l] for l in sorted_labels], sorted_labels)]
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Number of classes: {len(counts)}")
    
    # Print detailed distribution
    for label, count in zip(sorted_labels, sample_counts):
        attack_name = attack_types.get(label, f"Unknown ({label})")
        percentage = (count / total_samples) * 100
        print(f"     {attack_name} (Label {label}): {count:,} samples ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    max_count = max(sample_counts)
    min_count = min(sample_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
    
    # Create comprehensive visualizations
    print(f"\n3. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('UNSW-NB15 Dataset After SMOTE and Downsampling (Rebalanced)', fontsize=20, fontweight='bold')
    
    # 1. Main bar plot (top-left)
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(range(len(attack_names)), sample_counts, alpha=0.8, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(attack_names))))
    ax1.set_xlabel('Attack Categories', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Rebalanced Sample Counts by Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(attack_names)))
    ax1.set_xticklabels(attack_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Pie chart (top-center)
    ax2 = plt.subplot(2, 3, 2)
    wedges, texts, autotexts = ax2.pie(sample_counts, labels=attack_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Rebalanced Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 3. Log scale bar plot (top-right)
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(range(len(attack_names)), sample_counts, alpha=0.8, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(attack_names))))
    ax3.set_xlabel('Attack Categories (sorted by count)', fontsize=12)
    ax3.set_ylabel('Number of Samples (Log Scale)', fontsize=12)
    ax3.set_title('Rebalanced Class Imbalance (Log Scale)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(attack_names)))
    ax3.set_xticklabels(attack_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Horizontal bar plot (bottom-left)
    ax4 = plt.subplot(2, 3, 4)
    y_pos = np.arange(len(attack_names))
    bars4 = ax4.barh(y_pos, sample_counts, alpha=0.8, 
                     color=plt.cm.plasma(np.linspace(0, 1, len(attack_names))))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(attack_names)
    ax4.set_xlabel('Number of Samples', fontsize=12)
    ax4.set_title('Rebalanced Horizontal Bar Chart', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars4, sample_counts)):
        width = bar.get_width()
        ax4.text(width + 200, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 5. Imbalance analysis (bottom-center)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate statistics
    mean_count = np.mean(sample_counts)
    median_count = np.median(sample_counts)
    std_count = np.std(sample_counts)
    cv = std_count / mean_count if mean_count > 0 else 0
    
    stats_data = {
        'Max Class': max_count,
        'Min Class': min_count,
        'Mean Class': mean_count,
        'Median Class': median_count
    }
    
    bars5 = ax5.bar(stats_data.keys(), stats_data.values(), alpha=0.8, 
                    color=['red', 'blue', 'green', 'orange'])
    ax5.set_ylabel('Number of Samples', fontsize=12)
    ax5.set_title('Rebalanced Class Size Statistics', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars5, stats_data.values()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Percentage distribution (bottom-right)
    ax6 = plt.subplot(2, 3, 6)
    percentages = [(count / total_samples) * 100 for count in sample_counts]
    
    bars6 = ax6.bar(range(len(attack_names)), percentages, alpha=0.8, 
                    color=plt.cm.coolwarm(np.linspace(0, 1, len(attack_names))))
    ax6.set_xlabel('Attack Categories', fontsize=12)
    ax6.set_ylabel('Percentage of Total Samples (%)', fontsize=12)
    ax6.set_title('Rebalanced Percentage Distribution', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(attack_names)))
    ax6.set_xticklabels(attack_names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars6, percentages):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'rebalanced_class_distribution_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Detailed imbalance analysis
    print(f"\n4. Detailed Rebalanced Imbalance Analysis:")
    print(f"     Maximum class size: {max_count:,}")
    print(f"     Minimum class size: {min_count:,}")
    print(f"     Mean class size: {mean_count:.1f}")
    print(f"     Median class size: {median_count:.1f}")
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
    
    # Show improvement from original
    print(f"\n5. Improvement from Original Dataset:")
    print(f"     Original imbalance ratio: 534.48:1")
    print(f"     Rebalanced imbalance ratio: {imbalance_ratio:.2f}:1")
    improvement = (534.48 - imbalance_ratio) / 534.48 * 100
    print(f"     Imbalance reduction: {improvement:.1f}%")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("REBALANCED VISUALIZATION COMPLETE")
    print("="*80)
    
    return counts, sorted_labels, attack_names, sample_counts

if __name__ == "__main__":
    counts, sorted_labels, attack_names, sample_counts = visualize_rebalanced_distribution()



