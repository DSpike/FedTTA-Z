#!/usr/bin/env python3
"""
Visualize the overall class distribution of the UNSW-NB15 dataset
Shows the complete dataset without train/validation/test splits
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def visualize_dataset_class_distribution():
    """Visualize the overall class distribution of the UNSW-NB15 dataset"""
    
    print("=" * 80)
    print("UNSW-NB15 DATASET CLASS DISTRIBUTION VISUALIZATION")
    print("=" * 80)
    
    # Load the raw UNSW-NB15 datasets
    print("\n1. Loading UNSW-NB15 datasets...")
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    
    # Combine all data
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"   Total samples: {len(all_data):,}")
    print(f"   Features: {all_data.shape[1]}")
    
    # Get attack categories
    attack_categories = all_data['attack_cat'].unique()
    print(f"   Attack categories: {len(attack_categories)}")
    print(f"   Categories: {list(attack_categories)}")
    
    # Analyze class distribution
    print(f"\n2. Class Distribution Analysis:")
    label_counts = Counter(all_data['attack_cat'])
    total_samples = len(all_data)
    
    # Sort by count for better visualization
    sorted_categories = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Number of classes: {len(label_counts)}")
    
    # Print detailed distribution
    for category, count in sorted_categories:
        percentage = (count / total_samples) * 100
        print(f"     {category}: {count:,} samples ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
    
    # Create comprehensive visualizations
    print(f"\n3. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('UNSW-NB15 Dataset Complete Class Distribution Analysis', fontsize=20, fontweight='bold')
    
    # 1. Main bar plot (top-left)
    ax1 = plt.subplot(2, 3, 1)
    categories = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]
    
    bars = ax1.bar(range(len(categories)), counts, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
    ax1.set_xlabel('Attack Categories', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Sample Counts by Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Pie chart (top-center)
    ax2 = plt.subplot(2, 3, 2)
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 3. Log scale bar plot (top-right)
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(range(len(categories)), counts, alpha=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
    ax3.set_xlabel('Attack Categories (sorted by count)', fontsize=12)
    ax3.set_ylabel('Number of Samples (Log Scale)', fontsize=12)
    ax3.set_title('Class Imbalance (Log Scale)', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Horizontal bar plot (bottom-left)
    ax4 = plt.subplot(2, 3, 4)
    y_pos = np.arange(len(categories))
    bars4 = ax4.barh(y_pos, counts, alpha=0.8, color=plt.cm.plasma(np.linspace(0, 1, len(categories))))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(categories)
    ax4.set_xlabel('Number of Samples', fontsize=12)
    ax4.set_title('Horizontal Bar Chart', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars4, counts)):
        width = bar.get_width()
        ax4.text(width + 1000, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 5. Imbalance analysis (bottom-center)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate statistics
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    std_count = np.std(counts)
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
    ax5.set_title('Class Size Statistics', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars5, stats_data.values()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Percentage distribution (bottom-right)
    ax6 = plt.subplot(2, 3, 6)
    percentages = [(count / total_samples) * 100 for count in counts]
    
    bars6 = ax6.bar(range(len(categories)), percentages, alpha=0.8, 
                    color=plt.cm.coolwarm(np.linspace(0, 1, len(categories))))
    ax6.set_xlabel('Attack Categories', fontsize=12)
    ax6.set_ylabel('Percentage of Total Samples (%)', fontsize=12)
    ax6.set_title('Percentage Distribution', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(categories)))
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars6, percentages):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'dataset_class_distribution_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Detailed imbalance analysis
    print(f"\n4. Detailed Imbalance Analysis:")
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
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    
    return label_counts, sorted_categories

if __name__ == "__main__":
    counts, sorted_categories = visualize_dataset_class_distribution()



