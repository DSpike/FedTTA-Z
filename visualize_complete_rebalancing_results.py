#!/usr/bin/env python3
"""
Visualize the complete dataset rebalancing results
Shows before and after class distribution with focus on minority classes
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

def visualize_complete_rebalancing_results():
    """Visualize the complete dataset rebalancing results"""
    
    print("=" * 80)
    print("COMPLETE DATASET REBALANCING VISUALIZATION")
    print("=" * 80)
    
    # Load original dataset for comparison
    print("\n1. Loading original UNSW-NB15 dataset...")
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    original_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # Original class distribution
    original_counts = Counter(original_data['attack_cat'])
    original_total = len(original_data)
    
    print(f"   Original dataset: {original_total:,} samples")
    
    # Simulate the rebalanced distribution based on the test results
    rebalanced_counts = {
        'Normal': 20000,
        'Fuzzers': 20000, 
        'Analysis': 10000,  # Minority class - was 2,677
        'Backdoor': 10000,  # Minority class - was 2,329
        'DoS': 16353,
        'Exploits': 20000,
        'Generic': 20000,
        'Reconnaissance': 13987,
        'Shellcode': 10000,  # Minority class - was 1,511
        'Worms': 10000       # Minority class - was 174
    }
    
    rebalanced_total = sum(rebalanced_counts.values())
    
    print(f"   Rebalanced dataset: {rebalanced_total:,} samples")
    
    # Create comprehensive visualization
    print(f"\n2. Creating visualization...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('UNSW-NB15 Complete Dataset: Before vs After Rebalancing\nFocus on Minority Classes (Worms, Analysis, Backdoor, Shellcode)', 
                 fontsize=20, fontweight='bold')
    
    # Prepare data for visualization
    categories = list(original_counts.keys())
    original_counts_list = [original_counts[cat] for cat in categories]
    rebalanced_counts_list = [rebalanced_counts[cat] for cat in categories]
    
    # Identify minority classes for highlighting
    minority_classes = ['Worms', 'Analysis', 'Backdoor', 'Shellcode']
    minority_indices = [i for i, cat in enumerate(categories) if cat in minority_classes]
    
    # 1. Original dataset bar plot (top-left)
    ax1 = plt.subplot(3, 3, 1)
    bars1 = ax1.bar(range(len(categories)), original_counts_list, alpha=0.8, 
                    color=['red' if cat in minority_classes else 'lightblue' for cat in categories])
    ax1.set_xlabel('Attack Categories', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Original Dataset\n(Minority Classes in Red)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars1, original_counts_list)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Rebalanced dataset bar plot (top-center)
    ax2 = plt.subplot(3, 3, 2)
    bars2 = ax2.bar(range(len(categories)), rebalanced_counts_list, alpha=0.8, 
                    color=['red' if cat in minority_classes else 'lightgreen' for cat in categories])
    ax2.set_xlabel('Attack Categories', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Rebalanced Dataset\n(Minority Classes in Red)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars2, rebalanced_counts_list)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Side-by-side comparison (top-right)
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(len(categories))
    width = 0.35
    
    bars3_orig = ax3.bar(x - width/2, original_counts_list, width, label='Original', alpha=0.8, color='lightblue')
    bars3_rebal = ax3.bar(x + width/2, rebalanced_counts_list, width, label='Rebalanced', alpha=0.8, color='lightgreen')
    
    # Highlight minority classes
    for i in minority_indices:
        bars3_orig[i].set_color('red')
        bars3_rebal[i].set_color('red')
    
    ax3.set_xlabel('Attack Categories', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Side-by-Side Comparison\n(Minority Classes in Red)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Minority classes focus (middle-left)
    ax4 = plt.subplot(3, 3, 4)
    minority_original = [original_counts[cat] for cat in minority_classes]
    minority_rebalanced = [rebalanced_counts[cat] for cat in minority_classes]
    
    x_minority = np.arange(len(minority_classes))
    bars4_orig = ax4.bar(x_minority - width/2, minority_original, width, label='Original', alpha=0.8, color='red')
    bars4_rebal = ax4.bar(x_minority + width/2, minority_rebalanced, width, label='Rebalanced', alpha=0.8, color='darkred')
    
    ax4.set_xlabel('Minority Attack Categories', fontsize=12)
    ax4.set_ylabel('Number of Samples', fontsize=12)
    ax4.set_title('Minority Classes Focus\n(Worms, Analysis, Backdoor, Shellcode)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_minority)
    ax4.set_xticklabels(minority_classes, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars4_orig, minority_original):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, count in zip(bars4_rebal, minority_rebalanced):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Improvement ratios (middle-center)
    ax5 = plt.subplot(3, 3, 5)
    improvement_ratios = [rebalanced_counts[cat] / original_counts[cat] for cat in minority_classes]
    
    bars5 = ax5.bar(minority_classes, improvement_ratios, alpha=0.8, color='orange')
    ax5.set_xlabel('Minority Attack Categories', fontsize=12)
    ax5.set_ylabel('Improvement Ratio (x times)', fontsize=12)
    ax5.set_title('Minority Classes Improvement\n(How many times increased)', fontsize=14, fontweight='bold')
    ax5.set_xticklabels(minority_classes, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars5, improvement_ratios):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Log scale comparison (middle-right)
    ax6 = plt.subplot(3, 3, 6)
    bars6_orig = ax6.bar(x - width/2, original_counts_list, width, label='Original', alpha=0.8, color='lightblue')
    bars6_rebal = ax6.bar(x + width/2, rebalanced_counts_list, width, label='Rebalanced', alpha=0.8, color='lightgreen')
    
    # Highlight minority classes
    for i in minority_indices:
        bars6_orig[i].set_color('red')
        bars6_rebal[i].set_color('red')
    
    ax6.set_xlabel('Attack Categories', fontsize=12)
    ax6.set_ylabel('Number of Samples (Log Scale)', fontsize=12)
    ax6.set_title('Log Scale Comparison\n(Shows extreme differences)', fontsize=14, fontweight='bold')
    ax6.set_yscale('log')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Imbalance ratio comparison (bottom-left)
    ax7 = plt.subplot(3, 3, 7)
    
    # Calculate imbalance ratios
    original_max = max(original_counts_list)
    original_min = min(original_counts_list)
    original_imbalance = original_max / original_min
    
    rebalanced_max = max(rebalanced_counts_list)
    rebalanced_min = min(rebalanced_counts_list)
    rebalanced_imbalance = rebalanced_max / rebalanced_min
    
    imbalance_data = {
        'Original': original_imbalance,
        'Rebalanced': rebalanced_imbalance
    }
    
    bars7 = ax7.bar(imbalance_data.keys(), imbalance_data.values(), alpha=0.8, color=['red', 'green'])
    ax7.set_ylabel('Imbalance Ratio', fontsize=12)
    ax7.set_title('Imbalance Ratio Comparison', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars7, imbalance_data.values()):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value:.1f}:1', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 8. Sample count comparison (bottom-center)
    ax8 = plt.subplot(3, 3, 8)
    sample_data = {
        'Original': original_total,
        'Rebalanced': rebalanced_total
    }
    
    bars8 = ax8.bar(sample_data.keys(), sample_data.values(), alpha=0.8, color=['blue', 'orange'])
    ax8.set_ylabel('Total Samples', fontsize=12)
    ax8.set_title('Total Sample Count', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars8, sample_data.values()):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 9. Summary statistics (bottom-right)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate improvement statistics
    improvement_percentages = [(rebalanced_counts[cat] - original_counts[cat]) / original_counts[cat] * 100 
                              for cat in minority_classes]
    
    summary_text = f"""
    REBALANCING SUMMARY
    
    Original Dataset:
    • Total samples: {original_total:,}
    • Imbalance ratio: {original_imbalance:.1f}:1
    • Worms: {original_counts['Worms']:,} samples
    • Analysis: {original_counts['Analysis']:,} samples
    • Backdoor: {original_counts['Backdoor']:,} samples
    • Shellcode: {original_counts['Shellcode']:,} samples
    
    Rebalanced Dataset:
    • Total samples: {rebalanced_total:,}
    • Imbalance ratio: {rebalanced_imbalance:.1f}:1
    • Worms: {rebalanced_counts['Worms']:,} samples ({improvement_percentages[0]:+.0f}%)
    • Analysis: {rebalanced_counts['Analysis']:,} samples ({improvement_percentages[1]:+.0f}%)
    • Backdoor: {rebalanced_counts['Backdoor']:,} samples ({improvement_percentages[2]:+.0f}%)
    • Shellcode: {rebalanced_counts['Shellcode']:,} samples ({improvement_percentages[3]:+.0f}%)
    
    Improvement:
    • Imbalance reduction: {((original_imbalance - rebalanced_imbalance) / original_imbalance * 100):.1f}%
    • Sample change: {rebalanced_total - original_total:+,} samples
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'complete_dataset_rebalancing_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print detailed statistics
    print(f"\n3. Detailed Statistics:")
    print(f"   Original imbalance ratio: {original_imbalance:.1f}:1")
    print(f"   Rebalanced imbalance ratio: {rebalanced_imbalance:.1f}:1")
    print(f"   Imbalance reduction: {((original_imbalance - rebalanced_imbalance) / original_imbalance * 100):.1f}%")
    
    print(f"\n   Minority Classes Improvement:")
    for i, cat in enumerate(minority_classes):
        original_count = original_counts[cat]
        rebalanced_count = rebalanced_counts[cat]
        improvement = improvement_percentages[i]
        print(f"     {cat}: {original_count:,} → {rebalanced_count:,} ({improvement:+.0f}%)")
    
    print("\n" + "="*80)
    print("REBALANCING VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    visualize_complete_rebalancing_results()



