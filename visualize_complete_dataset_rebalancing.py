#!/usr/bin/env python3
"""
Visualize the complete UNSW-NB15 dataset class distribution and apply rebalancing
Shows the original dataset and then the rebalanced version
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def visualize_complete_dataset_rebalancing():
    """Visualize the complete dataset class distribution and rebalancing"""
    
    print("=" * 80)
    print("UNSW-NB15 COMPLETE DATASET CLASS DISTRIBUTION & REBALANCING")
    print("=" * 80)
    
    # Load the complete UNSW-NB15 dataset
    print("\n1. Loading complete UNSW-NB15 dataset...")
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    
    # Combine all data
    complete_data = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"   Total samples: {len(complete_data):,}")
    print(f"   Features: {complete_data.shape[1]}")
    
    # Get attack categories
    attack_categories = complete_data['attack_cat'].unique()
    print(f"   Attack categories: {len(attack_categories)}")
    print(f"   Categories: {list(attack_categories)}")
    
    # Analyze original class distribution
    print(f"\n2. Original Class Distribution Analysis:")
    original_counts = Counter(complete_data['attack_cat'])
    total_samples = len(complete_data)
    
    # Sort by count for better visualization
    sorted_categories = sorted(original_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Number of classes: {len(original_counts)}")
    
    # Print detailed distribution
    for category, count in sorted_categories:
        percentage = (count / total_samples) * 100
        print(f"     {category}: {count:,} samples ({percentage:.2f}%)")
    
    # Calculate original imbalance ratio
    max_count = max(original_counts.values())
    min_count = min(original_counts.values())
    original_imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n   Original imbalance ratio: {original_imbalance_ratio:.2f}:1 (max/min)")
    
    # Apply rebalancing to the complete dataset
    print(f"\n3. Applying Rebalancing to Complete Dataset...")
    
    # Prepare data for rebalancing
    feature_cols = [col for col in complete_data.columns if col not in ['attack_cat', 'label']]
    X = complete_data[feature_cols].values
    
    # Create label mapping
    label_mapping = {cat: idx for idx, cat in enumerate(sorted(attack_categories))}
    y = np.array([label_mapping[cat] for cat in complete_data['attack_cat']])
    
    print(f"   Features used for rebalancing: {len(feature_cols)}")
    print(f"   Label mapping: {label_mapping}")
    
    # Analyze class distribution before rebalancing
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n   Class distribution before rebalancing:")
    for class_label, count in zip(unique_classes, class_counts):
        attack_name = list(label_mapping.keys())[list(label_mapping.values()).index(class_label)]
        percentage = (count / len(y)) * 100
        print(f"     {attack_name} (Label {class_label}): {count:,} samples ({percentage:.2f}%)")
    
    # Calculate imbalance ratio before rebalancing
    max_count_before = np.max(class_counts)
    min_count_before = np.min(class_counts)
    imbalance_ratio_before = max_count_before / min_count_before if min_count_before > 0 else float('inf')
    print(f"   Imbalance ratio before rebalancing: {imbalance_ratio_before:.2f}:1")
    
    # Strategy: Balance classes to a reasonable size
    # Target: Make all classes have at least 10000 samples and at most 20000 samples
    # This will significantly oversample minority classes like Worms, Analysis, Backdoor, Shellcode
    target_min_samples = 10000
    target_max_samples = 20000
    
    # Create sampling strategy
    sampling_strategy = {}
    for class_label, count in zip(unique_classes, class_counts):
        if count < target_min_samples:
            # Oversample minority classes to target_min_samples
            sampling_strategy[class_label] = target_min_samples
        elif count > target_max_samples:
            # Undersample majority classes to target_max_samples
            sampling_strategy[class_label] = target_max_samples
        else:
            # Keep as is
            sampling_strategy[class_label] = count
    
    print(f"\n   Sampling strategy:")
    for class_label, target_count in sampling_strategy.items():
        attack_name = list(label_mapping.keys())[list(label_mapping.values()).index(class_label)]
        current_count = class_counts[unique_classes == class_label][0]
        action = "oversample" if target_count > current_count else "undersample" if target_count < current_count else "keep"
        print(f"     {attack_name}: {current_count:,} → {target_count:,} ({action})")
    
    # Step 1: Apply SMOTE for oversampling minority classes only
    print(f"\n   Step 1: Applying SMOTE oversampling...")
    
    # Create oversampling strategy (only for classes that need oversampling)
    oversample_strategy = {}
    for class_label, count in zip(unique_classes, class_counts):
        target_count = sampling_strategy[class_label]
        if target_count > count:  # Only oversample if target > current
            oversample_strategy[class_label] = target_count
    
    if oversample_strategy:
        smote = SMOTE(
            sampling_strategy=oversample_strategy,
            random_state=42,
            k_neighbors=3
        )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"   SMOTE completed: {len(X)} → {len(X_resampled)} samples")
        except Exception as e:
            print(f"   SMOTE failed: {e}")
            print("   Falling back to original data")
            X_resampled, y_resampled = X, y
    else:
        print("   No classes need oversampling, skipping SMOTE")
        X_resampled, y_resampled = X, y
    
    # Step 2: Apply RandomUnderSampler for undersampling majority classes
    print(f"   Step 2: Applying RandomUnderSampler...")
    
    # Create undersampling strategy (only for classes that need undersampling)
    undersample_strategy = {}
    for class_label, count in zip(unique_classes, class_counts):
        target_count = sampling_strategy[class_label]
        if target_count < count:  # Only undersample if target < current
            undersample_strategy[class_label] = target_count
    
    if undersample_strategy:
        undersampler = RandomUnderSampler(
            sampling_strategy=undersample_strategy,
            random_state=42
        )
        
        try:
            X_balanced, y_balanced = undersampler.fit_resample(X_resampled, y_resampled)
            print(f"   RandomUnderSampler completed: {len(X_resampled)} → {len(X_balanced)} samples")
        except Exception as e:
            print(f"   RandomUnderSampler failed: {e}")
            print("   Using SMOTE result")
            X_balanced, y_balanced = X_resampled, y_resampled
    else:
        print("   No classes need undersampling, skipping RandomUnderSampler")
        X_balanced, y_balanced = X_resampled, y_resampled
    
    # Analyze class distribution after rebalancing
    unique_classes_after, class_counts_after = np.unique(y_balanced, return_counts=True)
    print(f"\n   Class distribution after rebalancing:")
    for class_label, count in zip(unique_classes_after, class_counts_after):
        attack_name = list(label_mapping.keys())[list(label_mapping.values()).index(class_label)]
        percentage = (count / len(y_balanced)) * 100
        print(f"     {attack_name} (Label {class_label}): {count:,} samples ({percentage:.2f}%)")
    
    # Calculate new imbalance ratio
    max_count_after = np.max(class_counts_after)
    min_count_after = np.min(class_counts_after)
    imbalance_ratio_after = max_count_after / min_count_after if min_count_after > 0 else float('inf')
    print(f"   Imbalance ratio after rebalancing: {imbalance_ratio_after:.2f}:1")
    
    # Calculate improvement
    improvement = (imbalance_ratio_before - imbalance_ratio_after) / imbalance_ratio_before * 100
    print(f"   Imbalance reduction: {improvement:.1f}%")
    
    # Create comprehensive visualizations
    print(f"\n4. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('UNSW-NB15 Complete Dataset: Original vs Rebalanced Class Distribution', fontsize=20, fontweight='bold')
    
    # Prepare data for visualization
    original_categories = [item[0] for item in sorted_categories]
    original_counts_list = [item[1] for item in sorted_categories]
    
    # Rebalanced data
    rebalanced_categories = [list(label_mapping.keys())[list(label_mapping.values()).index(label)] for label in unique_classes_after]
    rebalanced_counts_list = [class_counts_after[unique_classes_after == label][0] for label in unique_classes_after]
    
    # Sort rebalanced data by category name to match original
    rebalanced_sorted = sorted(zip(rebalanced_categories, rebalanced_counts_list), key=lambda x: x[0])
    rebalanced_categories_sorted = [item[0] for item in rebalanced_sorted]
    rebalanced_counts_sorted = [item[1] for item in rebalanced_sorted]
    
    # 1. Original dataset bar plot (top-left)
    ax1 = plt.subplot(3, 3, 1)
    bars1 = ax1.bar(range(len(original_categories)), original_counts_list, alpha=0.8, 
                    color=plt.cm.Set3(np.linspace(0, 1, len(original_categories))))
    ax1.set_xlabel('Attack Categories', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Original Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(original_categories)))
    ax1.set_xticklabels(original_categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars1, original_counts_list):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Rebalanced dataset bar plot (top-center)
    ax2 = plt.subplot(3, 3, 2)
    bars2 = ax2.bar(range(len(rebalanced_categories_sorted)), rebalanced_counts_sorted, alpha=0.8, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(rebalanced_categories_sorted))))
    ax2.set_xlabel('Attack Categories', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Rebalanced Dataset', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(rebalanced_categories_sorted)))
    ax2.set_xticklabels(rebalanced_categories_sorted, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, rebalanced_counts_sorted):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Side-by-side comparison (top-right)
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(len(original_categories))
    width = 0.35
    
    bars3_orig = ax3.bar(x - width/2, original_counts_list, width, label='Original', alpha=0.8, color='skyblue')
    bars3_rebal = ax3.bar(x + width/2, rebalanced_counts_sorted, width, label='Rebalanced', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Attack Categories', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Original vs Rebalanced Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(original_categories, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Original pie chart (middle-left)
    ax4 = plt.subplot(3, 3, 4)
    wedges4, texts4, autotexts4 = ax4.pie(original_counts_list, labels=original_categories, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Original Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # 5. Rebalanced pie chart (middle-center)
    ax5 = plt.subplot(3, 3, 5)
    wedges5, texts5, autotexts5 = ax5.pie(rebalanced_counts_sorted, labels=rebalanced_categories_sorted, autopct='%1.1f%%', startangle=90)
    ax5.set_title('Rebalanced Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # 6. Log scale comparison (middle-right)
    ax6 = plt.subplot(3, 3, 6)
    bars6_orig = ax6.bar(x - width/2, original_counts_list, width, label='Original', alpha=0.8, color='skyblue')
    bars6_rebal = ax6.bar(x + width/2, rebalanced_counts_sorted, width, label='Rebalanced', alpha=0.8, color='lightcoral')
    
    ax6.set_xlabel('Attack Categories', fontsize=12)
    ax6.set_ylabel('Number of Samples (Log Scale)', fontsize=12)
    ax6.set_title('Log Scale Comparison', fontsize=14, fontweight='bold')
    ax6.set_yscale('log')
    ax6.set_xticks(x)
    ax6.set_xticklabels(original_categories, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Imbalance ratio comparison (bottom-left)
    ax7 = plt.subplot(3, 3, 7)
    imbalance_data = {
        'Original': imbalance_ratio_before,
        'Rebalanced': imbalance_ratio_after
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
        'Original': len(y),
        'Rebalanced': len(y_balanced)
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
    
    # 9. Improvement summary (bottom-right)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    improvement_text = f"""
    REBALANCING SUMMARY
    
    Original Dataset:
    • Total samples: {len(y):,}
    • Imbalance ratio: {imbalance_ratio_before:.1f}:1
    • Max class: {max_count_before:,}
    • Min class: {min_count_before:,}
    
    Rebalanced Dataset:
    • Total samples: {len(y_balanced):,}
    • Imbalance ratio: {imbalance_ratio_after:.1f}:1
    • Max class: {max_count_after:,}
    • Min class: {min_count_after:,}
    
    Improvement:
    • Imbalance reduction: {improvement:.1f}%
    • Sample change: {len(y_balanced) - len(y):+,} samples
    """
    
    ax9.text(0.1, 0.9, improvement_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'complete_dataset_rebalancing_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*80)
    print("COMPLETE DATASET REBALANCING ANALYSIS COMPLETE")
    print("="*80)
    
    return original_counts, rebalanced_counts_sorted, improvement

if __name__ == "__main__":
    original, rebalanced, improvement = visualize_complete_dataset_rebalancing()
