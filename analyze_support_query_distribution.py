#!/usr/bin/env python3
"""
Support and Query Set Distribution Analysis
Analyzes the distribution of samples in meta-learning tasks for both training and testing phases
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from models.transductive_fewshot_model import create_meta_tasks

def analyze_support_query_distribution():
    """Analyze support and query set distribution for meta-learning tasks"""
    
    print("🔍 SUPPORT AND QUERY SET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Load and preprocess data
    print("Loading and preprocessing UNSW-NB15 dataset...")
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    # Get data
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")
    
    # Analyze class distribution
    print("\n📊 CLASS DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    train_unique, train_counts = torch.unique(y_train, return_counts=True)
    test_unique, test_counts = torch.unique(y_test, return_counts=True)
    
    print("Training data classes:")
    for label, count in zip(train_unique, train_counts):
        print(f"  Class {label.item()}: {count.item():,} samples")
    
    print("\nTest data classes:")
    for label, count in zip(test_unique, test_counts):
        print(f"  Class {label.item()}: {count.item():,} samples")
    
    # Create meta-tasks for training phase
    print("\n🎯 TRAINING PHASE META-TASKS ANALYSIS")
    print("-" * 50)
    
    training_meta_tasks = create_meta_tasks(
        X_train, y_train,
        n_way=10, k_shot=50, n_query=100, n_tasks=20,
        phase="training",
        normal_query_ratio=0.8,
        zero_day_attack_label=4  # Exclude DoS (label 4) from training
    )
    
    print(f"Created {len(training_meta_tasks)} training meta-tasks")
    
    # Analyze training meta-tasks
    training_stats = analyze_meta_tasks(training_meta_tasks, "Training")
    
    # Create meta-tasks for testing phase
    print("\n🧪 TESTING PHASE META-TASKS ANALYSIS")
    print("-" * 50)
    
    # Convert to binary for testing (Normal vs Attack)
    y_test_binary = (y_test != 0).long()
    
    testing_meta_tasks = create_meta_tasks(
        X_test, y_test_binary,
        n_way=2, k_shot=50, n_query=100, n_tasks=20,
        phase="testing",
        normal_query_ratio=0.9,
        zero_day_attack_label=None  # No exclusion for testing
    )
    
    print(f"Created {len(testing_meta_tasks)} testing meta-tasks")
    
    # Analyze testing meta-tasks
    testing_stats = analyze_meta_tasks(testing_meta_tasks, "Testing")
    
    # Generate visualizations
    print("\n📈 GENERATING VISUALIZATIONS")
    print("-" * 30)
    
    create_distribution_plots(training_stats, testing_stats)
    
    # Summary report
    print("\n📋 SUMMARY REPORT")
    print("=" * 30)
    
    print(f"Training Phase:")
    print(f"  - Tasks created: {len(training_meta_tasks)}")
    print(f"  - Average support samples per task: {training_stats['avg_support_samples']:.1f}")
    print(f"  - Average query samples per task: {training_stats['avg_query_samples']:.1f}")
    print(f"  - Support set class diversity: {training_stats['avg_support_classes']:.1f} classes")
    print(f"  - Query set normal ratio: {training_stats['avg_query_normal_ratio']:.1%}")
    
    print(f"\nTesting Phase:")
    print(f"  - Tasks created: {len(testing_meta_tasks)}")
    print(f"  - Average support samples per task: {testing_stats['avg_support_samples']:.1f}")
    print(f"  - Average query samples per task: {testing_stats['avg_query_samples']:.1f}")
    print(f"  - Support set class diversity: {testing_stats['avg_support_classes']:.1f} classes")
    print(f"  - Query set normal ratio: {testing_stats['avg_query_normal_ratio']:.1%}")
    
    print("\n✅ Analysis completed successfully!")

def analyze_meta_tasks(meta_tasks, phase_name):
    """Analyze a set of meta-tasks and return statistics"""
    
    support_samples = []
    query_samples = []
    support_classes = []
    query_normal_ratios = []
    support_class_distributions = []
    query_class_distributions = []
    
    for i, task in enumerate(meta_tasks):
        # Support set analysis
        support_x = task['support_x']
        support_y = task['support_y']
        support_samples.append(len(support_x))
        
        # Count unique classes in support set
        unique_support_classes = torch.unique(support_y)
        support_classes.append(len(unique_support_classes))
        
        # Support set class distribution
        support_class_counts = torch.bincount(support_y)
        support_class_distributions.append(support_class_counts.tolist())
        
        # Query set analysis
        query_x = task['query_x']
        query_y = task['query_y']
        query_samples.append(len(query_x))
        
        # Query set normal ratio (assuming 0 is normal)
        if len(query_y) > 0:
            normal_count = (query_y == 0).sum().item()
            query_normal_ratios.append(normal_count / len(query_y))
        else:
            query_normal_ratios.append(0.0)
        
        # Query set class distribution
        query_class_counts = torch.bincount(query_y)
        query_class_distributions.append(query_class_counts.tolist())
        
        # Log first few tasks for debugging
        if i < 3:
            print(f"  Task {i+1}:")
            print(f"    Support: {len(support_x)} samples, {len(unique_support_classes)} classes")
            print(f"    Query: {len(query_x)} samples, Normal ratio: {query_normal_ratios[-1]:.1%}")
            print(f"    Support classes: {unique_support_classes.tolist()}")
            print(f"    Query classes: {torch.unique(query_y).tolist()}")
    
    # Calculate statistics
    stats = {
        'avg_support_samples': np.mean(support_samples),
        'std_support_samples': np.std(support_samples),
        'avg_query_samples': np.mean(query_samples),
        'std_query_samples': np.std(query_samples),
        'avg_support_classes': np.mean(support_classes),
        'std_support_classes': np.std(support_classes),
        'avg_query_normal_ratio': np.mean(query_normal_ratios),
        'std_query_normal_ratio': np.std(query_normal_ratios),
        'support_samples': support_samples,
        'query_samples': query_samples,
        'support_classes': support_classes,
        'query_normal_ratios': query_normal_ratios,
        'support_class_distributions': support_class_distributions,
        'query_class_distributions': query_class_distributions
    }
    
    return stats

def create_distribution_plots(training_stats, testing_stats):
    """Create visualization plots for the distribution analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Support and Query Set Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Support set sample sizes
    axes[0, 0].hist(training_stats['support_samples'], bins=10, alpha=0.7, label='Training', color='blue')
    axes[0, 0].hist(testing_stats['support_samples'], bins=10, alpha=0.7, label='Testing', color='red')
    axes[0, 0].set_title('Support Set Sample Sizes')
    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Query set sample sizes
    axes[0, 1].hist(training_stats['query_samples'], bins=10, alpha=0.7, label='Training', color='blue')
    axes[0, 1].hist(testing_stats['query_samples'], bins=10, alpha=0.7, label='Testing', color='red')
    axes[0, 1].set_title('Query Set Sample Sizes')
    axes[0, 1].set_xlabel('Number of Samples')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Support set class diversity
    axes[0, 2].hist(training_stats['support_classes'], bins=10, alpha=0.7, label='Training', color='blue')
    axes[0, 2].hist(testing_stats['support_classes'], bins=10, alpha=0.7, label='Testing', color='red')
    axes[0, 2].set_title('Support Set Class Diversity')
    axes[0, 2].set_xlabel('Number of Classes')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Query set normal ratios
    axes[1, 0].hist(training_stats['query_normal_ratios'], bins=10, alpha=0.7, label='Training', color='blue')
    axes[1, 0].hist(testing_stats['query_normal_ratios'], bins=10, alpha=0.7, label='Testing', color='red')
    axes[1, 0].set_title('Query Set Normal Sample Ratios')
    axes[1, 0].set_xlabel('Normal Sample Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Support vs Query sample size correlation
    axes[1, 1].scatter(training_stats['support_samples'], training_stats['query_samples'], 
                      alpha=0.7, label='Training', color='blue', s=50)
    axes[1, 1].scatter(testing_stats['support_samples'], testing_stats['query_samples'], 
                      alpha=0.7, label='Testing', color='red', s=50)
    axes[1, 1].set_title('Support vs Query Sample Sizes')
    axes[1, 1].set_xlabel('Support Samples')
    axes[1, 1].set_ylabel('Query Samples')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    # Create summary text
    summary_text = f"""
    TRAINING PHASE STATISTICS:
    • Support Samples: {training_stats['avg_support_samples']:.1f} ± {training_stats['std_support_samples']:.1f}
    • Query Samples: {training_stats['avg_query_samples']:.1f} ± {training_stats['std_query_samples']:.1f}
    • Support Classes: {training_stats['avg_support_classes']:.1f} ± {training_stats['std_support_classes']:.1f}
    • Normal Ratio: {training_stats['avg_query_normal_ratio']:.1%} ± {training_stats['std_query_normal_ratio']:.1%}
    
    TESTING PHASE STATISTICS:
    • Support Samples: {testing_stats['avg_support_samples']:.1f} ± {testing_stats['std_support_samples']:.1f}
    • Query Samples: {testing_stats['avg_query_samples']:.1f} ± {testing_stats['std_query_samples']:.1f}
    • Support Classes: {testing_stats['avg_support_classes']:.1f} ± {testing_stats['std_support_classes']:.1f}
    • Normal Ratio: {testing_stats['avg_query_normal_ratio']:.1%} ± {testing_stats['std_query_normal_ratio']:.1%}
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'support_query_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Distribution analysis plot saved: {output_path}")
    
    plt.show()
    
    # Create detailed class distribution analysis
    create_class_distribution_analysis(training_stats, testing_stats)

def create_class_distribution_analysis(training_stats, testing_stats):
    """Create detailed class distribution analysis"""
    
    # Analyze support set class distributions
    print("\n📊 DETAILED CLASS DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    # Training support set class analysis
    print("Training Support Set Class Distribution:")
    training_support_classes = []
    for dist in training_stats['support_class_distributions']:
        for class_id, count in enumerate(dist):
            if count > 0:
                training_support_classes.extend([class_id] * count)
    
    training_class_counts = Counter(training_support_classes)
    for class_id in sorted(training_class_counts.keys()):
        print(f"  Class {class_id}: {training_class_counts[class_id]} samples")
    
    # Testing support set class analysis
    print("\nTesting Support Set Class Distribution:")
    testing_support_classes = []
    for dist in testing_stats['support_class_distributions']:
        for class_id, count in enumerate(dist):
            if count > 0:
                testing_support_classes.extend([class_id] * count)
    
    testing_class_counts = Counter(testing_support_classes)
    for class_id in sorted(testing_class_counts.keys()):
        print(f"  Class {class_id}: {testing_class_counts[class_id]} samples")
    
    # Create class distribution comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training support set class distribution
    classes = sorted(training_class_counts.keys())
    counts = [training_class_counts[c] for c in classes]
    ax1.bar(classes, counts, color='blue', alpha=0.7)
    ax1.set_title('Training Support Set Class Distribution')
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Total Samples')
    ax1.grid(True, alpha=0.3)
    
    # Testing support set class distribution
    classes = sorted(testing_class_counts.keys())
    counts = [testing_class_counts[c] for c in classes]
    ax2.bar(classes, counts, color='red', alpha=0.7)
    ax2.set_title('Testing Support Set Class Distribution')
    ax2.set_xlabel('Class ID')
    ax2.set_ylabel('Total Samples')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'class_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Class distribution analysis plot saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    analyze_support_query_distribution()



