#!/usr/bin/env python3
"""
Script to examine the actual content of performance comparison plot
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from visualization.performance_visualization import PerformanceVisualizer

def examine_performance_comparison_data():
    """Examine what data is actually being passed to the performance comparison plot"""
    
    print("=== PERFORMANCE COMPARISON DATA EXAMINATION ===")
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    print("Base Model Results:")
    print(f"  Accuracy: {results['base_model']['accuracy_mean']:.4f} ± {results['base_model']['accuracy_std']:.4f}")
    print(f"  F1 Score: {results['base_model']['macro_f1_mean']:.4f} ± {results['base_model']['macro_f1_std']:.4f}")
    print(f"  MCC: {results['base_model']['mcc_mean']:.4f} ± {results['base_model']['mcc_std']:.4f}")
    print(f"  ROC AUC: {results['base_model']['roc_auc']:.4f}")
    print(f"  Confusion Matrix: {results['base_model']['confusion_matrix']}")
    
    print("\nTTT Model Results:")
    print(f"  Accuracy: {results['ttt_model']['accuracy_mean']:.4f} ± {results['ttt_model']['accuracy_std']:.4f}")
    print(f"  F1 Score: {results['ttt_model']['macro_f1_mean']:.4f} ± {results['ttt_model']['macro_f1_std']:.4f}")
    print(f"  MCC: {results['ttt_model']['mcc_mean']:.4f} ± {results['ttt_model']['mcc_std']:.4f}")
    print(f"  ROC AUC: {results['ttt_model']['roc_auc']:.4f}")
    print(f"  Confusion Matrix: {results['ttt_model']['confusion_matrix']}")
    
    # Check if the values are actually meaningful
    print("\n=== DATA VALIDATION ===")
    
    base_accuracy = results['base_model']['accuracy_mean']
    ttt_accuracy = results['ttt_model']['accuracy_mean']
    
    print(f"Base Model Accuracy: {base_accuracy}")
    print(f"TTT Model Accuracy: {ttt_accuracy}")
    
    if base_accuracy == 0.5:
        print("⚠️  WARNING: Base model accuracy is exactly 0.5 - this suggests random guessing!")
    
    if ttt_accuracy == 1.0:
        print("⚠️  WARNING: TTT model accuracy is exactly 1.0 - this suggests perfect classification!")
    
    # Check confusion matrices
    base_cm = results['base_model']['confusion_matrix']
    ttt_cm = results['ttt_model']['confusion_matrix']
    
    print(f"\nBase Model Confusion Matrix Analysis:")
    print(f"  TN: {base_cm[0][0]}, FP: {base_cm[0][1]}")
    print(f"  FN: {base_cm[1][0]}, TP: {base_cm[1][1]}")
    
    print(f"\nTTT Model Confusion Matrix Analysis:")
    print(f"  TN: {ttt_cm[0][0]}, FP: {ttt_cm[0][1]}")
    print(f"  FN: {ttt_cm[1][0]}, TP: {ttt_cm[1][1]}")
    
    # Check if base model is predicting everything as positive
    if base_cm[0][0] == 0 and base_cm[1][0] == 0:
        print("⚠️  WARNING: Base model is predicting everything as positive (attack)!")
        print("   This means it's not learning to distinguish between normal and attack traffic.")
    
    # Check if TTT model is perfect
    if ttt_cm[0][1] == 0 and ttt_cm[1][0] == 0:
        print("✅ TTT model shows perfect classification (no false positives or false negatives)")

def create_debug_performance_plot():
    """Create a debug version of the performance comparison plot with detailed logging"""
    
    print("\n=== CREATING DEBUG PERFORMANCE COMPARISON PLOT ===")
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # Create the plot with detailed logging
    try:
        plot_path = visualizer.plot_performance_comparison_with_annotations(
            results['base_model'], 
            results['ttt_model'],
            scenario_names=["Edge-IIoTset Debug"],
            save=True
        )
        print(f"✅ Debug performance comparison plot created: {plot_path}")
        
        # Also create a simple bar chart to verify the data
        create_simple_bar_chart(results)
        
    except Exception as e:
        print(f"❌ Debug performance comparison plot failed: {e}")
        import traceback
        traceback.print_exc()

def create_simple_bar_chart(results):
    """Create a simple bar chart to verify the data visually"""
    
    print("\n=== CREATING SIMPLE BAR CHART FOR VERIFICATION ===")
    
    # Extract data
    base_accuracy = results['base_model']['accuracy_mean']
    ttt_accuracy = results['ttt_model']['accuracy_mean']
    
    base_f1 = results['base_model']['macro_f1_mean']
    ttt_f1 = results['ttt_model']['macro_f1_mean']
    
    base_auc = results['base_model']['roc_auc']
    ttt_auc = results['ttt_model']['roc_auc']
    
    # Create simple bar chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    models = ['Base Model', 'TTT Model']
    accuracies = [base_accuracy, ttt_accuracy]
    bars1 = ax1.bar(models, accuracies, color=['red', 'green'], alpha=0.7)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1 Score comparison
    f1_scores = [base_f1, ttt_f1]
    bars2 = ax2.bar(models, f1_scores, color=['red', 'green'], alpha=0.7)
    ax2.set_title('F1 Score Comparison')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # ROC AUC comparison
    auc_scores = [base_auc, ttt_auc]
    bars3 = ax3.bar(models, auc_scores, color=['red', 'green'], alpha=0.7)
    ax3.set_title('ROC AUC Comparison')
    ax3.set_ylabel('ROC AUC')
    ax3.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, auc in zip(bars3, auc_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_plots/debug_simple_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Simple bar chart created: performance_plots/debug_simple_comparison.png")
    
    # Print the actual values
    print(f"\nActual Values:")
    print(f"Base Model - Accuracy: {base_accuracy:.4f}, F1: {base_f1:.4f}, AUC: {base_auc:.4f}")
    print(f"TTT Model  - Accuracy: {ttt_accuracy:.4f}, F1: {ttt_f1:.4f}, AUC: {ttt_auc:.4f}")

if __name__ == "__main__":
    examine_performance_comparison_data()
    create_debug_performance_plot()
