#!/usr/bin/env python3
"""
Detailed debug script to analyze plot content
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from visualization.performance_visualization import PerformanceVisualizer

def analyze_confusion_matrix_data():
    """Analyze the confusion matrix data in detail"""
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    print("=== CONFUSION MATRIX ANALYSIS ===")
    
    # Base model analysis
    base_cm = results['base_model']['confusion_matrix']
    print(f"\nBase Model Confusion Matrix: {base_cm}")
    print(f"Format: {type(base_cm)}")
    print(f"Shape: {np.array(base_cm).shape}")
    
    # Convert to numpy array for analysis
    base_cm_array = np.array(base_cm)
    tn, fp = base_cm_array[0]
    fn, tp = base_cm_array[1]
    
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"Total samples: {tn + fp + fn + tp}")
    print(f"Accuracy: {(tp + tn) / (tn + fp + fn + tp):.4f}")
    print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
    
    # TTT model analysis
    ttt_cm = results['ttt_model']['confusion_matrix']
    print(f"\nTTT Model Confusion Matrix: {ttt_cm}")
    print(f"Format: {type(ttt_cm)}")
    print(f"Shape: {np.array(ttt_cm).shape}")
    
    ttt_cm_array = np.array(ttt_cm)
    tn, fp = ttt_cm_array[0]
    fn, tp = ttt_cm_array[1]
    
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"Total samples: {tn + fp + fn + tp}")
    print(f"Accuracy: {(tp + tn) / (tn + fp + fn + tp):.4f}")
    print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")

def analyze_roc_data():
    """Analyze the ROC curve data"""
    
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    print("\n=== ROC CURVE ANALYSIS ===")
    
    # Base model ROC
    base_roc = results['base_model']['roc_curve']
    print(f"\nBase Model ROC Curve:")
    print(f"FPR points: {len(base_roc['fpr'])}")
    print(f"TPR points: {len(base_roc['tpr'])}")
    print(f"Thresholds: {len(base_roc['thresholds'])}")
    print(f"ROC AUC: {results['base_model']['roc_auc']:.4f}")
    print(f"Optimal Threshold: {results['base_model']['optimal_threshold']:.4f}")
    
    # TTT model ROC
    ttt_roc = results['ttt_model']['roc_curve']
    print(f"\nTTT Model ROC Curve:")
    print(f"FPR points: {len(ttt_roc['fpr'])}")
    print(f"TPR points: {len(ttt_roc['tpr'])}")
    print(f"Thresholds: {len(ttt_roc['thresholds'])}")
    print(f"ROC AUC: {results['ttt_model']['roc_auc']:.4f}")
    print(f"Optimal Threshold: {results['ttt_model']['optimal_threshold']:.4f}")

def create_simple_test_plot():
    """Create a simple test plot to verify matplotlib is working"""
    
    print("\n=== CREATING SIMPLE TEST PLOT ===")
    
    # Create a simple test plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Test data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    ax.plot(x, y, 'b-', linewidth=2, label='Test Line')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title('Simple Test Plot')
    ax.legend()
    ax.grid(True)
    
    # Save the plot
    test_path = 'performance_plots/test_plot.png'
    plt.savefig(test_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test plot saved to: {test_path}")

if __name__ == "__main__":
    analyze_confusion_matrix_data()
    analyze_roc_data()
    create_simple_test_plot()
