#!/usr/bin/env python3
"""
Debug script to test plot generation with real data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from visualization.performance_visualization import PerformanceVisualizer

def test_confusion_matrix_plotting():
    """Test confusion matrix plotting with real data"""
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    print("Base model confusion matrix:", results['base_model']['confusion_matrix'])
    print("TTT model confusion matrix:", results['ttt_model']['confusion_matrix'])
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # Test base model confusion matrix
    print("\nTesting base model confusion matrix plotting...")
    base_path = visualizer.plot_confusion_matrices(
        results['base_model'], 
        save=True, 
        title_suffix=" - Base Model (Debug)"
    )
    print(f"Base model plot saved to: {base_path}")
    
    # Test TTT model confusion matrix
    print("\nTesting TTT model confusion matrix plotting...")
    ttt_path = visualizer.plot_confusion_matrices(
        results['ttt_model'], 
        save=True, 
        title_suffix=" - TTT Model (Debug)"
    )
    print(f"TTT model plot saved to: {ttt_path}")
    
    # Test ROC curves
    print("\nTesting ROC curves plotting...")
    roc_path = visualizer.plot_roc_curves(
        results['base_model'],
        results['ttt_model'],
        save=True
    )
    print(f"ROC curves plot saved to: {roc_path}")

if __name__ == "__main__":
    test_confusion_matrix_plotting()
