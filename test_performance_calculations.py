#!/usr/bin/env python3
"""
Test script to verify performance comparison method calculations
"""

import json
from visualization.performance_visualization import PerformanceVisualizer

def test_performance_comparison_calculations():
    """Test the performance comparison method calculations directly"""
    
    print("=== TESTING PERFORMANCE COMPARISON METHOD CALCULATIONS ===")
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # Test the calculation logic directly
    base_results = results['base_model']
    ttt_results = results['ttt_model']
    
    base_metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'macro_f1_mean', 'mcc_mean']
    
    # Get confusion matrices for precision/recall calculation
    base_cm = base_results.get('confusion_matrix', [[0, 0], [0, 0]])
    ttt_cm = ttt_results.get('confusion_matrix', [[0, 0], [0, 0]])
    
    print("Base model confusion matrix:", base_cm)
    print("TTT model confusion matrix:", ttt_cm)
    
    # Calculate metrics using the same logic as the performance comparison method
    base_values = []
    ttt_values = []
    
    for metric in base_metrics:
        if metric == 'accuracy_mean':
            base_val = base_results.get('accuracy_mean', 0)
            ttt_val = ttt_results.get('accuracy_mean', 0)
        elif metric == 'precision_mean':
            # Calculate precision from confusion matrix
            if len(base_cm) == 2 and len(base_cm[0]) == 2:
                tp, fp = base_cm[1][1], base_cm[0][1]
                base_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            else:
                base_val = 0
            if len(ttt_cm) == 2 and len(ttt_cm[0]) == 2:
                tp, fp = ttt_cm[1][1], ttt_cm[0][1]
                ttt_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            else:
                ttt_val = 0
        elif metric == 'recall_mean':
            # Calculate recall from confusion matrix
            if len(base_cm) == 2 and len(base_cm[0]) == 2:
                tp, fn = base_cm[1][1], base_cm[1][0]
                base_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                base_val = 0
            if len(ttt_cm) == 2 and len(ttt_cm[0]) == 2:
                tp, fn = ttt_cm[1][1], ttt_cm[1][0]
                ttt_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                ttt_val = 0
        elif metric == 'macro_f1_mean':
            base_val = base_results.get('macro_f1_mean', 0)
            ttt_val = ttt_results.get('macro_f1_mean', 0)
        elif metric == 'mcc_mean':
            base_val = base_results.get('mcc_mean', 0)
            ttt_val = ttt_results.get('mcc_mean', 0)
        else:
            base_val = base_results.get(metric, 0)
            ttt_val = ttt_results.get(metric, 0)
        
        base_values.append(base_val)
        ttt_values.append(ttt_val)
        
        print(f"{metric}:")
        print(f"  Base: {base_val:.4f}")
        print(f"  TTT:  {ttt_val:.4f}")
    
    print(f"\nBase values: {base_values}")
    print(f"TTT values: {ttt_values}")
    
    # Test the actual plot method
    print("\n=== TESTING ACTUAL PLOT METHOD ===")
    try:
        plot_path = visualizer.plot_performance_comparison_with_annotations(
            base_results, 
            ttt_results,
            scenario_names=["Test"],
            save=True
        )
        print(f"✅ Performance comparison plot created: {plot_path}")
    except Exception as e:
        print(f"❌ Performance comparison plot failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_performance_comparison_calculations()
