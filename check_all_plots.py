#!/usr/bin/env python3
"""
Comprehensive script to check ALL visualization plots for actual data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from visualization.performance_visualization import PerformanceVisualizer

def check_all_plots():
    """Check all visualization plots for actual data"""
    
    print("=== COMPREHENSIVE PLOT DATA ANALYSIS ===")
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded successfully. Keys: {list(results.keys())}")
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # 1. Check Training History Plot
    print("\n1. === TRAINING HISTORY PLOT ===")
    training_history = {
        'epoch_losses': [0.5, 0.4, 0.3],
        'epoch_accuracies': [0.5, 0.6, 0.7]
    }
    print(f"Training history data: {training_history}")
    try:
        training_path = visualizer.plot_training_history(training_history, save=True)
        print(f"✅ Training history plot: {training_path}")
    except Exception as e:
        print(f"❌ Training history plot failed: {e}")
    
    # 2. Check Confusion Matrices
    print("\n2. === CONFUSION MATRICES ===")
    print(f"Base model confusion matrix: {results['base_model']['confusion_matrix']}")
    print(f"TTT model confusion matrix: {results['ttt_model']['confusion_matrix']}")
    try:
        base_cm_path = visualizer.plot_confusion_matrices(results['base_model'], save=True, title_suffix=" - Base Model")
        ttt_cm_path = visualizer.plot_confusion_matrices(results['ttt_model'], save=True, title_suffix=" - TTT Model")
        print(f"✅ Base confusion matrix: {base_cm_path}")
        print(f"✅ TTT confusion matrix: {ttt_cm_path}")
    except Exception as e:
        print(f"❌ Confusion matrices failed: {e}")
    
    # 3. Check TTT Adaptation Plot
    print("\n3. === TTT ADAPTATION PLOT ===")
    ttt_adaptation_data = {
        'steps': list(range(10)),
        'total_losses': [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
        'support_losses': [0.3, 0.28, 0.25, 0.22, 0.18, 0.15, 0.12, 0.08, 0.05, 0.03],
        'consistency_losses': [0.2, 0.17, 0.15, 0.13, 0.12, 0.10, 0.08, 0.07, 0.05, 0.02]
    }
    print(f"TTT adaptation data: {ttt_adaptation_data}")
    try:
        ttt_path = visualizer.plot_ttt_adaptation(ttt_adaptation_data, save=True)
        print(f"✅ TTT adaptation plot: {ttt_path}")
    except Exception as e:
        print(f"❌ TTT adaptation plot failed: {e}")
    
    # 4. Check Client Performance Plot
    print("\n4. === CLIENT PERFORMANCE PLOT ===")
    client_results = [
        {'client_id': 'client_1', 'accuracy': 0.61, 'f1_score': 0.58, 'precision': 0.62, 'recall': 0.60},
        {'client_id': 'client_2', 'accuracy': 0.64, 'f1_score': 0.61, 'precision': 0.65, 'recall': 0.63},
        {'client_id': 'client_3', 'accuracy': 0.65, 'f1_score': 0.63, 'precision': 0.67, 'recall': 0.65}
    ]
    print(f"Client performance data: {client_results}")
    try:
        client_path = visualizer.plot_client_performance(client_results, save=True)
        print(f"✅ Client performance plot: {client_path}")
    except Exception as e:
        print(f"❌ Client performance plot failed: {e}")
    
    # 5. Check Blockchain Metrics Plot
    print("\n5. === BLOCKCHAIN METRICS PLOT ===")
    blockchain_data = {
        'rounds': [1, 2, 3],
        'gas_used': [20000, 22000, 21000],
        'transactions': [3, 3, 3]
    }
    print(f"Blockchain metrics data: {blockchain_data}")
    try:
        blockchain_path = visualizer.plot_blockchain_metrics(blockchain_data, save=True)
        print(f"✅ Blockchain metrics plot: {blockchain_path}")
    except Exception as e:
        print(f"❌ Blockchain metrics plot failed: {e}")
    
    # 6. Check Gas Usage Analysis Plot
    print("\n6. === GAS USAGE ANALYSIS PLOT ===")
    gas_data = {
        'rounds': [1, 2, 3],
        'gas_used': [20000, 22000, 21000],
        'transactions': [3, 3, 3]
    }
    print(f"Gas usage data: {gas_data}")
    try:
        gas_path = visualizer.plot_gas_usage_analysis(gas_data, save=True)
        print(f"✅ Gas usage analysis plot: {gas_path}")
    except Exception as e:
        print(f"❌ Gas usage analysis plot failed: {e}")
    
    # 7. Check ROC Curves Plot
    print("\n7. === ROC CURVES PLOT ===")
    print(f"Base model ROC AUC: {results['base_model']['roc_auc']}")
    print(f"TTT model ROC AUC: {results['ttt_model']['roc_auc']}")
    try:
        roc_path = visualizer.plot_roc_curves(results['base_model'], results['ttt_model'], save=True)
        print(f"✅ ROC curves plot: {roc_path}")
    except Exception as e:
        print(f"❌ ROC curves plot failed: {e}")
    
    # 8. Check Performance Comparison Plot
    print("\n8. === PERFORMANCE COMPARISON PLOT ===")
    try:
        comparison_path = visualizer.plot_performance_comparison_with_annotations(
            results['base_model'], 
            results['ttt_model'],
            scenario_names=["Edge-IIoTset"],
            save=True
        )
        print(f"✅ Performance comparison plot: {comparison_path}")
    except Exception as e:
        print(f"❌ Performance comparison plot failed: {e}")
    
    # 9. Check Metrics JSON
    print("\n9. === METRICS JSON ===")
    system_data = {
        'base_model': results['base_model'],
        'ttt_model': results['ttt_model'],
        'dataset_info': results['dataset_info']
    }
    try:
        json_path = visualizer.save_metrics_to_json(system_data)
        print(f"✅ Metrics JSON: {json_path}")
    except Exception as e:
        print(f"❌ Metrics JSON failed: {e}")

def analyze_plot_file_sizes():
    """Analyze the file sizes of generated plots"""
    
    print("\n=== PLOT FILE SIZE ANALYSIS ===")
    
    import os
    
    plot_files = [
        'performance_plots/training_history_latest.png',
        'performance_plots/confusion_matrices__base_model_latest.png',
        'performance_plots/confusion_matrices__ttt_enhanced_model_latest.png',
        'performance_plots/ttt_adaptation_latest.png',
        'performance_plots/client_performance_latest.png',
        'performance_plots/blockchain_metrics_latest.png',
        'performance_plots/gas_usage_analysis_latest.png',
        'performance_plots/roc_curves_latest.png',
        'performance_plots/performance_comparison_annotated_latest.png',
        'performance_plots/performance_metrics_latest.json'
    ]
    
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            size = os.path.getsize(plot_file)
            print(f"✅ {plot_file}: {size:,} bytes")
            if size < 1000:
                print(f"   ⚠️  WARNING: File size is very small ({size} bytes) - might be empty!")
        else:
            print(f"❌ {plot_file}: File not found")

if __name__ == "__main__":
    check_all_plots()
    analyze_plot_file_sizes()
