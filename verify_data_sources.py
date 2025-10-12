#!/usr/bin/env python3
"""
Script to verify if ALL plot data is real system data vs placeholder data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from visualization.performance_visualization import PerformanceVisualizer

def verify_all_plot_data_sources():
    """Verify if all plot data comes from real system execution"""
    
    print("=== VERIFYING ALL PLOT DATA SOURCES ===")
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded successfully. Keys: {list(results.keys())}")
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # 1. CHECK TRAINING HISTORY PLOT DATA SOURCE
    print("\n1. === TRAINING HISTORY PLOT DATA SOURCE ===")
    check_training_history_data_source(visualizer, results)
    
    # 2. CHECK TTT ADAPTATION PLOT DATA SOURCE
    print("\n2. === TTT ADAPTATION PLOT DATA SOURCE ===")
    check_ttt_adaptation_data_source(visualizer, results)
    
    # 3. CHECK CLIENT PERFORMANCE PLOT DATA SOURCE
    print("\n3. === CLIENT PERFORMANCE PLOT DATA SOURCE ===")
    check_client_performance_data_source(visualizer, results)
    
    # 4. CHECK BLOCKCHAIN METRICS PLOT DATA SOURCE
    print("\n4. === BLOCKCHAIN METRICS PLOT DATA SOURCE ===")
    check_blockchain_metrics_data_source(visualizer, results)
    
    # 5. CHECK GAS USAGE PLOT DATA SOURCE
    print("\n5. === GAS USAGE PLOT DATA SOURCE ===")
    check_gas_usage_data_source(visualizer, results)
    
    # 6. CHECK CONFUSION MATRICES DATA SOURCE
    print("\n6. === CONFUSION MATRICES DATA SOURCE ===")
    check_confusion_matrices_data_source(visualizer, results)
    
    # 7. CHECK ROC CURVES DATA SOURCE
    print("\n7. === ROC CURVES DATA SOURCE ===")
    check_roc_curves_data_source(visualizer, results)
    
    # 8. CHECK PERFORMANCE COMPARISON DATA SOURCE
    print("\n8. === PERFORMANCE COMPARISON DATA SOURCE ===")
    check_performance_comparison_data_source(visualizer, results)

def check_training_history_data_source(visualizer, results):
    """Check if training history data is real or placeholder"""
    
    print("Checking training history data source...")
    
    # Check what data is actually being passed to the training history plot
    # This is likely placeholder data since we don't see real training history in results
    
    training_history = {
        'epoch_losses': [0.5, 0.4, 0.3],
        'epoch_accuracies': [0.5, 0.6, 0.7]
    }
    
    print(f"Training history data: {training_history}")
    print("⚠️  WARNING: This appears to be PLACEHOLDER data, not real training history!")
    print("   Real training history should come from actual federated learning rounds.")
    
    # Check if there's real training history in results
    if 'training_history' in results:
        print(f"✅ Found real training history in results: {results['training_history']}")
    else:
        print("❌ No real training history found in results - using placeholder data")

def check_ttt_adaptation_data_source(visualizer, results):
    """Check if TTT adaptation data is real or placeholder"""
    
    print("Checking TTT adaptation data source...")
    
    # Check what data is actually being passed to the TTT adaptation plot
    ttt_adaptation_data = {
        'steps': list(range(10)),
        'total_losses': [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
        'support_losses': [0.3, 0.28, 0.25, 0.22, 0.18, 0.15, 0.12, 0.08, 0.05, 0.03],
        'consistency_losses': [0.2, 0.17, 0.15, 0.13, 0.12, 0.1, 0.08, 0.07, 0.05, 0.02]
    }
    
    print(f"TTT adaptation data: {ttt_adaptation_data}")
    print("⚠️  WARNING: This appears to be PLACEHOLDER data, not real TTT adaptation!")
    print("   Real TTT adaptation should come from actual TTT steps during evaluation.")
    
    # Check if there's real TTT adaptation data in results
    if 'ttt_adaptation_data' in results:
        print(f"✅ Found real TTT adaptation data in results: {results['ttt_adaptation_data']}")
    else:
        print("❌ No real TTT adaptation data found in results - using placeholder data")

def check_client_performance_data_source(visualizer, results):
    """Check if client performance data is real or placeholder"""
    
    print("Checking client performance data source...")
    
    # Check what data is actually being passed to the client performance plot
    client_results = [
        {'client_id': 'client_1', 'accuracy': 0.61, 'f1_score': 0.58, 'precision': 0.62, 'recall': 0.60},
        {'client_id': 'client_2', 'accuracy': 0.64, 'f1_score': 0.61, 'precision': 0.65, 'recall': 0.63},
        {'client_id': 'client_3', 'accuracy': 0.65, 'f1_score': 0.63, 'precision': 0.67, 'recall': 0.65}
    ]
    
    print(f"Client performance data: {client_results}")
    print("⚠️  WARNING: This appears to be PLACEHOLDER data, not real client performance!")
    print("   Real client performance should come from actual federated learning rounds.")
    
    # Check if there's real client performance data in results
    if 'client_performance' in results:
        print(f"✅ Found real client performance data in results: {results['client_performance']}")
    else:
        print("❌ No real client performance data found in results - using placeholder data")

def check_blockchain_metrics_data_source(visualizer, results):
    """Check if blockchain metrics data is real or placeholder"""
    
    print("Checking blockchain metrics data source...")
    
    # Check what data is actually being passed to the blockchain metrics plot
    blockchain_data = {
        'rounds': [1, 2, 3],
        'gas_used': [20000, 22000, 21000],
        'transactions': [3, 3, 3]
    }
    
    print(f"Blockchain metrics data: {blockchain_data}")
    print("⚠️  WARNING: This appears to be PLACEHOLDER data, not real blockchain metrics!")
    print("   Real blockchain metrics should come from actual blockchain transactions.")
    
    # Check if there's real blockchain data in results
    if 'blockchain_metrics' in results:
        print(f"✅ Found real blockchain metrics in results: {results['blockchain_metrics']}")
    else:
        print("❌ No real blockchain metrics found in results - using placeholder data")

def check_gas_usage_data_source(visualizer, results):
    """Check if gas usage data is real or placeholder"""
    
    print("Checking gas usage data source...")
    
    # Check what data is actually being passed to the gas usage plot
    gas_data = {
        'rounds': [1, 2, 3],
        'gas_used': [20000, 22000, 21000],
        'transactions': [3, 3, 3]
    }
    
    print(f"Gas usage data: {gas_data}")
    print("⚠️  WARNING: This appears to be PLACEHOLDER data, not real gas usage!")
    print("   Real gas usage should come from actual blockchain transactions.")
    
    # Check if there's real gas usage data in results
    if 'gas_usage' in results:
        print(f"✅ Found real gas usage data in results: {results['gas_usage']}")
    else:
        print("❌ No real gas usage data found in results - using placeholder data")

def check_confusion_matrices_data_source(visualizer, results):
    """Check if confusion matrices data is real or placeholder"""
    
    print("Checking confusion matrices data source...")
    
    base_cm = results['base_model']['confusion_matrix']
    ttt_cm = results['ttt_model']['confusion_matrix']
    
    print(f"Base model confusion matrix: {base_cm}")
    print(f"TTT model confusion matrix: {ttt_cm}")
    
    # Check if confusion matrices are real
    base_total = sum(sum(row) for row in base_cm)
    ttt_total = sum(sum(row) for row in ttt_cm)
    
    print(f"Base model total samples: {base_total}")
    print(f"TTT model total samples: {ttt_total}")
    
    if base_total > 0 and ttt_total > 0:
        print("✅ Confusion matrices contain REAL data from actual model evaluation")
    else:
        print("❌ Confusion matrices appear to be empty or placeholder data")

def check_roc_curves_data_source(visualizer, results):
    """Check if ROC curves data is real or placeholder"""
    
    print("Checking ROC curves data source...")
    
    base_roc = results['base_model']['roc_curve']
    ttt_roc = results['ttt_model']['roc_curve']
    
    print(f"Base model ROC curve points: {len(base_roc['fpr'])}")
    print(f"TTT model ROC curve points: {len(ttt_roc['fpr'])}")
    print(f"Base model ROC AUC: {results['base_model']['roc_auc']:.4f}")
    print(f"TTT model ROC AUC: {results['ttt_model']['roc_auc']:.4f}")
    
    if len(base_roc['fpr']) > 0 and len(ttt_roc['fpr']) > 0:
        print("✅ ROC curves contain REAL data from actual model evaluation")
    else:
        print("❌ ROC curves appear to be empty or placeholder data")

def check_performance_comparison_data_source(visualizer, results):
    """Check if performance comparison data is real or placeholder"""
    
    print("Checking performance comparison data source...")
    
    base_accuracy = results['base_model']['accuracy_mean']
    ttt_accuracy = results['ttt_model']['accuracy_mean']
    
    print(f"Base model accuracy: {base_accuracy:.4f}")
    print(f"TTT model accuracy: {ttt_accuracy:.4f}")
    
    if base_accuracy > 0 and ttt_accuracy > 0:
        print("✅ Performance comparison contains REAL data from actual model evaluation")
    else:
        print("❌ Performance comparison appears to be empty or placeholder data")

def create_summary_report():
    """Create a summary report of data source verification"""
    
    print("\n=== DATA SOURCE VERIFICATION SUMMARY ===")
    print("REAL DATA (from actual system execution):")
    print("  ✅ Confusion Matrices - Real evaluation results")
    print("  ✅ ROC Curves - Real model performance")
    print("  ✅ Performance Comparison - Real calculated metrics")
    print("")
    print("PLACEHOLDER DATA (not from actual system execution):")
    print("  ❌ Training History - Placeholder values")
    print("  ❌ TTT Adaptation - Placeholder values")
    print("  ❌ Client Performance - Placeholder values")
    print("  ❌ Blockchain Metrics - Placeholder values")
    print("  ❌ Gas Usage Analysis - Placeholder values")
    print("")
    print("RECOMMENDATION: Update the system to collect and use real data for all plots!")

if __name__ == "__main__":
    verify_all_plot_data_sources()
    create_summary_report()
