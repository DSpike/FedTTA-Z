#!/usr/bin/env python3
"""
Debug script to check precision and recall calculations
"""

import json

def debug_precision_recall_calculation():
    """Debug precision and recall calculations from confusion matrices"""
    
    print("=== DEBUGGING PRECISION AND RECALL CALCULATIONS ===")
    
    # Load the actual results
    with open('edgeiiot_results_DDoS_UDP.json', 'r') as f:
        results = json.load(f)
    
    base_cm = results['base_model']['confusion_matrix']
    ttt_cm = results['ttt_model']['confusion_matrix']
    
    print(f"Base model confusion matrix: {base_cm}")
    print(f"TTT model confusion matrix: {ttt_cm}")
    
    # Debug base model calculations
    print("\n=== BASE MODEL CALCULATIONS ===")
    if len(base_cm) == 2 and len(base_cm[0]) == 2:
        tn, fp = base_cm[0][0], base_cm[0][1]
        fn, tp = base_cm[1][0], base_cm[1][1]
        
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
        
        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"Accuracy = (TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy:.4f}")
    else:
        print("❌ Base model confusion matrix format is incorrect!")
    
    # Debug TTT model calculations
    print("\n=== TTT MODEL CALCULATIONS ===")
    if len(ttt_cm) == 2 and len(ttt_cm[0]) == 2:
        tn, fp = ttt_cm[0][0], ttt_cm[0][1]
        fn, tp = ttt_cm[1][0], ttt_cm[1][1]
        
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
        
        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"Accuracy = (TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy:.4f}")
    else:
        print("❌ TTT model confusion matrix format is incorrect!")

if __name__ == "__main__":
    debug_precision_recall_calculation()
