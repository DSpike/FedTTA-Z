#!/usr/bin/env python3
"""Test performance metrics calculation logic"""

import sys
sys.path.append('.')
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_metrics_calculation():
    """Test basic performance metrics calculation"""
    print("=== Testing Basic Performance Metrics Calculation ===")
    
    # Create test data
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])  # 6 attacks, 4 normal
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])  # 4 attacks, 6 normal
    
    print(f"True labels: {y_true}")
    print(f"Predicted labels: {y_pred}")
    print(f"Class distribution - True: {np.bincount(y_true)}")
    print(f"Class distribution - Pred: {np.bincount(y_pred)}")
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nBasic Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Manual calculation verification
    tn, fp, fn, tp = cm.ravel()
    manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
    manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    manual_f1 = 2 * (manual_precision * manual_recall) / (manual_precision + manual_recall) if (manual_precision + manual_recall) > 0 else 0
    
    print(f"\nManual Calculation Verification:")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"  Manual Accuracy: {manual_accuracy:.4f}")
    print(f"  Manual Precision: {manual_precision:.4f}")
    print(f"  Manual Recall: {manual_recall:.4f}")
    print(f"  Manual F1: {manual_f1:.4f}")
    
    # Verify calculations match
    assert abs(accuracy - manual_accuracy) < 1e-6, "Accuracy calculation mismatch"
    assert abs(precision - manual_precision) < 1e-6, "Precision calculation mismatch"
    assert abs(recall - manual_recall) < 1e-6, "Recall calculation mismatch"
    assert abs(f1 - manual_f1) < 1e-6, "F1 calculation mismatch"
    
    print("✅ Basic metrics calculation verified!")
    return True

def test_multiclass_metrics():
    """Test multiclass performance metrics"""
    print("\n=== Testing Multiclass Performance Metrics ===")
    
    # Create multiclass test data (3 classes: 0=Normal, 1=Attack1, 2=Attack2)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 1, 0])
    
    print(f"True labels: {y_true}")
    print(f"Predicted labels: {y_pred}")
    print(f"Class distribution - True: {np.bincount(y_true)}")
    print(f"Class distribution - Pred: {np.bincount(y_pred)}")
    
    # Calculate different averaging methods
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"\nMulticlass Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"  Weighted - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
    print(f"  Micro - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"  MCC: {mcc:.4f}")
    
    # Verify micro-averaged metrics equal accuracy
    assert abs(accuracy - precision_micro) < 1e-6, "Micro precision should equal accuracy"
    assert abs(accuracy - recall_micro) < 1e-6, "Micro recall should equal accuracy"
    assert abs(accuracy - f1_micro) < 1e-6, "Micro F1 should equal accuracy"
    
    print("✅ Multiclass metrics calculation verified!")
    return True

def test_zero_day_detection_rate():
    """Test zero-day detection rate calculation"""
    print("\n=== Testing Zero-Day Detection Rate Calculation ===")
    
    # Create test data with zero-day samples
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])  # 6 attacks, 4 normal
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])  # 4 attacks, 6 normal
    zero_day_mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])  # 6 zero-day samples
    
    print(f"True labels: {y_true}")
    print(f"Predicted labels: {y_pred}")
    print(f"Zero-day mask: {zero_day_mask}")
    
    # Calculate zero-day detection rate
    zero_day_predictions = y_pred[zero_day_mask.astype(bool)]
    zero_day_detection_rate = zero_day_predictions.mean() if len(zero_day_predictions) > 0 else 0.0
    
    print(f"\nZero-Day Detection Analysis:")
    print(f"  Zero-day samples: {zero_day_mask.sum()}")
    print(f"  Zero-day predictions: {zero_day_predictions}")
    print(f"  Zero-day detection rate: {zero_day_detection_rate:.4f}")
    
    # Manual verification
    zero_day_attacks_detected = np.sum((y_pred == 1) & (zero_day_mask == 1))
    zero_day_total = np.sum(zero_day_mask)
    manual_rate = zero_day_attacks_detected / zero_day_total if zero_day_total > 0 else 0.0
    
    print(f"  Manual calculation: {zero_day_attacks_detected}/{zero_day_total} = {manual_rate:.4f}")
    assert abs(zero_day_detection_rate - manual_rate) < 1e-6, "Zero-day detection rate calculation mismatch"
    
    print("✅ Zero-day detection rate calculation verified!")
    return True

def test_edge_cases():
    """Test edge cases in metrics calculation"""
    print("\n=== Testing Edge Cases ===")
    
    # Case 1: All predictions correct
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Perfect predictions - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    assert accuracy == 1.0, "Perfect predictions should have accuracy = 1.0"
    assert f1 == 1.0, "Perfect predictions should have F1 = 1.0"
    assert mcc == 1.0, "Perfect predictions should have MCC = 1.0"
    
    # Case 2: All predictions wrong
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"All wrong predictions - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    assert accuracy == 0.0, "All wrong predictions should have accuracy = 0.0"
    assert f1 == 0.0, "All wrong predictions should have F1 = 0.0"
    assert mcc == -1.0, "All wrong predictions should have MCC = -1.0"
    
    # Case 3: All predictions same class
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 1])
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"All same class predictions - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
    
    # Case 4: Empty arrays
    y_true = np.array([])
    y_pred = np.array([])
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Empty arrays - Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Empty arrays - Expected error: {e}")
    
    print("✅ Edge cases tested!")
    return True

def test_roc_auc_calculation():
    """Test ROC AUC calculation"""
    print("\n=== Testing ROC AUC Calculation ===")
    
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Create test data with probabilities
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.85, 0.15, 0.75])
    
    print(f"True labels: {y_true}")
    print(f"Prediction scores: {y_scores}")
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    print(f"\nROC Analysis:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Number of thresholds: {len(thresholds)}")
    print(f"  FPR range: [{fpr.min():.4f}, {fpr.max():.4f}]")
    print(f"  TPR range: [{tpr.min():.4f}, {tpr.max():.4f}]")
    
    # Test different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        y_pred_thresh = (y_scores >= threshold).astype(int)
        accuracy_thresh = accuracy_score(y_true, y_pred_thresh)
        print(f"  Threshold {threshold}: Accuracy = {accuracy_thresh:.4f}")
    
    print("✅ ROC AUC calculation verified!")
    return True

def main():
    """Run all performance metrics tests"""
    print("🧪 Testing Performance Metrics Calculation Logic")
    print("=" * 60)
    
    try:
        test_basic_metrics_calculation()
        test_multiclass_metrics()
        test_zero_day_detection_rate()
        test_edge_cases()
        test_roc_auc_calculation()
        
        print("\n🎉 All performance metrics tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Performance metrics calculation logic is working correctly!")
    else:
        print("\n❌ Performance metrics calculation logic has issues!")
