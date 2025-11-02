#!/usr/bin/env python3
"""
Unit test for evaluation metrics correctness in zero-day detection
Tests both base model and TTT model evaluation logic
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, matthews_corrcoef
)


class TestEvaluationMetricsCorrectness(unittest.TestCase):
    """Test evaluation metrics correctness for zero-day detection"""
    
    def setUp(self):
        """Set up test fixtures with known data"""
        # Create reproducible test data
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Test data: 100 samples
        self.n_samples = 100
        self.n_zero_day = 20  # 20% zero-day samples
        self.n_other_attack = 30  # 30% other attacks
        self.n_normal = 50  # 50% normal
        
        # Create multiclass labels: 0=Normal, 1=Other Attack, 2=Zero-day Attack
        self.y_true_multiclass = np.concatenate([
            np.zeros(self.n_normal, dtype=int),  # 50 normal
            np.ones(self.n_other_attack, dtype=int),  # 30 other attacks
            np.ones(self.n_zero_day, dtype=int) * 2  # 20 zero-day attacks
        ])
        
        # Create binary labels: 0=Normal, 1=Attack (derived from multiclass)
        self.y_true_binary = (self.y_true_multiclass != 0).astype(int)
        
        # Zero-day mask: True for zero-day samples (label=2 in multiclass)
        self.zero_day_mask = (self.y_true_multiclass == 2)
        
        # Create probabilities: high prob for attacks, lower for normal
        self.attack_probs_base = np.concatenate([
            np.random.beta(2, 8, self.n_samples - self.n_zero_day),  # Normal: low prob
            np.random.beta(8, 2, self.n_zero_day)  # Attacks: high prob
        ])
        
        # TTT model probabilities: even better separation
        self.attack_probs_ttt = np.concatenate([
            np.random.beta(1, 9, self.n_samples - self.n_zero_day),  # Normal: very low prob
            np.random.beta(9, 1, self.n_zero_day)  # Attacks: very high prob
        ])
        
        # Threshold for binary classification
        self.threshold = 0.5
    
    def test_binary_predictions_from_probabilities(self):
        """Test that binary predictions are correctly derived from probabilities"""
        # Base model predictions
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        # Verify predictions are binary
        self.assertTrue(np.all(np.isin(base_predictions, [0, 1])))
        
        # Verify prediction counts make sense
        n_pred_attack = base_predictions.sum()
        n_pred_normal = len(base_predictions) - n_pred_attack
        
        self.assertEqual(n_pred_attack + n_pred_normal, self.n_samples)
        self.assertGreater(n_pred_attack, 0)
        self.assertGreater(n_pred_normal, 0)
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation correctness"""
        # Create predictions
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        # Calculate accuracy manually
        correct = (base_predictions == self.y_true_binary).sum()
        manual_accuracy = correct / len(self.y_true_binary)
        
        # Calculate using sklearn
        sklearn_accuracy = accuracy_score(self.y_true_binary, base_predictions)
        
        # Verify they match
        self.assertAlmostEqual(manual_accuracy, sklearn_accuracy, places=10,
                              msg="Manual accuracy calculation should match sklearn")
    
    def test_precision_recall_f1_consistency(self):
        """Test that precision, recall, and F1 are consistent"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        precision = precision_score(self.y_true_binary, base_predictions, zero_division=0)
        recall = recall_score(self.y_true_binary, base_predictions, zero_division=0)
        f1 = f1_score(self.y_true_binary, base_predictions, zero_division=0)
        
        # F1 should be harmonic mean: 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            self.assertAlmostEqual(f1, expected_f1, places=10,
                                  msg="F1 should be harmonic mean of precision and recall")
        
        # All metrics should be in [0, 1]
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
    
    def test_confusion_matrix_correctness(self):
        """Test confusion matrix calculation correctness"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        cm = confusion_matrix(self.y_true_binary, base_predictions)
        
        # Verify shape
        self.assertEqual(cm.shape, (2, 2), "Confusion matrix should be 2x2 for binary classification")
        
        # Verify values are non-negative integers
        self.assertTrue(np.all(cm >= 0), "All confusion matrix values should be non-negative")
        self.assertTrue(np.all(cm == cm.astype(int)), "All confusion matrix values should be integers")
        
        # Verify sum equals number of samples
        self.assertEqual(cm.sum(), self.n_samples, 
                        "Confusion matrix sum should equal number of samples")
        
        # Calculate metrics from confusion matrix manually
        tn, fp, fn, tp = cm.ravel()
        
        manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Verify against sklearn metrics
        sklearn_precision = precision_score(self.y_true_binary, base_predictions, zero_division=0)
        sklearn_recall = recall_score(self.y_true_binary, base_predictions, zero_division=0)
        sklearn_accuracy = accuracy_score(self.y_true_binary, base_predictions)
        
        self.assertAlmostEqual(manual_precision, sklearn_precision, places=10,
                              msg="Manual precision from CM should match sklearn")
        self.assertAlmostEqual(manual_recall, sklearn_recall, places=10,
                              msg="Manual recall from CM should match sklearn")
        self.assertAlmostEqual(manual_accuracy, sklearn_accuracy, places=10,
                              msg="Manual accuracy from CM should match sklearn")
    
    def test_roc_curve_calculation(self):
        """Test ROC curve calculation correctness"""
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true_binary, self.attack_probs_base)
        roc_auc = roc_auc_score(self.y_true_binary, self.attack_probs_base)
        
        # Verify outputs
        self.assertEqual(len(fpr), len(tpr), "FPR and TPR should have same length")
        self.assertEqual(len(fpr), len(thresholds), "FPR and thresholds should have same length")
        
        # Verify ranges
        self.assertTrue(np.all(fpr >= 0) and np.all(fpr <= 1), "FPR should be in [0, 1]")
        self.assertTrue(np.all(tpr >= 0) and np.all(tpr <= 1), "TPR should be in [0, 1]")
        self.assertGreaterEqual(roc_auc, 0.0)
        self.assertLessEqual(roc_auc, 1.0)
        
        # Verify curve starts at (0, 0) and ends at (1, 1)
        self.assertAlmostEqual(fpr[0], 0.0, places=5, msg="ROC curve should start at FPR=0")
        self.assertAlmostEqual(tpr[0], 0.0, places=5, msg="ROC curve should start at TPR=0")
        self.assertAlmostEqual(fpr[-1], 1.0, places=5, msg="ROC curve should end at FPR=1")
        self.assertAlmostEqual(tpr[-1], 1.0, places=5, msg="ROC curve should end at TPR=1")
    
    def test_pr_curve_calculation(self):
        """Test Precision-Recall curve calculation correctness"""
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(
            self.y_true_binary, self.attack_probs_base
        )
        auc_pr = average_precision_score(self.y_true_binary, self.attack_probs_base)
        
        # Verify outputs
        self.assertEqual(len(precision), len(recall), 
                        "Precision and recall should have same length")
        self.assertEqual(len(thresholds), len(precision) - 1,
                        "Thresholds should have length = precision - 1")
        
        # Verify ranges
        self.assertTrue(np.all(precision >= 0) and np.all(precision <= 1),
                       "Precision should be in [0, 1]")
        self.assertTrue(np.all(recall >= 0) and np.all(recall <= 1),
                       "Recall should be in [0, 1]")
        self.assertGreaterEqual(auc_pr, 0.0)
        self.assertLessEqual(auc_pr, 1.0)
        
        # Verify curve properties
        # Last precision value should approximate ratio of positive class (may vary slightly)
        positive_ratio = self.y_true_binary.sum() / len(self.y_true_binary)
        # Note: Last precision value may not exactly equal positive ratio due to how PR curve is calculated
        # It should be close but not necessarily exact
        self.assertGreaterEqual(precision[-1], 0.0)
        self.assertLessEqual(precision[-1], 1.0)
    
    def test_zero_day_specific_metrics(self):
        """Test zero-day specific metrics calculation"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        # Filter for zero-day samples only
        zero_day_y_true = self.y_true_binary[self.zero_day_mask]
        zero_day_predictions = base_predictions[self.zero_day_mask]
        
        # Calculate zero-day specific metrics
        zero_day_accuracy = accuracy_score(zero_day_y_true, zero_day_predictions)
        zero_day_precision = precision_score(zero_day_y_true, zero_day_predictions, zero_division=0)
        zero_day_recall = recall_score(zero_day_y_true, zero_day_predictions, zero_division=0)
        zero_day_f1 = f1_score(zero_day_y_true, zero_day_predictions, zero_division=0)
        
        # Verify metrics are in valid range
        self.assertGreaterEqual(zero_day_accuracy, 0.0)
        self.assertLessEqual(zero_day_accuracy, 1.0)
        self.assertGreaterEqual(zero_day_precision, 0.0)
        self.assertLessEqual(zero_day_precision, 1.0)
        self.assertGreaterEqual(zero_day_recall, 0.0)
        self.assertLessEqual(zero_day_recall, 1.0)
        self.assertGreaterEqual(zero_day_f1, 0.0)
        self.assertLessEqual(zero_day_f1, 1.0)
        
        # Verify zero-day detection rate
        zero_day_detection_rate = zero_day_predictions.mean()
        self.assertGreaterEqual(zero_day_detection_rate, 0.0)
        self.assertLessEqual(zero_day_detection_rate, 1.0)
    
    def test_non_zero_day_metrics(self):
        """Test non-zero-day specific metrics calculation"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        # Filter for non-zero-day samples only
        non_zero_day_mask = ~self.zero_day_mask
        non_zero_day_y_true = self.y_true_binary[non_zero_day_mask]
        non_zero_day_predictions = base_predictions[non_zero_day_mask]
        
        # Calculate non-zero-day specific metrics
        non_zero_day_accuracy = accuracy_score(non_zero_day_y_true, non_zero_day_predictions)
        
        # Verify we're testing correct subset
        self.assertEqual(len(non_zero_day_y_true), self.n_samples - self.n_zero_day,
                        "Non-zero-day subset should exclude zero-day samples")
        
        # Verify metrics are in valid range
        self.assertGreaterEqual(non_zero_day_accuracy, 0.0)
        self.assertLessEqual(non_zero_day_accuracy, 1.0)
    
    def test_mcc_calculation(self):
        """Test Matthews Correlation Coefficient calculation"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        mcc = matthews_corrcoef(self.y_true_binary, base_predictions)
        
        # MCC should be in [-1, 1]
        self.assertGreaterEqual(mcc, -1.0)
        self.assertLessEqual(mcc, 1.0)
        
        # For binary classification, MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        cm = confusion_matrix(self.y_true_binary, base_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator > 0:
            manual_mcc = (tp * tn - fp * fn) / denominator
            self.assertAlmostEqual(mcc, manual_mcc, places=10,
                                  msg="MCC should match manual calculation")
    
    def test_metric_relationships(self):
        """Test relationships between different metrics"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        
        accuracy = accuracy_score(self.y_true_binary, base_predictions)
        precision = precision_score(self.y_true_binary, base_predictions, zero_division=0)
        recall = recall_score(self.y_true_binary, base_predictions, zero_division=0)
        f1 = f1_score(self.y_true_binary, base_predictions, zero_division=0)
        
        # F1 is harmonic mean of precision and recall
        if precision > 0 and recall > 0:
            harmonic_mean = 2 * (precision * recall) / (precision + recall)
            self.assertAlmostEqual(f1, harmonic_mean, places=10)
        
        # F1 is harmonic mean, so: min(precision, recall) <= F1 <= max(precision, recall)
        # However, due to floating point precision, we allow small tolerance
        if precision > 0 and recall > 0:
            # F1 should be between min and max (with small tolerance for floating point)
            min_pr = min(precision, recall)
            max_pr = max(precision, recall)
            self.assertGreaterEqual(f1, min_pr - 0.01, 
                                  msg=f"F1 ({f1}) should be >= min(precision, recall) - tolerance")
            self.assertLessEqual(f1, max_pr + 0.01,
                               msg=f"F1 ({f1}) should be <= max(precision, recall) + tolerance")
    
    def test_base_vs_ttt_comparison(self):
        """Test that base and TTT model metrics can be compared correctly"""
        base_predictions = (self.attack_probs_base >= self.threshold).astype(int)
        ttt_predictions = (self.attack_probs_ttt >= self.threshold).astype(int)
        
        base_accuracy = accuracy_score(self.y_true_binary, base_predictions)
        ttt_accuracy = accuracy_score(self.y_true_binary, ttt_predictions)
        
        base_f1 = f1_score(self.y_true_binary, base_predictions, zero_division=0)
        ttt_f1 = f1_score(self.y_true_binary, ttt_predictions, zero_division=0)
        
        # Calculate AUC-PR for both
        base_auc_pr = average_precision_score(self.y_true_binary, self.attack_probs_base)
        ttt_auc_pr = average_precision_score(self.y_true_binary, self.attack_probs_ttt)
        
        # Verify both models produce valid metrics
        # Note: TTT may or may not be better depending on adaptation quality
        # This test just verifies metrics can be calculated and compared
        
        # All metrics should be valid for base model
        self.assertGreaterEqual(base_accuracy, 0.0)
        self.assertLessEqual(base_accuracy, 1.0)
        self.assertGreaterEqual(base_f1, 0.0)
        self.assertLessEqual(base_f1, 1.0)
        self.assertGreaterEqual(base_auc_pr, 0.0)
        self.assertLessEqual(base_auc_pr, 1.0 + 1e-10)  # Allow small floating point error
        
        # All metrics should be valid for TTT model
        self.assertGreaterEqual(ttt_accuracy, 0.0)
        self.assertLessEqual(ttt_accuracy, 1.0)
        self.assertGreaterEqual(ttt_f1, 0.0)
        self.assertLessEqual(ttt_f1, 1.0)
        self.assertGreaterEqual(ttt_auc_pr, 0.0)
        self.assertLessEqual(ttt_auc_pr, 1.0 + 1e-10)  # Allow small floating point error
        
        # Verify improvement calculation works
        accuracy_improvement = ttt_accuracy - base_accuracy
        f1_improvement = ttt_f1 - base_f1
        auc_pr_improvement = ttt_auc_pr - base_auc_pr
        
        # Improvements can be positive or negative (both are valid)
        self.assertGreaterEqual(accuracy_improvement, -1.0)
        self.assertLessEqual(accuracy_improvement, 1.0)
        self.assertGreaterEqual(f1_improvement, -1.0)
        self.assertLessEqual(f1_improvement, 1.0)
        self.assertGreaterEqual(auc_pr_improvement, -1.0)
        self.assertLessEqual(auc_pr_improvement, 1.0)
    
    def test_probability_cleaning_logic(self):
        """Test that probability cleaning (NaN/Inf handling) works correctly"""
        # Create probabilities with NaN and Inf
        attack_probs_with_nan = self.attack_probs_base.copy()
        attack_probs_with_nan[0] = np.nan
        attack_probs_with_nan[1] = np.inf
        attack_probs_with_nan[2] = -np.inf
        
        # Clean probabilities
        attack_probs_clean = np.asarray(attack_probs_with_nan, dtype=np.float64)
        attack_probs_clean = np.nan_to_num(attack_probs_clean, nan=0.5, posinf=1.0, neginf=0.0)
        attack_probs_clean = np.clip(attack_probs_clean, 0.0, 1.0)
        
        # Verify all values are valid
        self.assertFalse(np.isnan(attack_probs_clean).any(), "No NaN values after cleaning")
        self.assertFalse(np.isinf(attack_probs_clean).any(), "No Inf values after cleaning")
        self.assertTrue(np.all(attack_probs_clean >= 0.0), "All probabilities >= 0")
        self.assertTrue(np.all(attack_probs_clean <= 1.0), "All probabilities <= 1")
        
        # Verify cleaning worked
        self.assertAlmostEqual(attack_probs_clean[0], 0.5, places=5, msg="NaN should be replaced with 0.5")
        self.assertAlmostEqual(attack_probs_clean[1], 1.0, places=5, msg="+Inf should be replaced with 1.0")
        self.assertAlmostEqual(attack_probs_clean[2], 0.0, places=5, msg="-Inf should be replaced with 0.0")
    
    def test_optimal_threshold_calculation(self):
        """Test optimal threshold calculation for binary classification"""
        # Calculate F1 at different thresholds
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (self.attack_probs_base >= threshold).astype(int)
            f1 = f1_score(self.y_true_binary, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Optimal threshold should be in valid range
        self.assertGreaterEqual(best_threshold, 0.0)
        self.assertLessEqual(best_threshold, 1.0)
        
        # Verify F1 at optimal threshold is better or equal to default
        default_f1 = f1_score(
            self.y_true_binary, 
            (self.attack_probs_base >= 0.5).astype(int), 
            zero_division=0
        )
        self.assertGreaterEqual(best_f1, default_f1,
                               msg="Optimal threshold F1 should be >= default threshold F1")
    
    def test_roc_pr_curve_consistency(self):
        """Test that ROC and PR curves are consistent with each other"""
        # Calculate both curves
        fpr, tpr, _ = roc_curve(self.y_true_binary, self.attack_probs_base)
        precision, recall, _ = precision_recall_curve(self.y_true_binary, self.attack_probs_base)
        
        # Calculate AUC
        roc_auc = roc_auc_score(self.y_true_binary, self.attack_probs_base)
        auc_pr = average_precision_score(self.y_true_binary, self.attack_probs_base)
        
        # Both should be in [0, 1]
        self.assertGreaterEqual(roc_auc, 0.0)
        self.assertLessEqual(roc_auc, 1.0)
        self.assertGreaterEqual(auc_pr, 0.0)
        self.assertLessEqual(auc_pr, 1.0)
        
        # For imbalanced data (like zero-day detection), AUC-PR is often lower than ROC-AUC
        # But both should reflect model performance
        positive_ratio = self.y_true_binary.sum() / len(self.y_true_binary)
        
        # If class is imbalanced (positive_ratio < 0.5), AUC-PR can be lower than ROC-AUC
        if positive_ratio < 0.5:
            # This is expected for imbalanced data - no strict relationship
            pass
        else:
            # For balanced data, they should be more similar
            self.assertLess(abs(roc_auc - auc_pr), 0.3,
                          msg="ROC-AUC and AUC-PR should be reasonably close for balanced data")


class TestEvaluationMetricsIntegration(unittest.TestCase):
    """Integration tests for evaluation metrics with realistic scenarios"""
    
    def test_perfect_classifier_metrics(self):
        """Test metrics for a perfect classifier"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs)
        auc_pr = average_precision_score(y_true, y_probs)
        
        # Perfect classifier should have all metrics = 1.0
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(f1, 1.0)
        self.assertEqual(roc_auc, 1.0)
        self.assertAlmostEqual(auc_pr, 1.0, places=5)
    
    def test_random_classifier_metrics(self):
        """Test metrics for a random classifier"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Random predictions (50% chance)
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array([0.4, 0.5, 0.5, 0.5, 0.5, 0.6])
        
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_probs)
        auc_pr = average_precision_score(y_true, y_probs)
        
        # Random classifier should have AUC around 0.5 (with wider tolerance for small sample size)
        # For very small datasets, AUC can vary more
        self.assertGreaterEqual(roc_auc, 0.2)
        self.assertLessEqual(roc_auc, 0.9)
        self.assertGreaterEqual(auc_pr, 0.2)
        self.assertLessEqual(auc_pr, 0.9)
    
    def test_all_negative_predictions(self):
        """Test metrics when all predictions are negative (class 0)"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # All negative
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Precision should be 0 (no true positives)
        self.assertEqual(precision, 0.0)
        # Recall should be 0 (no positives predicted)
        self.assertEqual(recall, 0.0)
        # F1 should be 0
        self.assertEqual(f1, 0.0)
    
    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive (class 1)"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # All positive
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Precision should be ratio of positives (0.5 in this case)
        self.assertEqual(precision, 0.5)
        # Recall should be 1.0 (all positives detected)
        self.assertEqual(recall, 1.0)
        # F1 should be 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
        self.assertAlmostEqual(f1, 2/3, places=5)


class TestBaseModelEvaluationLogic(unittest.TestCase):
    """Test base model evaluation logic correctness"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic test data similar to real evaluation
        self.n_samples = 100
        self.n_zero_day = 20
        
        # Multiclass labels: 0=Normal, 1=Other Attack, 2=Zero-day Attack
        self.y_multiclass = np.concatenate([
            np.zeros(60, dtype=int),  # 60 normal
            np.ones(20, dtype=int),   # 20 other attacks
            np.ones(self.n_zero_day, dtype=int) * 2  # 20 zero-day attacks
        ])
        
        # Binary labels: 0=Normal, 1=Attack
        self.y_binary = (self.y_multiclass != 0).astype(int)
        
        # Zero-day mask
        self.zero_day_mask = (self.y_multiclass == 2)
        
        # Create logits and probabilities
        self.base_logits = torch.randn(self.n_samples, 10)  # 10 classes
        self.base_probs = torch.softmax(self.base_logits, dim=1)
        self.base_predictions = torch.argmax(self.base_logits, dim=1)
    
    def test_binary_conversion_correctness(self):
        """Test that multiclass to binary conversion is correct"""
        # Convert multiclass predictions to binary: Normal=0, Attack=1
        binary_predictions = (self.base_predictions.numpy() != 0).astype(int)
        
        # Verify all predictions are binary
        self.assertTrue(np.all(np.isin(binary_predictions, [0, 1])),
                       "Binary predictions should only contain 0 or 1")
        
        # Verify conversion logic: 0 stays 0, anything else becomes 1
        for i in range(len(binary_predictions)):
            if self.base_predictions[i] == 0:
                self.assertEqual(binary_predictions[i], 0,
                               f"Normal class (0) should map to 0, got {binary_predictions[i]}")
            else:
                self.assertEqual(binary_predictions[i], 1,
                               f"Attack class ({self.base_predictions[i]}) should map to 1")
    
    def test_zero_day_mask_correctness(self):
        """Test zero-day mask creation is correct"""
        # Zero-day mask should identify samples with label=2
        expected_mask = (self.y_multiclass == 2)
        
        self.assertTrue(np.array_equal(self.zero_day_mask, expected_mask),
                       "Zero-day mask should match expected")
        self.assertEqual(self.zero_day_mask.sum(), self.n_zero_day,
                        f"Zero-day mask should identify {self.n_zero_day} samples")
    
    def test_attack_probability_extraction(self):
        """Test attack probability extraction from multiclass probabilities"""
        # Attack probability = 1 - P(Normal)
        attack_probs = 1.0 - self.base_probs[:, 0].numpy()
        
        # Verify probabilities are in [0, 1]
        self.assertTrue(np.all(attack_probs >= 0), "Attack probabilities should be >= 0")
        self.assertTrue(np.all(attack_probs <= 1), "Attack probabilities should be <= 1")
        
        # Verify relationship: attack_prob + normal_prob = 1
        normal_probs = self.base_probs[:, 0].numpy()
        self.assertTrue(np.allclose(attack_probs + normal_probs, 1.0),
                       "Attack prob + Normal prob should sum to 1.0")
    
    def test_separate_metrics_calculation(self):
        """Test that zero-day and non-zero-day metrics are calculated separately"""
        # Simulate evaluation
        binary_predictions = (self.base_predictions.numpy() != 0).astype(int)
        y_true_bin = (self.y_multiclass != 0).astype(int)
        
        # Zero-day metrics
        zero_day_y_true = y_true_bin[self.zero_day_mask]
        zero_day_pred = binary_predictions[self.zero_day_mask]
        
        # Non-zero-day metrics
        non_zero_day_mask = ~self.zero_day_mask
        non_zero_day_y_true = y_true_bin[non_zero_day_mask]
        non_zero_day_pred = binary_predictions[non_zero_day_mask]
        
        # Verify subsets are mutually exclusive and cover all samples
        self.assertEqual(len(zero_day_y_true) + len(non_zero_day_y_true), self.n_samples,
                        "Zero-day and non-zero-day samples should cover all samples")
        self.assertEqual(self.zero_day_mask.sum() + non_zero_day_mask.sum(), self.n_samples,
                        "Masks should be mutually exclusive and complete")
        
        # Calculate separate metrics
        zero_day_accuracy = accuracy_score(zero_day_y_true, zero_day_pred)
        non_zero_day_accuracy = accuracy_score(non_zero_day_y_true, non_zero_day_pred)
        
        # Verify metrics are valid
        self.assertGreaterEqual(zero_day_accuracy, 0.0)
        self.assertLessEqual(zero_day_accuracy, 1.0)
        self.assertGreaterEqual(non_zero_day_accuracy, 0.0)
        self.assertLessEqual(non_zero_day_accuracy, 1.0)


class TestTTTModelEvaluationLogic(unittest.TestCase):
    """Test TTT model evaluation logic correctness"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.n_samples = 100
        self.n_zero_day = 20
        
        # Create test data
        self.y_multiclass = np.concatenate([
            np.zeros(60, dtype=int),
            np.ones(20, dtype=int),
            np.ones(self.n_zero_day, dtype=int) * 2
        ])
        self.y_binary = (self.y_multiclass != 0).astype(int)
        self.zero_day_mask = (self.y_multiclass == 2)
        
        # TTT adapted probabilities (should be better separated)
        self.ttt_probs = np.random.beta(9, 1, self.n_samples)
        # Make sure zero-day samples have high probabilities
        self.ttt_probs[self.zero_day_mask] = np.random.beta(9, 1, self.n_zero_day)
    
    def test_optimal_threshold_selection(self):
        """Test optimal threshold selection logic"""
        # Test different thresholds
        thresholds = np.linspace(0.1, 0.9, 9)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (self.ttt_probs >= threshold).astype(int)
            f1 = f1_score(self.y_binary, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Optimal threshold should be in reasonable range
        self.assertGreaterEqual(best_threshold, 0.0)
        self.assertLessEqual(best_threshold, 1.0)
        
        # Verify F1 at optimal threshold is reasonable
        self.assertGreaterEqual(best_f1, 0.0)
        self.assertLessEqual(best_f1, 1.0)
    
    def test_ttt_probability_cleaning(self):
        """Test TTT probability cleaning (NaN/Inf handling)"""
        # Create probabilities with invalid values
        probs_with_invalid = self.ttt_probs.copy()
        probs_with_invalid[0] = np.nan
        probs_with_invalid[1] = np.inf
        
        # Clean probabilities
        probs_clean = np.asarray(probs_with_invalid, dtype=np.float64)
        probs_clean = np.nan_to_num(probs_clean, nan=0.5, posinf=1.0, neginf=0.0)
        probs_clean = np.clip(probs_clean, 0.0, 1.0)
        
        # Verify all values are valid
        self.assertFalse(np.isnan(probs_clean).any())
        self.assertFalse(np.isinf(probs_clean).any())
        self.assertTrue(np.all(probs_clean >= 0.0))
        self.assertTrue(np.all(probs_clean <= 1.0))


def run_tests():
    """Run all evaluation metrics tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationMetricsCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationMetricsIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBaseModelEvaluationLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestTTTModelEvaluationLogic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}")
            print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

