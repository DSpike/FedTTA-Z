#!/usr/bin/env python3
"""Investigate why base model is performing high"""

import json
import numpy as np

# Load performance metrics
with open('performance_plots/performance_metrics_.json', 'r') as f:
    data = json.load(f)

base = data.get('evaluation_results', {}).get('base_model', {})
cm = base.get('confusion_matrix', [])

print("=" * 60)
print("BASE MODEL PERFORMANCE INVESTIGATION")
print("=" * 60)

print("\n1. CONFUSION MATRIX:")
print(f"   TN (Normal predicted as Normal): {cm[0][0]}")
print(f"   FP (Normal predicted as Attack): {cm[0][1]}")
print(f"   FN (Attack predicted as Normal): {cm[1][0]}")
print(f"   TP (Attack predicted as Attack): {cm[1][1]}")

total_samples = sum([cm[0][0], cm[0][1], cm[1][0], cm[1][1]])
print(f"\n   Total Samples: {total_samples}")

print("\n2. CLASS DISTRIBUTION IN TEST SET:")
normal_in_test = cm[0][0] + cm[0][1]  # Normal samples in test
attack_in_test = cm[1][0] + cm[1][1]  # Attack samples in test
print(f"   Normal samples: {normal_in_test} ({normal_in_test/total_samples*100:.1f}%)")
print(f"   Attack samples: {attack_in_test} ({attack_in_test/total_samples*100:.1f}%)")

print("\n3. MODEL PREDICTIONS:")
normal_predicted = cm[0][0] + cm[1][0]  # All Normal predictions
attack_predicted = cm[0][1] + cm[1][1]  # All Attack predictions
print(f"   Predicted Normal: {normal_predicted} ({normal_predicted/total_samples*100:.1f}%)")
print(f"   Predicted Attack: {attack_predicted} ({attack_predicted/total_samples*100:.1f}%)")

print("\n4. METRICS:")
print(f"   Accuracy: {base.get('accuracy', 0):.4f}")
print(f"   Precision: {base.get('precision', 0):.4f}")
print(f"   Recall: {base.get('recall', 0):.4f}")
print(f"   F1-Score: {base.get('f1_score', 0):.4f}")
print(f"   MCC: {base.get('mcc', 0):.4f}")

print("\n5. POTENTIAL ISSUES:")

# Check if model is predicting all as Normal
if attack_predicted < total_samples * 0.05:
    print("   ⚠️  WARNING: Model is predicting < 5% as Attack!")
    print("      This suggests model collapse or bias toward Normal class")

# Check if model is predicting all as Attack
if normal_predicted < total_samples * 0.05:
    print("   ⚠️  WARNING: Model is predicting < 5% as Normal!")
    print("      This suggests model is over-aggressive")

# Check class imbalance
imbalance_ratio = max(normal_in_test, attack_in_test) / min(normal_in_test, attack_in_test) if min(normal_in_test, attack_in_test) > 0 else float('inf')
if imbalance_ratio > 3:
    print(f"   ⚠️  WARNING: Severe class/sample imbalance ({imbalance_ratio:.1f}:1)")
    print("      High accuracy may be due to predicting the majority class")

# Check if accuracy is suspiciously high
if base.get('accuracy', 0) > 0.95 and base.get('f1_score', 0) < 0.5:
    print("   ⚠️  WARNING: High accuracy but low F1-score!")
    print("      This indicates model is exploiting class imbalance")
    print("      (predicting majority class for everything)")

# Check if high accuracy is legitimate
if base.get('accuracy', 0) > 0.9 and base.get('f1_score', 0) > 0.8:
    print("   ✅ Model performance appears legitimate (high accuracy + high F1)")

print("\n6. ZERO-DAY DETECTION:")
print(f"   Zero-day detection rate: {base.get('zero_day_detection_rate', 0):.4f}")
print("   (This is the % of zero-day samples predicted as Attack)")

print("\n" + "=" * 60)


