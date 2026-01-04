#!/usr/bin/env python3
"""
Analyze Base Model Performance on Known vs Zero-Day Attacks
"""

import json
import numpy as np
from pathlib import Path

results_file = Path('multi_episode_results/backdoor_100_episodes_phase2.json')

if not results_file.exists():
    print(f"Error: {results_file} not found")
    exit(1)

with open(results_file, 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("BASE MODEL PERFORMANCE ANALYSIS - KNOWN vs ZERO-DAY")
print("="*80)

# Aggregate metrics
print("\n📊 OVERALL BASE MODEL PERFORMANCE:")
base = data['base_model']
print(f"  Overall Accuracy:        {base['accuracy']['mean']*100:.2f}%")
print(f"  F1-Score:                {base['f1_score']['mean']*100:.2f}%")
print(f"  Zero-Day Detection Rate: {base['zero_day_detection_rate']['mean']*100:.2f}%")
print(f"  False Alarm Rate:        {base['false_alarm_rate']['mean']*100:.2f}%")

# Analyze per-episode confusion matrices
print("\n" + "-"*80)
print("ANALYZING CONFUSION MATRICES ACROSS ALL EPISODES")
print("-"*80)

episodes = data['per_episode_results']

# Collect confusion matrices
all_cms = []
for ep in episodes:
    if 'base_model' in ep and 'confusion_matrix' in ep['base_model']:
        cm = np.array(ep['base_model']['confusion_matrix'])
        all_cms.append(cm)

if len(all_cms) == 0:
    print("No confusion matrices found!")
    exit(1)

# Average confusion matrix
avg_cm = np.mean(all_cms, axis=0)

print(f"\n📊 AVERAGE CONFUSION MATRIX (over {len(all_cms)} episodes):")
print(f"                  Predicted Normal | Predicted Attack")
print(f"  Actual Normal:        {avg_cm[0,0]:.1f}     |     {avg_cm[0,1]:.1f}")
print(f"  Actual Attack:        {avg_cm[1,0]:.1f}     |     {avg_cm[1,1]:.1f}")

# Calculate metrics from confusion matrix
TN, FP = avg_cm[0,0], avg_cm[0,1]
FN, TP = avg_cm[1,0], avg_cm[1,1]

# Overall metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
far = FP / (TN + FP) if (TN + FP) > 0 else 0

print(f"\n📊 DERIVED METRICS:")
print(f"  True Positives (TP):   {TP:.1f}  (Attacks correctly detected)")
print(f"  False Positives (FP):  {FP:.1f}  (Normal traffic flagged as attack)")
print(f"  True Negatives (TN):   {TN:.1f}  (Normal traffic correctly identified)")
print(f"  False Negatives (FN):  {FN:.1f}  (Attacks missed)")
print(f"\n  Attack Detection Rate: {recall*100:.2f}%  (TP / (TP + FN))")
print(f"  Normal Detection Rate: {TN/(TN+FP)*100:.2f}%  (TN / (TN + FP))")
print(f"  False Alarm Rate:      {far*100:.2f}%")

# Analyze zero-day vs known attack performance
print("\n" + "="*80)
print("ZERO-DAY vs KNOWN ATTACK BREAKDOWN")
print("="*80)

# Collect per-episode data
zdr_values = []
known_recall_values = []

for ep in episodes:
    if 'base_model' not in ep:
        continue

    ep_base = ep['base_model']

    # Zero-day detection rate
    zdr = ep_base.get('zero_day_detection_rate', 0)
    zdr_values.append(zdr)

    # Try to infer known attack performance from confusion matrix
    cm = np.array(ep_base.get('confusion_matrix', [[0,0],[0,0]]))

    # Total attacks = TP + FN
    total_attacks = cm[1,0] + cm[1,1]

    # Zero-day samples (25% of test set typically)
    zero_day_samples = ep_base.get('zero_day_samples', int(total_attacks * 0.25))

    # Known attack samples
    known_attack_samples = total_attacks - zero_day_samples

    # Zero-day correctly detected
    zero_day_detected = int(zdr * zero_day_samples)

    # Known attacks correctly detected (from TP - zero_day_detected)
    known_detected = cm[1,1] - zero_day_detected

    # Known attack recall
    known_recall = known_detected / known_attack_samples if known_attack_samples > 0 else 0
    known_recall_values.append(known_recall)

print(f"\n📊 PERFORMANCE BREAKDOWN (estimated):")
print(f"\n  Zero-Day (Backdoor) Detection:")
print(f"    Mean Detection Rate: {np.mean(zdr_values)*100:.2f}%")
print(f"    Std Dev:             {np.std(zdr_values)*100:.2f}%")
print(f"    Min:                 {np.min(zdr_values)*100:.2f}%")
print(f"    Max:                 {np.max(zdr_values)*100:.2f}%")

print(f"\n  Known Attack Detection (estimated):")
print(f"    Mean Detection Rate: {np.mean(known_recall_values)*100:.2f}%")
print(f"    Std Dev:             {np.std(known_recall_values)*100:.2f}%")
print(f"    Min:                 {np.min(known_recall_values)*100:.2f}%")
print(f"    Max:                 {np.max(known_recall_values)*100:.2f}%")

# Comparison
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

zdr_mean = np.mean(zdr_values)
known_mean = np.mean(known_recall_values)

print(f"\n1. Zero-Day Detection:  {zdr_mean*100:.2f}%")
print(f"2. Known Attack Detection: {known_mean*100:.2f}%")
print(f"3. Performance Gap:     {(zdr_mean - known_mean)*100:+.2f}%")

if zdr_mean > known_mean:
    print(f"\n✅ SURPRISING: Zero-day detection ({zdr_mean*100:.1f}%) is BETTER than known attack detection ({known_mean*100:.1f}%)")
    print("   This suggests the model generalizes well to unseen attacks!")
elif known_mean > zdr_mean + 0.05:
    print(f"\n⚠️  EXPECTED: Known attack detection ({known_mean*100:.1f}%) is better than zero-day ({zdr_mean*100:.1f}%)")
    print("   Model performs better on seen attack types.")
else:
    print(f"\n✅ BALANCED: Similar performance on known ({known_mean*100:.1f}%) and zero-day ({zdr_mean*100:.1f}%)")

# False Alarm Rate
print(f"\n4. False Alarm Rate:    {far*100:.2f}%")
if far > 0.2:
    print("   ⚠️  HIGH: More than 20% of normal traffic flagged as attacks")
elif far > 0.1:
    print("   ⚠️  MODERATE: 10-20% of normal traffic flagged as attacks")
else:
    print("   ✅ ACCEPTABLE: Less than 10% false alarms")

# Normal traffic detection
normal_detection = TN / (TN + FP)
print(f"\n5. Normal Traffic Detection: {normal_detection*100:.2f}%")
if normal_detection < 0.7:
    print("   ❌ POOR: Model struggles to identify normal traffic")
elif normal_detection < 0.85:
    print("   ⚠️  MODERATE: Room for improvement on normal traffic")
else:
    print("   ✅ GOOD: Model effectively identifies normal traffic")

print("\n" + "="*80)

# Overall assessment
print("\nOVERALL ASSESSMENT:")
print("-"*80)

if known_mean < 0.7:
    print("❌ POOR Known Attack Detection (<70%)")
    print("   The base model struggles with attacks it was trained on!")
    print("   Possible causes:")
    print("   - Imbalanced support set (dominated by Generic/Exploits)")
    print("   - Poor embedding quality")
    print("   - Insufficient training")
    print("   - Wrong distance metric")
elif known_mean < 0.8:
    print("⚠️  MODERATE Known Attack Detection (70-80%)")
    print("   Room for improvement on known attacks")
else:
    print("✅ GOOD Known Attack Detection (>80%)")

print("="*80 + "\n")
