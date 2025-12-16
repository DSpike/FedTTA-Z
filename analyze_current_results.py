#!/usr/bin/env python3
"""Extract and analyze current run results"""

import json
from datetime import datetime

# Load results
with open('performance_plots/performance_metrics_.json', 'r') as f:
    data = json.load(f)

base = data['evaluation_results']['base_model']
ttt = data['evaluation_results']['adapted_model']

# Load embedding quality if available
try:
    with open('embedding_quality_diagnostics/embedding_quality_results.json', 'r') as f:
        embedding_data = json.load(f)
    prototype_sep = embedding_data['prototype_separation']['avg_inter_class_distance']
    embedding_sep = embedding_data['embedding_separability']['silhouette_score']
    prototype_acc = embedding_data['prototype_accuracy']['overall_accuracy']
except:
    prototype_sep = None
    embedding_sep = None
    prototype_acc = None

print("=" * 80)
print("CURRENT RUN RESULTS ANALYSIS (With Conservative Improvements)")
print("=" * 80)

print("\n📊 BASE MODEL Performance:")
print(f"  Accuracy:     {base['accuracy']:.4f} ({base['accuracy']*100:.2f}%)")
print(f"  F1-Score:     {base['f1_score']:.4f} ({base['f1_score']*100:.2f}%)")
print(f"  Precision:    {base['precision']:.4f} ({base['precision']*100:.2f}%)")
print(f"  Recall:       {base['recall']:.4f} ({base['recall']*100:.2f}%)")
print(f"  ROC-AUC:      {base.get('roc_auc', 0):.4f} ({base.get('roc_auc', 0)*100:.2f}%)")
print(f"  AUC-PR:       {base.get('auc_pr', 0):.4f} ({base.get('auc_pr', 0)*100:.2f}%)")
print(f"  ZDR:          {base.get('zero_day_detection_rate', 0):.4f} ({base.get('zero_day_detection_rate', 0)*100:.2f}%)")

print("\n📊 TTT MODEL Performance:")
print(f"  Accuracy:     {ttt['accuracy']:.4f} ({ttt['accuracy']*100:.2f}%)")
print(f"  F1-Score:     {ttt['f1_score']:.4f} ({ttt['f1_score']*100:.2f}%)")
print(f"  Precision:    {ttt['precision']:.4f} ({ttt['precision']*100:.2f}%)")
print(f"  Recall:       {ttt['recall']:.4f} ({ttt['recall']*100:.2f}%)")
print(f"  ROC-AUC:      {ttt.get('roc_auc', 0):.4f} ({ttt.get('roc_auc', 0)*100:.2f}%)")
print(f"  AUC-PR:       {ttt.get('auc_pr', 0):.4f} ({ttt.get('auc_pr', 0)*100:.2f}%)")
print(f"  ZDR:          {ttt.get('zero_day_detection_rate', 0):.4f} ({ttt.get('zero_day_detection_rate', 0)*100:.2f}%)")

print("\n📈 IMPROVEMENTS (TTT vs Base):")
acc_imp = (ttt['accuracy'] - base['accuracy']) * 100
f1_imp = (ttt['f1_score'] - base['f1_score']) * 100
zdr_imp = (ttt.get('zero_day_detection_rate', 0) - base.get('zero_day_detection_rate', 0)) * 100
recall_imp = (ttt['recall'] - base['recall']) * 100
precision_imp = (ttt['precision'] - base['precision']) * 100
auc_pr_imp = (ttt.get('auc_pr', 0) - base.get('auc_pr', 0)) * 100

print(f"  Accuracy:     +{acc_imp:.2f}pp")
print(f"  F1-Score:     +{f1_imp:.2f}pp")
print(f"  ZDR:          +{zdr_imp:.2f}pp")
print(f"  Recall:       +{recall_imp:.2f}pp")
print(f"  Precision:    +{precision_imp:.2f}pp")
print(f"  AUC-PR:       +{auc_pr_imp:.2f}pp")

if prototype_sep is not None:
    print("\n📊 EMBEDDING QUALITY:")
    print(f"  Prototype Separation: {prototype_sep:.4f}")
    print(f"  Embedding Separability: {embedding_sep:.4f} (target: >0.3)")
    print(f"  Prototype Accuracy: {prototype_acc:.4f} ({prototype_acc*100:.2f}%)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)









