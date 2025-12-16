import json
import os

# Load the latest performance metrics
with open('performance_plots/performance_metrics_.json', 'r') as f:
    data = json.load(f)

base = data['evaluation_results']['base_model']
ttt = data['evaluation_results']['adapted_model']

print("=" * 80)
print("FULL RUN RESULTS (15 rounds, 228 TTT steps, 18 meta-epochs)")
print("=" * 80)

print("\nBASE MODEL:")
print(f"  Accuracy:     {base['accuracy']:.4f} ({base['accuracy']*100:.2f}%)")
print(f"  F1-Score:     {base['f1_score']:.4f} ({base['f1_score']*100:.2f}%)")
print(f"  Precision:    {base['precision']:.4f} ({base['precision']*100:.2f}%)")
print(f"  Recall:       {base['recall']:.4f} ({base['recall']*100:.2f}%)")
print(f"  ROC-AUC:      {base['roc_auc']:.4f} ({base['roc_auc']*100:.2f}%)")
print(f"  AUC-PR:       {base['auc_pr']:.4f} ({base['auc_pr']*100:.2f}%)")
print(f"  ZDR:          {base.get('zero_day_detection_rate', 0):.4f} ({base.get('zero_day_detection_rate', 0)*100:.2f}%)")

print("\nTTT MODEL:")
print(f"  Accuracy:     {ttt['accuracy']:.4f} ({ttt['accuracy']*100:.2f}%)")
print(f"  F1-Score:     {ttt['f1_score']:.4f} ({ttt['f1_score']*100:.2f}%)")
print(f"  Precision:    {ttt['precision']:.4f} ({ttt['precision']*100:.2f}%)")
print(f"  Recall:       {ttt['recall']:.4f} ({ttt['recall']*100:.2f}%)")
print(f"  ROC-AUC:      {ttt['roc_auc']:.4f} ({ttt['roc_auc']*100:.2f}%)")
print(f"  AUC-PR:       {ttt['auc_pr']:.4f} ({ttt['auc_pr']*100:.2f}%)")
print(f"  ZDR:          {ttt.get('zero_day_detection_rate', 0):.4f} ({ttt.get('zero_day_detection_rate', 0)*100:.2f}%)")

print("\nIMPROVEMENTS (TTT vs Base):")
acc_imp = (ttt['accuracy'] - base['accuracy']) * 100
f1_imp = (ttt['f1_score'] - base['f1_score']) * 100
zdr_imp = (ttt.get('zero_day_detection_rate', 0) - base.get('zero_day_detection_rate', 0)) * 100
recall_imp = (ttt['recall'] - base['recall']) * 100
precision_imp = (ttt['precision'] - base['precision']) * 100
auc_pr_imp = (ttt['auc_pr'] - base['auc_pr']) * 100

print(f"  Accuracy:     +{acc_imp:.2f}pp ({acc_imp/base['accuracy']*100 if base['accuracy'] > 0 else 0:.1f}% relative)")
print(f"  F1-Score:     +{f1_imp:.2f}pp ({f1_imp/base['f1_score']*100 if base['f1_score'] > 0 else 0:.1f}% relative)")
print(f"  ZDR:          +{zdr_imp:.2f}pp ({zdr_imp/base.get('zero_day_detection_rate', 0.001)*100 if base.get('zero_day_detection_rate', 0) > 0 else 0:.1f}% relative)")
print(f"  Recall:       +{recall_imp:.2f}pp ({recall_imp/base['recall']*100 if base['recall'] > 0 else 0:.1f}% relative)")
print(f"  Precision:    +{precision_imp:.2f}pp ({precision_imp/base['precision']*100 if base['precision'] > 0 else 0:.1f}% relative)")
print(f"  AUC-PR:       +{auc_pr_imp:.2f}pp ({auc_pr_imp/base['auc_pr']*100 if base['auc_pr'] > 0 else 0:.1f}% relative)")

# Check if there's zero-day specific metrics
if 'zero_day_metrics' in ttt:
    zd = ttt['zero_day_metrics']
    print("\nTTT MODEL - Zero-Day Specific:")
    print(f"  Accuracy:     {zd.get('accuracy', 0):.4f} ({zd.get('accuracy', 0)*100:.2f}%)")
    print(f"  F1-Score:     {zd.get('f1_score', 0):.4f} ({zd.get('f1_score', 0)*100:.2f}%)")
    print(f"  Precision:    {zd.get('precision', 0):.4f} ({zd.get('precision', 0)*100:.2f}%)")
    print(f"  Recall:       {zd.get('recall', 0):.4f} ({zd.get('recall', 0)*100:.2f}%)")
    print(f"  AUC-PR:       {zd.get('auc_pr', 0):.4f} ({zd.get('auc_pr', 0)*100:.2f}%)")

print("\n" + "=" * 80)









