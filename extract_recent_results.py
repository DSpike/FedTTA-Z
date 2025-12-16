import json

with open('performance_plots/performance_metrics_.json', 'r') as f:
    data = json.load(f)

base = data['evaluation_results']['base_model']
ttt = data['evaluation_results']['adapted_model']

print('='*80)
print('RECENT RESULTS ANALYSIS')
print('='*80)
print(f'\nBASE MODEL:')
print(f'  Accuracy: {base["accuracy"]:.4f} ({base["accuracy"]*100:.2f}%)')
print(f'  F1-Score: {base.get("f1_score", 0):.4f} ({base.get("f1_score", 0)*100:.2f}%)')
print(f'  Precision: {base.get("precision", 0):.4f}')
print(f'  Recall: {base.get("recall", 0):.4f}')
print(f'  ROC-AUC: {base.get("roc_auc", 0):.4f}')
print(f'  AUC-PR: {base.get("auc_pr", 0):.4f}')
print(f'  ZDR: {base.get("zero_day_detection_rate", 0):.4f} ({base.get("zero_day_detection_rate", 0)*100:.2f}%)')

print(f'\nTTT MODEL:')
print(f'  Accuracy: {ttt["accuracy"]:.4f} ({ttt["accuracy"]*100:.2f}%)')
print(f'  F1-Score: {ttt.get("f1_score", 0):.4f} ({ttt.get("f1_score", 0)*100:.2f}%)')
print(f'  Precision: {ttt.get("precision", 0):.4f}')
print(f'  Recall: {ttt.get("recall", 0):.4f}')
print(f'  ROC-AUC: {ttt.get("roc_auc", 0):.4f}')
print(f'  AUC-PR: {ttt.get("auc_pr", 0):.4f}')
print(f'  ZDR: {ttt.get("zero_day_detection_rate", 0):.4f} ({ttt.get("zero_day_detection_rate", 0)*100:.2f}%)')

print(f'\nIMPROVEMENTS (TTT vs Base):')
acc_imp = (ttt["accuracy"] - base["accuracy"]) * 100
f1_imp = (ttt.get("f1_score", 0) - base.get("f1_score", 0)) * 100
zdr_imp = (ttt.get("zero_day_detection_rate", 0) - base.get("zero_day_detection_rate", 0)) * 100
recall_imp = (ttt.get("recall", 0) - base.get("recall", 0)) * 100
precision_imp = (ttt.get("precision", 0) - base.get("precision", 0)) * 100
auc_pr_imp = (ttt.get("auc_pr", 0) - base.get("auc_pr", 0)) * 100

print(f'  Accuracy: +{acc_imp:.2f}pp ({acc_imp/abs(base["accuracy"]*100)*100:.1f}% relative)')
print(f'  F1-Score: +{f1_imp:.2f}pp ({f1_imp/abs(base.get("f1_score", 0.001)*100)*100:.1f}% relative)')
print(f'  ZDR: +{zdr_imp:.2f}pp ({zdr_imp/abs(base.get("zero_day_detection_rate", 0.001)*100)*100:.1f}% relative)')
print(f'  Recall: +{recall_imp:.2f}pp')
print(f'  Precision: {precision_imp:+.2f}pp')
print(f'  AUC-PR: +{auc_pr_imp:.2f}pp')

print(f'\nCONFIGURATION:')
print(f'  Quick test mode: 3 clients, 3 rounds, 5 meta_epochs')
print(f'  Zero-day attack: DoS')
print(f'  Dataset: UNSW-NB15')









