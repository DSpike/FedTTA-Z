"""
Add AUC metrics to publication table using single-run values
"""
import json
import pandas as pd
from pathlib import Path

print("=" * 80)
print("Adding AUC Metrics to Publication Table")
print("=" * 80)

# Load single-run results (has AUC)
print("\n📊 Loading single-run AUC values...")
with open('performance_plots/performance_metrics_.json', 'r') as f:
    single_run = json.load(f)

base_auc = single_run['evaluation_results']['base_model']['roc_auc']
base_auc_pr = single_run['evaluation_results']['base_model']['auc_pr']
ttt_auc = single_run['evaluation_results']['adapted_model']['roc_auc']
ttt_auc_pr = single_run['evaluation_results']['adapted_model']['auc_pr']

print(f"   Base Model ROC AUC: {base_auc:.4f}")
print(f"   Base Model AUC-PR:  {base_auc_pr:.4f}")
print(f"   TTT Model ROC AUC:  {ttt_auc:.4f}")
print(f"   TTT Model AUC-PR:   {ttt_auc_pr:.4f}")

# Load existing publication table
print("\n📄 Loading existing publication table...")
csv_path = Path('publication_results/performance_table.csv')
if not csv_path.exists():
    print("❌ Publication table not found. Run create_publication_results.py first!")
    exit(1)

df = pd.read_csv(csv_path)
print(f"   Loaded {len(df)} metrics")

# Add AUC rows
print("\n➕ Adding AUC metrics...")

auc_rows = [
    {
        'Metric': 'ROC AUC',
        'Base Model': f"{base_auc:.4f}*",
        'TTT Model': f"{ttt_auc:.4f}*",
        'Improvement': f"{ttt_auc - base_auc:+.4f}"
    },
    {
        'Metric': 'AUC-PR',
        'Base Model': f"{base_auc_pr:.4f}*",
        'TTT Model': f"{ttt_auc_pr:.4f}*",
        'Improvement': f"{ttt_auc_pr - base_auc_pr:+.4f}"
    }
]

# Append to dataframe
df_new = pd.concat([df, pd.DataFrame(auc_rows)], ignore_index=True)

# Save updated table
print("\n💾 Saving updated table...")
try:
    df_new.to_csv(csv_path, index=False)
    print(f"✅ Updated table saved to: {csv_path}")
except PermissionError:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fallback_path = str(csv_path).replace('.csv', f'_with_auc_{timestamp}.csv')
    df_new.to_csv(fallback_path, index=False)
    print(f"✅ Saved to fallback: {fallback_path}")

# Also update LaTeX table
print("\n📝 Updating LaTeX table...")
latex_path = str(csv_path).replace('.csv', '.tex')
with open(latex_path, 'w') as f:
    f.write("% LaTeX table for publication\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Performance Comparison: Base Model vs. TTT-Enhanced Model (100 Episodes)}\n")
    f.write("\\label{tab:performance}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\hline\n")
    f.write("Metric & Base Model & TTT Model & Improvement \\\\\n")
    f.write("\\hline\n")

    for _, row in df_new.iterrows():
        metric = row['Metric'].replace('%', '\\%')
        f.write(f"{metric} & {row['Base Model']} & {row['TTT Model']} & {row['Improvement']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\small\n")
    f.write("\\item Results averaged over 100 independent episodes (mean $\\pm$ 95\\% CI).\n")
    f.write("\\item * ROC AUC and AUC-PR values are from a representative single run.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print(f"✅ LaTeX table updated: {latex_path}")

# Print final table
print("\n" + "=" * 80)
print("UPDATED PERFORMANCE TABLE")
print("=" * 80)
print(df_new.to_string(index=False))
print("\nNote: * indicates values from single run (not 100-episode average)")
print("=" * 80)

print("\n✅ DONE! Your table now includes AUC metrics.")
print("\nFor your paper, add this note:")
print("   'ROC AUC and AUC-PR values reported from representative single")
print("    evaluation runs; all other metrics validated over 100 episodes.'")
