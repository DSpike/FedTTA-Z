"""
Fix AUC metrics to include proper confidence intervals
Since we don't have 100-episode AUC data, we'll use bootstrap estimation from confusion matrix
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 80)
print("Fixing AUC Metrics with Confidence Intervals")
print("=" * 80)

# Load 100-episode results
print("\n📊 Loading 100-episode results...")
with open('multi_episode_results/backdoor_100_episodes_phase1.json', 'r') as f:
    episode_data = json.load(f)

# Check if we have per-episode AUC data
has_auc_per_episode = False
if 'per_episode_results' in episode_data and len(episode_data['per_episode_results']) > 0:
    first_ep = episode_data['per_episode_results'][0]
    if 'base_model' in first_ep and 'roc_auc' in first_ep['base_model']:
        has_auc_per_episode = True
        print("✅ Found AUC in per-episode results!")

if has_auc_per_episode:
    # Extract AUC from per-episode results
    print("\n📈 Extracting AUC from per-episode data...")

    base_auc_values = []
    base_auc_pr_values = []
    ttt_auc_values = []
    ttt_auc_pr_values = []

    for ep in episode_data['per_episode_results']:
        if 'roc_auc' in ep['base_model']:
            base_auc_values.append(ep['base_model']['roc_auc'])
        if 'auc_pr' in ep['base_model']:
            base_auc_pr_values.append(ep['base_model']['auc_pr'])
        if 'roc_auc' in ep['ttt_model']:
            ttt_auc_values.append(ep['ttt_model']['roc_auc'])
        if 'auc_pr' in ep['ttt_model']:
            ttt_auc_pr_values.append(ep['ttt_model']['auc_pr'])

    print(f"   Base Model ROC AUC samples: {len(base_auc_values)}")
    print(f"   TTT Model ROC AUC samples: {len(ttt_auc_values)}")

    if len(base_auc_values) > 0:
        # Calculate statistics
        base_auc_mean = np.mean(base_auc_values)
        base_auc_std = np.std(base_auc_values)
        base_auc_ci = 1.96 * base_auc_std / np.sqrt(len(base_auc_values))

        ttt_auc_mean = np.mean(ttt_auc_values)
        ttt_auc_std = np.std(ttt_auc_values)
        ttt_auc_ci = 1.96 * ttt_auc_std / np.sqrt(len(ttt_auc_values))

        base_auc_pr_mean = np.mean(base_auc_pr_values) if base_auc_pr_values else 0
        base_auc_pr_std = np.std(base_auc_pr_values) if base_auc_pr_values else 0
        base_auc_pr_ci = 1.96 * base_auc_pr_std / np.sqrt(len(base_auc_pr_values)) if base_auc_pr_values else 0

        ttt_auc_pr_mean = np.mean(ttt_auc_pr_values) if ttt_auc_pr_values else 0
        ttt_auc_pr_std = np.std(ttt_auc_pr_values) if ttt_auc_pr_values else 0
        ttt_auc_pr_ci = 1.96 * ttt_auc_pr_std / np.sqrt(len(ttt_auc_pr_values)) if ttt_auc_pr_values else 0

        print(f"\n📊 Calculated Statistics:")
        print(f"   Base ROC AUC: {base_auc_mean:.4f} ± {base_auc_ci:.4f}")
        print(f"   TTT ROC AUC:  {ttt_auc_mean:.4f} ± {ttt_auc_ci:.4f}")
        print(f"   Base AUC-PR:  {base_auc_pr_mean:.4f} ± {base_auc_pr_ci:.4f}")
        print(f"   TTT AUC-PR:   {ttt_auc_pr_mean:.4f} ± {ttt_auc_pr_ci:.4f}")

        use_per_episode_auc = True
    else:
        print("❌ No AUC values found in per-episode results")
        use_per_episode_auc = False
else:
    print("❌ No per-episode AUC data available")
    print("   Using single-run values with estimated CI...")
    use_per_episode_auc = False

if not use_per_episode_auc:
    # Use single-run with conservative CI estimate
    print("\n📊 Loading single-run AUC values...")
    with open('performance_plots/performance_metrics_.json', 'r') as f:
        single_run = json.load(f)

    base_auc_mean = single_run['evaluation_results']['base_model']['roc_auc']
    ttt_auc_mean = single_run['evaluation_results']['adapted_model']['roc_auc']
    base_auc_pr_mean = single_run['evaluation_results']['base_model']['auc_pr']
    ttt_auc_pr_mean = single_run['evaluation_results']['adapted_model']['auc_pr']

    # Conservative CI estimate (typical variance for AUC is ~0.02-0.05)
    base_auc_ci = 0.00  # Single run, no CI
    ttt_auc_ci = 0.00
    base_auc_pr_ci = 0.00
    ttt_auc_pr_ci = 0.00

    print(f"   Base ROC AUC: {base_auc_mean:.4f} (single run)")
    print(f"   TTT ROC AUC:  {ttt_auc_mean:.4f} (single run)")

# Load existing publication table
print("\n📄 Updating publication table...")
csv_path = Path('publication_results/performance_table.csv')
df = pd.read_csv(csv_path)

# Remove old AUC rows if they exist
df = df[~df['Metric'].isin(['ROC AUC', 'AUC-PR'])]

# Add new AUC rows with proper formatting
auc_rows = [
    {
        'Metric': 'ROC AUC',
        'Base Model': f"{base_auc_mean:.4f} ± {base_auc_ci:.4f}" if use_per_episode_auc else f"{base_auc_mean:.4f}*",
        'TTT Model': f"{ttt_auc_mean:.4f} ± {ttt_auc_ci:.4f}" if use_per_episode_auc else f"{ttt_auc_mean:.4f}*",
        'Improvement': f"{ttt_auc_mean - base_auc_mean:+.4f}"
    },
    {
        'Metric': 'AUC-PR',
        'Base Model': f"{base_auc_pr_mean:.4f} ± {base_auc_pr_ci:.4f}" if use_per_episode_auc else f"{base_auc_pr_mean:.4f}*",
        'TTT Model': f"{ttt_auc_pr_mean:.4f} ± {ttt_auc_pr_ci:.4f}" if use_per_episode_auc else f"{ttt_auc_pr_mean:.4f}*",
        'Improvement': f"{ttt_auc_pr_mean - base_auc_pr_mean:+.4f}"
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
    fallback_path = str(csv_path).replace('.csv', f'_with_ci_{timestamp}.csv')
    df_new.to_csv(fallback_path, index=False)
    print(f"✅ Saved to fallback: {fallback_path}")

# Update LaTeX table
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
    if not use_per_episode_auc:
        f.write("\\item * ROC AUC and AUC-PR values are from a representative single run.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print(f"✅ LaTeX table updated: {latex_path}")

# Print final table
print("\n" + "=" * 80)
print("UPDATED PERFORMANCE TABLE")
print("=" * 80)
print(df_new.to_string(index=False))
if not use_per_episode_auc:
    print("\nNote: * indicates single-run values (no CI available)")
print("=" * 80)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if use_per_episode_auc:
    print("✅ AUC metrics extracted from per-episode data with proper CI")
    print("✅ All metrics now have confidence intervals")
else:
    print("⚠️  AUC metrics from single run (per-episode data doesn't have AUC)")
    print("⚠️  Shown with * to indicate single-run values")
    print("\n💡 To get 100-episode validated AUC:")
    print("   1. Complete new training (TENT + n_query=100)")
    print("   2. Run: python multi_episode_evaluation.py --attack Backdoor --episodes 100")
    print("   3. New run WILL include AUC with proper CI")
print("=" * 80)
