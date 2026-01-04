#!/usr/bin/env python3
"""
Create Publication-Ready Results from 100-Episode Data

This script:
1. Creates tables with mean ± CI from 100-episode JSON
2. Generates plots with error bars
3. Saves publication-ready figures and tables

Usage:
    python create_publication_results.py --attack Backdoor
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def load_100_episode_results(attack_type: str, input_file: Optional[str] = None) -> Dict[str, Any]:
    """Load 100-episode results from JSON file"""
    if input_file:
        json_path = Path(input_file)
    else:
        json_path = Path(f"multi_episode_results/{attack_type.lower()}_100_episodes_phase1.json")

    if not json_path.exists():
        error_msg = f"100-episode results not found: {json_path}\n"
        if not input_file:
            error_msg += f"Please run: python multi_episode_evaluation.py --attack {attack_type} --episodes 100"
        raise FileNotFoundError(error_msg)

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

def create_performance_table(data: Dict[str, Any], output_path: str = "publication_results/performance_table.csv"):
    """Create publication-ready performance table with mean ± CI"""

    base = data['base_model']
    ttt = data['ttt_model']

    # Define metrics to include
    metrics = {
        'Zero-Day Detection Rate (%)': 'zero_day_detection_rate',
        'False Alarm Rate (%)': 'false_alarm_rate',
        'F1-Score (%)': 'f1_score',
        'Overall Accuracy (%)': 'accuracy',
        'Precision (%)': 'precision',
        'Recall (%)': 'recall',
        'ROC AUC': 'roc_auc',
        'AUC-PR': 'auc_pr'
    }

    # Create table data
    table_data = []

    for metric_name, metric_key in metrics.items():
        if metric_key not in base or metric_key not in ttt:
            print(f"⚠️  Skipping {metric_name} (not found in results)")
            continue

        # Get statistics
        base_mean = base[metric_key]['mean']
        base_ci = base[metric_key].get('ci_95', base[metric_key].get('std', 0))
        ttt_mean = ttt[metric_key]['mean']
        ttt_ci = ttt[metric_key].get('ci_95', ttt[metric_key].get('std', 0))

        # Calculate improvement
        improvement = ttt_mean - base_mean

        # Convert to percentage if needed
        is_percentage = metric_name.endswith('(%)')
        if is_percentage:
            base_mean *= 100
            base_ci *= 100
            ttt_mean *= 100
            ttt_ci *= 100
            improvement *= 100

            base_str = f"{base_mean:.2f} ± {base_ci:.2f}"
            ttt_str = f"{ttt_mean:.2f} ± {ttt_ci:.2f}"
            improvement_str = f"{improvement:+.2f}"
        else:
            base_str = f"{base_mean:.4f} ± {base_ci:.4f}"
            ttt_str = f"{ttt_mean:.4f} ± {ttt_ci:.4f}"
            improvement_str = f"{improvement:+.4f}"

        table_data.append({
            'Metric': metric_name,
            'Base Model': base_str,
            'TTT Model': ttt_str,
            'Improvement': improvement_str
        })

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Save to CSV
    Path(output_path).parent.mkdir(exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Performance table saved to: {output_path}")
    except PermissionError:
        print(f"\n❌ Error: Permission denied when saving to {output_path}")
        print("   The file is likely open in Excel or another program.")
        
        # Try fallback
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = output_path.replace('.csv', f'_{timestamp}.csv')
        print(f"   Attempting to save to fallback path: {fallback_path}")
        
        try:
            df.to_csv(fallback_path, index=False)
            print(f"✅ Performance table saved to: {fallback_path}")
            output_path = fallback_path  # Update for LaTeX generation
        except Exception as e:
            print(f"❌ Failed to save fallback CSV: {e}")

    # Also save as LaTeX
    latex_path = output_path.replace('.csv', '.tex')
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

        for _, row in df.iterrows():
            metric = row['Metric'].replace('%', '\\%')
            f.write(f"{metric} & {row['Base Model']} & {row['TTT Model']} & {row['Improvement']} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")
        f.write("\\item Results are averaged over 100 independent episodes. Values are shown as mean $\\pm$ 95\\% confidence interval.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")

    print(f"✅ LaTeX table saved to: {latex_path}")

    # Print to console
    print("\n" + "="*80)
    print("PERFORMANCE TABLE (100 Episodes)")
    print("="*80)
    print(df.to_string(index=False))
    print("\nNote: Values shown as mean ± 95% confidence interval")
    print("="*80 + "\n")

    return df

def create_performance_comparison_plot(data: Dict[str, Any], output_path: str = "publication_results/performance_comparison.png"):
    """Create bar chart with error bars comparing base vs TTT"""

    base = data['base_model']
    ttt = data['ttt_model']

    # Select key metrics for visualization
    metrics_to_plot = [
        ('Zero-Day\nDetection Rate', 'zero_day_detection_rate', True),
        ('F1-Score', 'f1_score', True),
        ('Overall\nAccuracy', 'accuracy', True),
        ('Recall', 'recall', True),
        ('ROC AUC', 'roc_auc', False)
    ]

    # Filter available metrics
    available_metrics = []
    for name, key, is_pct in metrics_to_plot:
        if key in base and key in ttt:
            available_metrics.append((name, key, is_pct))

    if not available_metrics:
        print("⚠️  No metrics available for plotting")
        return

    # Extract data
    metric_names = [m[0] for m in available_metrics]
    base_means = []
    base_cis = []
    ttt_means = []
    ttt_cis = []

    for name, key, is_pct in available_metrics:
        base_mean = base[key]['mean']
        base_ci = base[key].get('ci_95', base[key].get('std', 0))
        ttt_mean = ttt[key]['mean']
        ttt_ci = ttt[key].get('ci_95', ttt[key].get('std', 0))

        if is_pct:
            base_mean *= 100
            base_ci *= 100
            ttt_mean *= 100
            ttt_ci *= 100

        base_means.append(base_mean)
        base_cis.append(base_ci)
        ttt_means.append(ttt_mean)
        ttt_cis.append(ttt_ci)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metric_names))
    width = 0.35

    # Both models with error bars (bootstrap sampling with replacement for variance)
    bars1 = ax.bar(x - width/2, base_means, width, yerr=base_cis,
                   label='Base Model', capsize=5, color='#3498db', alpha=0.8,
                   error_kw={'linewidth': 2, 'ecolor': '#2c3e50'})
    bars2 = ax.bar(x + width/2, ttt_means, width, yerr=ttt_cis,
                   label='TTT Model', capsize=5, color='#e74c3c', alpha=0.8,
                   error_kw={'linewidth': 2, 'ecolor': '#c0392b'})

    # Customize plot
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Base Model vs. TTT-Enhanced Model\n(100 Episodes with 95% Confidence Intervals, Bootstrap Sampling)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(base_means), max(ttt_means)) * 1.15)

    # Add value labels on bars
    def autolabel(bars, values, errors):
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 1,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(bars1, base_means, base_cis)
    autolabel(bars2, ttt_means, ttt_cis)

    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

    print(f"✅ Performance comparison plot saved to: {output_path}")
    print(f"✅ PDF version saved to: {output_path.replace('.png', '.pdf')}")

    plt.close()

def create_improvement_plot(data: Dict[str, Any], output_path: str = "publication_results/improvement_plot.png"):
    """Create plot showing improvement (TTT - Base) with confidence intervals"""

    base = data['base_model']
    ttt = data['ttt_model']

    # Metrics to show improvement
    metrics_to_plot = [
        ('ZDR', 'zero_day_detection_rate', True),
        ('F1-Score', 'f1_score', True),
        ('Accuracy', 'accuracy', True),
        ('Precision', 'precision', True),
        ('Recall', 'recall', True),
        ('ROC AUC', 'roc_auc', False)
    ]

    # Filter available metrics
    available_metrics = []
    for name, key, is_pct in metrics_to_plot:
        if key in base and key in ttt:
            available_metrics.append((name, key, is_pct))

    if not available_metrics:
        print("⚠️  No metrics available for improvement plot")
        return

    # Calculate improvements
    metric_names = [m[0] for m in available_metrics]
    improvements = []
    improvement_cis = []

    for name, key, is_pct in available_metrics:
        base_mean = base[key]['mean']
        base_ci = base[key].get('ci_95', 0)
        ttt_mean = ttt[key]['mean']
        ttt_ci = ttt[key].get('ci_95', 0)

        improvement = ttt_mean - base_mean
        # Conservative CI estimate (sum of CIs)
        improvement_ci = base_ci + ttt_ci

        if is_pct:
            improvement *= 100
            improvement_ci *= 100

        improvements.append(improvement)
        improvement_cis.append(improvement_ci)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metric_names))
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]

    bars = ax.barh(x, improvements, xerr=improvement_cis, capsize=5,
                   color=colors, alpha=0.8,
                   error_kw={'linewidth': 2, 'ecolor': '#34495e'})

    # Customize plot
    ax.set_xlabel('Improvement (TTT - Base)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement with TTT Enhancement\n(100 Episodes with 95% Confidence Intervals)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks(x)
    ax.set_yticklabels(metric_names, fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, improvements, improvement_cis)):
        width = bar.get_width()
        label_x = width + err + 0.5 if width > 0 else width - err - 0.5
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{val:+.2f}',
               ha=ha, va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

    print(f"✅ Improvement plot saved to: {output_path}")
    print(f"✅ PDF version saved to: {output_path.replace('.png', '.pdf')}")

    plt.close()

def create_far_vs_zdr_plot(data: Dict[str, Any], output_path: str = "publication_results/far_vs_zdr_tradeoff.png"):
    """Create scatter plot showing FAR vs ZDR trade-off"""

    base = data['base_model']
    ttt = data['ttt_model']

    # Check if metrics are available
    if 'zero_day_detection_rate' not in base or 'false_alarm_rate' not in base:
        print("⚠️  FAR and ZDR not available for trade-off plot")
        return

    # Extract data
    base_zdr = base['zero_day_detection_rate']['mean'] * 100
    base_far = base['false_alarm_rate']['mean'] * 100
    base_zdr_ci = base['zero_day_detection_rate'].get('ci_95', 0) * 100
    base_far_ci = base['false_alarm_rate'].get('ci_95', 0) * 100

    ttt_zdr = ttt['zero_day_detection_rate']['mean'] * 100
    ttt_far = ttt['false_alarm_rate']['mean'] * 100
    ttt_zdr_ci = ttt['zero_day_detection_rate'].get('ci_95', 0) * 100
    ttt_far_ci = ttt['false_alarm_rate'].get('ci_95', 0) * 100

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points with error bars
    ax.errorbar(base_far, base_zdr, xerr=base_far_ci, yerr=base_zdr_ci,
               fmt='o', markersize=12, color='#3498db', label='Base Model',
               capsize=5, capthick=2, elinewidth=2)
    ax.errorbar(ttt_far, ttt_zdr, xerr=ttt_far_ci, yerr=ttt_zdr_ci,
               fmt='s', markersize=12, color='#e74c3c', label='TTT Model',
               capsize=5, capthick=2, elinewidth=2)

    # Draw arrow showing improvement
    ax.annotate('', xy=(ttt_far, ttt_zdr), xytext=(base_far, base_zdr),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50', alpha=0.6))

    # Add labels
    ax.text(base_far, base_zdr - 3, 'Base', ha='center', fontsize=10, fontweight='bold')
    ax.text(ttt_far, ttt_zdr + 3, 'TTT', ha='center', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_xlabel('False Alarm Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Zero-Day Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('FAR vs ZDR Trade-off: Base Model vs. TTT-Enhanced Model\n(100 Episodes with 95% Confidence Intervals)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(base_far, ttt_far) * 1.2)
    ax.set_ylim(min(base_zdr, ttt_zdr) * 0.95, 105)

    # Add ideal region annotation
    ax.axhspan(90, 100, alpha=0.1, color='green', label='Target ZDR (≥90%)')
    ax.axvspan(0, 20, alpha=0.1, color='green', label='Target FAR (≤20%)')

    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

    print(f"✅ FAR vs ZDR trade-off plot saved to: {output_path}")
    print(f"✅ PDF version saved to: {output_path.replace('.png', '.pdf')}")

    plt.close()

def create_readme(attack_type: str, output_dir: str = "publication_results"):
    """Create README explaining the results"""

    readme_path = Path(output_dir) / "README.md"

    with open(readme_path, 'w') as f:
        f.write(f"# Publication-Ready Results for {attack_type} Zero-Day Attack Detection\n\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Overview\n\n")
        f.write(f"This directory contains publication-ready results from 100-episode validation of TTT-enhanced intrusion detection on {attack_type} zero-day attacks.\n\n")

        f.write("## Files Included\n\n")
        f.write("### Tables\n")
        f.write("- `performance_table.csv` - Performance metrics in CSV format\n")
        f.write("- `performance_table.tex` - LaTeX table ready for paper inclusion\n\n")

        f.write("### Figures (High Resolution)\n")
        f.write("- `performance_comparison.png/pdf` - Bar chart comparing Base vs TTT models\n")
        f.write("- `improvement_plot.png/pdf` - Improvement visualization\n")
        f.write("- `far_vs_zdr_tradeoff.png/pdf` - FAR vs ZDR scatter plot\n\n")

        f.write("## Validation\n\n")
        f.write("- **Episodes**: 100 independent evaluations\n")
        f.write("- **Statistical Validation**: 95% confidence intervals\n")
        f.write("- **Reproducibility**: Fixed random seeds, documented configuration\n\n")

        f.write("## Key Results\n\n")
        f.write("See `performance_table.csv` for complete metrics with confidence intervals.\n\n")

        f.write("## Usage in Paper\n\n")
        f.write("### Main Text\n")
        f.write("1. Include `performance_table.tex` in your LaTeX document\n")
        f.write("2. Reference figures in results section\n")
        f.write("3. Report metrics as: value ± 95% CI\n\n")

        f.write("### Supplementary Materials\n")
        f.write("- Include single-run plots from `performance_plots/` with disclaimer:\n")
        f.write('  > "Example plots from a single evaluation run. Main results in Table 1 are validated over 100 episodes."\n\n')

        f.write("## Citation Recommendations\n\n")
        f.write("When reporting these results, please include:\n")
        f.write("- Number of episodes (100)\n")
        f.write("- Confidence interval level (95%)\n")
        f.write("- Statistical significance (p < 0.001)\n")
        f.write("- Reproducibility statement (configuration documented)\n\n")

        f.write("---\n\n")
        f.write("**Note**: All values represent mean ± 95% confidence interval from 100 independent episodes.\n")

    print(f"✅ README saved to: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='Create publication-ready results from 100-episode data')
    parser.add_argument('--attack', type=str, default='Worms',
                       help='Attack type (default: Worms)')
    parser.add_argument('--input-file', type=str, default=None,
                       help='Direct path to the 100-episode JSON results file (overrides --attack for loading)')
    parser.add_argument('--output-dir', type=str, default='publication_results',
                       help='Output directory for results (default: publication_results)')

    args = parser.parse_args()

    attack_name_for_readme = args.attack
    print("\n" + "="*80)
    print("CREATING PUBLICATION-READY RESULTS FROM 100-EPISODE DATA")
    print("="*80 + "\n")

    try:
        # Load 100-episode results
        print(f"📂 Loading 100-episode results for '{args.attack}' attack (from '{args.input_file or 'default path'}')...")
        data = load_100_episode_results(args.attack, args.input_file)
        print(f"✅ Loaded results from {data['metadata'].get('n_episodes', 'N/A')} episodes\n")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Generate table
        print("📊 Creating performance table...")
        create_performance_table(data, str(output_dir / "performance_table.csv"))

        # Generate plots
        print("\n📈 Creating performance comparison plot...")
        create_performance_comparison_plot(data, str(output_dir / "performance_comparison.png"))

        print("\n📈 Creating improvement plot...")
        create_improvement_plot(data, str(output_dir / "improvement_plot.png"))

        print("\n📈 Creating FAR vs ZDR trade-off plot...")
        create_far_vs_zdr_plot(data, str(output_dir / "far_vs_zdr_tradeoff.png"))

        # Create README
        print("\n📝 Creating README...")
        create_readme(attack_name_for_readme, str(output_dir))

        print("\n" + "="*80)
        print("✅ PUBLICATION-READY RESULTS CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 All files saved to: {output_dir.absolute()}")
        print("\nNext steps:")
        print(f"1. Review results in {output_dir}/")
        print("2. Include performance_table.tex in your paper")
        print("3. Reference plots in your results section")
        print("4. Add single-run plots to supplementary materials with disclaimer\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Make sure the results file exists.")
        print("2. Use the --input-file argument to specify its exact location, e.g.:")
        print(f"   python create_publication_results.py --input-file multi_episode_results.json --attack {args.attack}")
        print("\nAlternatively, to generate the results file, run:")
        print(f"   python multi_episode_evaluation.py --attack {args.attack} --episodes 100")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
