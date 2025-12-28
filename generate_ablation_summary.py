"""
Generate Comprehensive Ablation Study Summary

Creates publication-ready summaries from existing multi-episode results including:
- All attack types (Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms)
- Statistical analysis across attack types
- LaTeX tables
- Performance plots
- Publication summary

Usage:
    python generate_ablation_summary.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

def load_multi_episode_results():
    """Load all available 100-episode results"""
    results_dir = Path('multi_episode_results')

    attacks = [
        'Analysis', 'Backdoor', 'DoS', 'Exploits',
        'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
    ]

    results = {}

    for attack in attacks:
        filename = f'{attack.lower()}_100_episodes_phase1.json'
        filepath = results_dir / filename

        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[attack] = data
                    print(f"✅ Loaded: {attack} (100 episodes)")
            except Exception as e:
                print(f"❌ Failed to load {attack}: {e}")
        else:
            print(f"⚠️  Missing: {attack}")

    return results

def extract_metrics(attack_results):
    """Extract key metrics from attack results"""
    base = attack_results.get('base_model', {})
    ttt = attack_results.get('ttt_model', {}) or attack_results.get('adapted_model', {})

    metrics = {
        'base_accuracy': base.get('accuracy', {}).get('mean', 0),
        'base_accuracy_std': base.get('accuracy', {}).get('std', 0),
        'base_precision': base.get('precision', {}).get('mean', 0),
        'base_precision_std': base.get('precision', {}).get('std', 0),
        'base_recall': base.get('recall', {}).get('mean', 0),
        'base_recall_std': base.get('recall', {}).get('std', 0),
        'base_f1': base.get('f1_score', {}).get('mean', 0),
        'base_f1_std': base.get('f1_score', {}).get('std', 0),
        'base_zdr': base.get('zero_day_detection_rate', {}).get('mean', 0),
        'base_zdr_std': base.get('zero_day_detection_rate', {}).get('std', 0),

        'ttt_accuracy': ttt.get('accuracy', {}).get('mean', 0),
        'ttt_accuracy_std': ttt.get('accuracy', {}).get('std', 0),
        'ttt_precision': ttt.get('precision', {}).get('mean', 0),
        'ttt_precision_std': ttt.get('precision', {}).get('std', 0),
        'ttt_recall': ttt.get('recall', {}).get('mean', 0),
        'ttt_recall_std': ttt.get('recall', {}).get('std', 0),
        'ttt_f1': ttt.get('f1_score', {}).get('mean', 0),
        'ttt_f1_std': ttt.get('f1_score', {}).get('std', 0),
        'ttt_zdr': ttt.get('zero_day_detection_rate', {}).get('mean', 0),
        'ttt_zdr_std': ttt.get('zero_day_detection_rate', {}).get('std', 0),

        'n_episodes': attack_results.get('metadata', {}).get('n_episodes', 0),
    }

    # Calculate improvements
    metrics['accuracy_improvement'] = metrics['ttt_accuracy'] - metrics['base_accuracy']
    metrics['f1_improvement'] = metrics['ttt_f1'] - metrics['base_f1']
    metrics['zdr_improvement'] = metrics['ttt_zdr'] - metrics['base_zdr']

    return metrics

def create_latex_table(all_results, output_dir):
    """Create publication-ready LaTeX table"""

    table_file = output_dir / 'multi_attack_ablation_table.tex'

    with open(table_file, 'w') as f:
        f.write("% Multi-Attack Ablation Study Results\n")
        f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")

        f.write("\\begin{table*}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Across Multiple Zero-Day Attack Types (Mean ± Std over 100 episodes, k\\_shot=152)}\n")
        f.write("\\label{tab:multi_attack_ablation}\n")
        f.write("\\begin{tabular}{l|cccc|cccc}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{Attack Type} & \\multicolumn{4}{c|}{Base Model} & \\multicolumn{4}{c}{TTT Model} \\\\\n")
        f.write(" & Acc (\\%) & Prec (\\%) & Rec (\\%) & ZDR (\\%) & Acc (\\%) & Prec (\\%) & Rec (\\%) & ZDR (\\%) \\\\\n")
        f.write("\\hline\n")

        for attack, metrics in sorted(all_results.items()):
            f.write(f"{attack} & ")
            f.write(f"{metrics['base_accuracy']*100:.1f}$\\pm${metrics['base_accuracy_std']*100:.1f} & ")
            f.write(f"{metrics['base_precision']*100:.1f}$\\pm${metrics['base_precision_std']*100:.1f} & ")
            f.write(f"{metrics['base_recall']*100:.1f}$\\pm${metrics['base_recall_std']*100:.1f} & ")
            f.write(f"{metrics['base_zdr']*100:.1f}$\\pm${metrics['base_zdr_std']*100:.1f} & ")
            f.write(f"{metrics['ttt_accuracy']*100:.1f}$\\pm${metrics['ttt_accuracy_std']*100:.1f} & ")
            f.write(f"{metrics['ttt_precision']*100:.1f}$\\pm${metrics['ttt_precision_std']*100:.1f} & ")
            f.write(f"{metrics['ttt_recall']*100:.1f}$\\pm${metrics['ttt_recall_std']*100:.1f} & ")
            f.write(f"{metrics['ttt_zdr']*100:.1f}$\\pm${metrics['ttt_zdr_std']*100:.1f} \\\\\n")

        # Add average row
        avg_metrics = compute_average_metrics(all_results)
        f.write("\\hline\n")
        f.write("Average & ")
        f.write(f"{avg_metrics['base_accuracy']*100:.1f}$\\pm${avg_metrics['base_accuracy_std']*100:.1f} & ")
        f.write(f"{avg_metrics['base_precision']*100:.1f}$\\pm${avg_metrics['base_precision_std']*100:.1f} & ")
        f.write(f"{avg_metrics['base_recall']*100:.1f}$\\pm${avg_metrics['base_recall_std']*100:.1f} & ")
        f.write(f"{avg_metrics['base_zdr']*100:.1f}$\\pm${avg_metrics['base_zdr_std']*100:.1f} & ")
        f.write(f"{avg_metrics['ttt_accuracy']*100:.1f}$\\pm${avg_metrics['ttt_accuracy_std']*100:.1f} & ")
        f.write(f"{avg_metrics['ttt_precision']*100:.1f}$\\pm${avg_metrics['ttt_precision_std']*100:.1f} & ")
        f.write(f"{avg_metrics['ttt_recall']*100:.1f}$\\pm${avg_metrics['ttt_recall_std']*100:.1f} & ")
        f.write(f"{avg_metrics['ttt_zdr']*100:.1f}$\\pm${avg_metrics['ttt_zdr_std']*100:.1f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    print(f"✅ LaTeX table saved: {table_file}")

def compute_average_metrics(all_results):
    """Compute average across all attacks"""
    metrics_list = list(all_results.values())

    avg = {}
    for key in ['base_accuracy', 'base_precision', 'base_recall', 'base_zdr',
                'ttt_accuracy', 'ttt_precision', 'ttt_recall', 'ttt_zdr',
                'base_accuracy_std', 'base_precision_std', 'base_recall_std', 'base_zdr_std',
                'ttt_accuracy_std', 'ttt_precision_std', 'ttt_recall_std', 'ttt_zdr_std']:
        values = [m[key] for m in metrics_list if key in m]
        avg[key] = np.mean(values) if values else 0.0

    return avg

def create_performance_plots(all_results, output_dir):
    """Create performance visualization"""

    attacks = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Attack Performance Analysis (100 episodes, k_shot=152)',
                 fontsize=16, fontweight='bold')

    x_pos = np.arange(len(attacks))
    width = 0.35

    # Plot 1: Accuracy
    ax = axes[0, 0]
    base_acc = [all_results[a]['base_accuracy']*100 for a in attacks]
    base_acc_std = [all_results[a]['base_accuracy_std']*100 for a in attacks]
    ttt_acc = [all_results[a]['ttt_accuracy']*100 for a in attacks]
    ttt_acc_std = [all_results[a]['ttt_accuracy_std']*100 for a in attacks]

    ax.bar(x_pos - width/2, base_acc, width, yerr=base_acc_std, label='Base', capsize=5, alpha=0.8)
    ax.bar(x_pos + width/2, ttt_acc, width, yerr=ttt_acc_std, label='TTT', capsize=5, alpha=0.8)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Across Attack Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: ZDR
    ax = axes[0, 1]
    base_zdr = [all_results[a]['base_zdr']*100 for a in attacks]
    base_zdr_std = [all_results[a]['base_zdr_std']*100 for a in attacks]
    ttt_zdr = [all_results[a]['ttt_zdr']*100 for a in attacks]
    ttt_zdr_std = [all_results[a]['ttt_zdr_std']*100 for a in attacks]

    ax.bar(x_pos - width/2, base_zdr, width, yerr=base_zdr_std, label='Base', capsize=5, alpha=0.8, color='red')
    ax.bar(x_pos + width/2, ttt_zdr, width, yerr=ttt_zdr_std, label='TTT', capsize=5, alpha=0.8, color='darkred')
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Zero-Day Detection Rate (%)', fontsize=12)
    ax.set_title('ZDR Across Attack Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')

    # Plot 3: F1 Score
    ax = axes[1, 0]
    base_f1 = [all_results[a]['base_f1']*100 for a in attacks]
    base_f1_std = [all_results[a]['base_f1_std']*100 for a in attacks]
    ttt_f1 = [all_results[a]['ttt_f1']*100 for a in attacks]
    ttt_f1_std = [all_results[a]['ttt_f1_std']*100 for a in attacks]

    ax.bar(x_pos - width/2, base_f1, width, yerr=base_f1_std, label='Base', capsize=5, alpha=0.8)
    ax.bar(x_pos + width/2, ttt_f1, width, yerr=ttt_f1_std, label='TTT', capsize=5, alpha=0.8)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('F1 Score Across Attack Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Improvement
    ax = axes[1, 1]
    zdr_improvement = [all_results[a]['zdr_improvement']*100 for a in attacks]
    colors = ['green' if imp > 0 else 'red' for imp in zdr_improvement]

    ax.bar(x_pos, zdr_improvement, width*2, color=colors, alpha=0.7)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('ZDR Improvement (%)', fontsize=12)
    ax.set_title('TTT Improvement in ZDR', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_file = output_dir / 'multi_attack_performance.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Performance plot saved: {plot_file}")
    plt.close()

def print_console_summary(all_results):
    """Print comprehensive console summary"""

    print("\n" + "=" * 100)
    print("MULTI-ATTACK ABLATION STUDY SUMMARY")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total attack types: {len(all_results)}")
    print(f"Episodes per attack: 100")
    print(f"Configuration: k_shot=152 (current config)\n")

    print(f"{'Attack':<15} {'Base ZDR':<18} {'TTT ZDR':<18} {'Improvement':<15} {'Significance'}")
    print("-" * 100)

    for attack in sorted(all_results.keys()):
        metrics = all_results[attack]
        base_zdr = metrics['base_zdr'] * 100
        base_zdr_std = metrics['base_zdr_std'] * 100
        ttt_zdr = metrics['ttt_zdr'] * 100
        ttt_zdr_std = metrics['ttt_zdr_std'] * 100
        improvement = metrics['zdr_improvement'] * 100

        sig = '✅ Significant' if abs(improvement) > 5 else '⚠️  Marginal'

        print(f"{attack:<15} {base_zdr:>6.2f}±{base_zdr_std:<5.2f}%  "
              f"{ttt_zdr:>6.2f}±{ttt_zdr_std:<5.2f}%  "
              f"{improvement:>+6.2f}%         {sig}")

    # Calculate and print averages
    avg = compute_average_metrics(all_results)
    print("-" * 100)
    print(f"{'AVERAGE':<15} {avg['base_zdr']*100:>6.2f}±{avg['base_zdr_std']*100:<5.2f}%  "
          f"{avg['ttt_zdr']*100:>6.2f}±{avg['ttt_zdr_std']*100:<5.2f}%  "
          f"{(avg['ttt_zdr']-avg['base_zdr'])*100:>+6.2f}%")
    print("=" * 100)

    # Summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"   Average Base ZDR: {avg['base_zdr']*100:.2f}% ± {avg['base_zdr_std']*100:.2f}%")
    print(f"   Average TTT ZDR:  {avg['ttt_zdr']*100:.2f}% ± {avg['ttt_zdr_std']*100:.2f}%")
    print(f"   Average Improvement: {(avg['ttt_zdr']-avg['base_zdr'])*100:+.2f}%")

    # Count improvements
    improvements = [m['zdr_improvement'] > 0 for m in all_results.values()]
    significant_improvements = [m['zdr_improvement'] > 0.05 for m in all_results.values()]

    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Attacks with positive improvement: {sum(improvements)}/{len(all_results)}")
    print(f"   Attacks with significant improvement (>5%): {sum(significant_improvements)}/{len(all_results)}")

    # Best and worst
    best_attack = max(all_results.items(), key=lambda x: x[1]['ttt_zdr'])
    worst_attack = min(all_results.items(), key=lambda x: x[1]['ttt_zdr'])
    most_improved = max(all_results.items(), key=lambda x: x[1]['zdr_improvement'])

    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   {best_attack[0]}: {best_attack[1]['ttt_zdr']*100:.2f}% ZDR")

    print(f"\n⚠️  WORST PERFORMANCE:")
    print(f"   {worst_attack[0]}: {worst_attack[1]['ttt_zdr']*100:.2f}% ZDR")

    print(f"\n📈 MOST IMPROVED:")
    print(f"   {most_improved[0]}: +{most_improved[1]['zdr_improvement']*100:.2f}% improvement")

    print("\n" + "=" * 100)

def create_summary_json(all_results, output_dir):
    """Create comprehensive JSON summary"""

    summary = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_attacks': len(all_results),
            'episodes_per_attack': 100,
            'k_shot': 152,
            'dataset': 'UNSW-NB15'
        },
        'average_performance': compute_average_metrics(all_results),
        'per_attack_results': all_results,
        'analysis': {
            'best_zdr': max(all_results.items(), key=lambda x: x[1]['ttt_zdr']),
            'worst_zdr': min(all_results.items(), key=lambda x: x[1]['ttt_zdr']),
            'most_improved': max(all_results.items(), key=lambda x: x[1]['zdr_improvement']),
            'attacks_with_improvement': sum(1 for m in all_results.values() if m['zdr_improvement'] > 0),
            'attacks_significantly_improved': sum(1 for m in all_results.values() if m['zdr_improvement'] > 0.05)
        }
    }

    summary_file = output_dir / 'multi_attack_ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✅ JSON summary saved: {summary_file}")

def main():
    """Main execution"""
    print("\n" + "=" * 100)
    print("GENERATING MULTI-ATTACK ABLATION STUDY SUMMARY")
    print("=" * 100)
    print()

    # Load results
    print("📂 Loading multi-episode results...")
    raw_results = load_multi_episode_results()

    if not raw_results:
        print("\n❌ No results found! Please run multi-episode evaluations first.")
        return

    print(f"\n✅ Loaded {len(raw_results)} attack types\n")

    # Extract metrics
    print("📊 Extracting metrics...")
    all_results = {}
    for attack, data in raw_results.items():
        all_results[attack] = extract_metrics(data)

    # Create output directory
    output_dir = Path('publication_results')
    output_dir.mkdir(exist_ok=True)

    # Generate outputs
    print("\n📝 Generating outputs...")
    create_latex_table(all_results, output_dir)
    create_performance_plots(all_results, output_dir)
    create_summary_json(all_results, output_dir)
    print_console_summary(all_results)

    print(f"\n✅ All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - multi_attack_ablation_table.tex (LaTeX table)")
    print(f"  - multi_attack_performance.png (Performance plots)")
    print(f"  - multi_attack_ablation_summary.json (Comprehensive summary)")

if __name__ == '__main__':
    main()
