"""
K-Shot Ablation Study with Multi-Episode Evaluation
====================================================

This script runs a comprehensive ablation study over different k_shot values
with MULTI-EPISODE evaluation for statistical robustness.

Ablation values: k_shot ∈ {5, 10, 20, 50, 100, 152}
Episodes per k_shot: 100 (configurable)

Metrics evaluated (with mean ± std):
- Base Model: Accuracy, Precision, Recall, F1 Score, ZDR
- TTT Model: Accuracy, Precision, Recall, F1 Score, ZDR
- Statistical significance (p-values)

Usage:
    python run_kshot_ablation_multiepisode.py --episodes 100

    # Quick test (10 episodes per k_shot)
    python run_kshot_ablation_multiepisode.py --episodes 10

Output:
    - Individual results: ablation_results_multiepisode/k_shot_{value}_results.json
    - Comprehensive table: ablation_results_multiepisode/kshot_ablation_summary.json
    - LaTeX table: ablation_results_multiepisode/kshot_ablation_table.tex
    - Visualization: ablation_results_multiepisode/kshot_performance_plot.png
"""

import subprocess
import json
import os
import shutil
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from scipy import stats

# K-shot values to evaluate
K_SHOT_VALUES = [5, 10, 20, 50, 100, 152]

# Zero-day attack to use for all experiments (consistent evaluation)
ZERO_DAY_ATTACK = "Exploits"  # UNSW-NB15

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='K-Shot Ablation Study with Multi-Episode Evaluation')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes per k_shot value (default: 100)')
    parser.add_argument('--k-shot-values', type=int, nargs='+', default=None,
                       help='Custom k_shot values (default: [5, 10, 20, 50, 100, 152])')
    parser.add_argument('--attack', type=str, default='Exploits',
                       help='Zero-day attack type (default: Exploits)')
    return parser.parse_args()

def backup_config():
    """Backup original config.py"""
    print("=" * 80)
    print("BACKING UP CONFIGURATION")
    print("=" * 80)
    if os.path.exists('config.py'):
        shutil.copy('config.py', 'config.py.ablation_multiepisode_backup')
        print("✅ Backup created: config.py.ablation_multiepisode_backup")
    else:
        print("❌ config.py not found!")
        sys.exit(1)

def restore_config():
    """Restore original config.py"""
    print("\n" + "=" * 80)
    print("RESTORING ORIGINAL CONFIGURATION")
    print("=" * 80)
    if os.path.exists('config.py.ablation_multiepisode_backup'):
        shutil.copy('config.py.ablation_multiepisode_backup', 'config.py')
        print("✅ Config restored from backup")
    else:
        print("⚠️  No backup found, config.py unchanged")

def update_k_shot_in_config(k_shot_value: int):
    """Update k_shot value in config.py"""
    print(f"\n{'=' * 80}")
    print(f"UPDATING CONFIG: k_shot = {k_shot_value}")
    print("=" * 80)

    with open('config.py', 'r') as f:
        lines = f.readlines()

    modified = False
    with open('config.py', 'w') as f:
        for line in lines:
            # Update k_shot
            if 'k_shot:' in line and 'int' in line and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + f'k_shot: int = {k_shot_value}  # Ablation study value\n'
                f.write(new_line)
                print(f"   ✅ Updated k_shot = {k_shot_value}")
                modified = True

            # Update n_query to maintain ratio (use 2x k_shot for query set)
            elif 'n_query:' in line and 'int' in line and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                n_query_value = k_shot_value * 2  # 1:2 support:query ratio
                new_line = ' ' * indent + f'n_query: int = {n_query_value}  # Ablation study: 2x k_shot\n'
                f.write(new_line)
                print(f"   ✅ Updated n_query = {n_query_value} (2x k_shot)")
                modified = True

            else:
                f.write(line)

    if modified:
        print(f"✅ Configuration updated successfully")
    else:
        print(f"⚠️  Warning: k_shot line not found in config.py")
        return False

    return True

def fix_symmetric_shots_in_create_meta_tasks():
    """Fix create_meta_tasks() to use symmetric shots"""
    print(f"\n{'=' * 80}")
    print("FIXING ASYMMETRIC SHOT CONFIGURATION")
    print("=" * 80)

    model_file = 'models/transductive_fewshot_model.py'

    if not os.path.exists(model_file):
        print(f"❌ File not found: {model_file}")
        return False

    # Backup the model file
    if not os.path.exists(f'{model_file}.asymmetric_backup'):
        shutil.copy(model_file, f'{model_file}.asymmetric_backup')
        print(f"✅ Backed up: {model_file}.asymmetric_backup")

    with open(model_file, 'r') as f:
        content = f.read()

    # Replace asymmetric shot calculation
    old_pattern = 'normal_shot_target = min(100, max(64, k_shot * 2))'
    new_pattern = 'normal_shot_target = k_shot  # ABLATION: Symmetric shots (same as attack class)'

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)

        with open(model_file, 'w') as f:
            f.write(content)

        print(f"✅ Fixed asymmetric shot configuration")
        print(f"   Both Normal and Attack classes now use k_shot samples")
        return True
    else:
        print(f"⚠️  Pattern not found (may already be fixed)")
        return False

def restore_asymmetric_shots():
    """Restore original asymmetric shot configuration"""
    print(f"\n{'=' * 80}")
    print("RESTORING ASYMMETRIC SHOT CONFIGURATION")
    print("=" * 80)

    model_file = 'models/transductive_fewshot_model.py'
    backup_file = f'{model_file}.asymmetric_backup'

    if os.path.exists(backup_file):
        shutil.copy(backup_file, model_file)
        print(f"✅ Restored from: {backup_file}")
    else:
        print(f"⚠️  No backup found: {backup_file}")

def run_multi_episode_evaluation(k_shot_value: int, n_episodes: int, attack: str, results_dir: Path) -> Dict:
    """Run multi-episode evaluation for a specific k_shot value"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING MULTI-EPISODE EVALUATION: k_shot = {k_shot_value}")
    print(f"Episodes: {n_episodes}, Zero-day attack: {attack}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        # Run multi_episode_evaluation.py
        result = subprocess.run(
            [sys.executable, 'multi_episode_evaluation.py',
             '--attack', attack,
             '--episodes', str(n_episodes)],
            capture_output=True,
            text=True,
            timeout=36000  # 10 hour timeout (100 episodes can take long)
        )

        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"⚠️  Warning: multi_episode_evaluation.py exited with code {result.returncode}")
            print(f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}")
            return {
                'k_shot': k_shot_value,
                'n_episodes': n_episodes,
                'status': 'failed',
                'error': result.stderr[-1000:],
                'elapsed_time': elapsed_time
            }

        print(f"✅ Multi-episode evaluation completed in {elapsed_time/60:.1f} minutes")

        # Extract results from multi_episode_results file
        results = extract_multiepisode_results(attack, n_episodes)

        if results:
            results['k_shot'] = k_shot_value
            results['n_episodes'] = n_episodes
            results['status'] = 'success'
            results['elapsed_time'] = elapsed_time

            # Save individual result
            result_file = results_dir / f'k_shot_{k_shot_value}_results.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved: {result_file}")

            return results
        else:
            print(f"⚠️  Could not extract results from multi-episode files")
            return {
                'k_shot': k_shot_value,
                'n_episodes': n_episodes,
                'status': 'incomplete',
                'error': 'Failed to extract results',
                'elapsed_time': elapsed_time
            }

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"❌ Evaluation timed out after {elapsed_time/60:.1f} minutes")
        return {
            'k_shot': k_shot_value,
            'n_episodes': n_episodes,
            'status': 'timeout',
            'error': 'Evaluation exceeded 10 hour timeout',
            'elapsed_time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Evaluation failed: {str(e)}")
        return {
            'k_shot': k_shot_value,
            'n_episodes': n_episodes,
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time
        }

def extract_multiepisode_results(attack: str, n_episodes: int) -> Dict:
    """Extract results from multi-episode evaluation JSON file"""

    # Look for results file pattern
    result_patterns = [
        f'multi_episode_results/{attack.lower()}_{n_episodes}_episodes_phase1.json',
        f'multi_episode_results/{attack.lower()}_{n_episodes}_episodes.json',
        f'multi_episode_results/multi_episode_{attack}.json'
    ]

    for pattern in result_patterns:
        result_file = Path(pattern)
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Extract statistics
                results = {}

                # Base model statistics (FIXED: Handle nested dict structure)
                if 'base_model' in data:
                    base = data['base_model']
                    results['base_accuracy_mean'] = base.get('accuracy', {}).get('mean', 0.0)
                    results['base_accuracy_std'] = base.get('accuracy', {}).get('std', 0.0)
                    results['base_precision_mean'] = base.get('precision', {}).get('mean', 0.0)
                    results['base_precision_std'] = base.get('precision', {}).get('std', 0.0)
                    results['base_recall_mean'] = base.get('recall', {}).get('mean', 0.0)
                    results['base_recall_std'] = base.get('recall', {}).get('std', 0.0)
                    results['base_f1_mean'] = base.get('f1_score', {}).get('mean', 0.0)
                    results['base_f1_std'] = base.get('f1_score', {}).get('std', 0.0)
                    results['base_zdr_mean'] = base.get('zero_day_detection_rate', {}).get('mean', 0.0)
                    results['base_zdr_std'] = base.get('zero_day_detection_rate', {}).get('std', 0.0)

                # TTT model statistics (FIXED: Handle nested dict structure)
                if 'ttt_model' in data or 'adapted_model' in data:
                    ttt = data.get('ttt_model', data.get('adapted_model', {}))
                    results['ttt_accuracy_mean'] = ttt.get('accuracy', {}).get('mean', 0.0)
                    results['ttt_accuracy_std'] = ttt.get('accuracy', {}).get('std', 0.0)
                    results['ttt_precision_mean'] = ttt.get('precision', {}).get('mean', 0.0)
                    results['ttt_precision_std'] = ttt.get('precision', {}).get('std', 0.0)
                    results['ttt_recall_mean'] = ttt.get('recall', {}).get('mean', 0.0)
                    results['ttt_recall_std'] = ttt.get('recall', {}).get('std', 0.0)
                    results['ttt_f1_mean'] = ttt.get('f1_score', {}).get('mean', 0.0)
                    results['ttt_f1_std'] = ttt.get('f1_score', {}).get('std', 0.0)
                    results['ttt_zdr_mean'] = ttt.get('zero_day_detection_rate', {}).get('mean', 0.0)
                    results['ttt_zdr_std'] = ttt.get('zero_day_detection_rate', {}).get('std', 0.0)

                # Statistical significance
                if 'statistical_tests' in data:
                    stat = data['statistical_tests']
                    results['accuracy_pvalue'] = stat.get('accuracy_pvalue', 1.0)
                    results['f1_pvalue'] = stat.get('f1_pvalue', 1.0)
                    results['zdr_pvalue'] = stat.get('zdr_pvalue', 1.0)

                print(f"✅ Extracted results from: {result_file}")
                return results if results else None

            except Exception as e:
                print(f"Error reading {result_file}: {e}")
                continue

    print(f"⚠️  No multi-episode results file found for {attack}")
    return None

def compute_statistical_significance(results_list: List[Dict]) -> Dict:
    """Compute statistical significance across k_shot values"""

    # Extract arrays for each metric
    k_shots = [r['k_shot'] for r in results_list if r.get('status') == 'success']

    # Check if we have enough data for correlation analysis
    if len(k_shots) < 3:
        return {'note': 'Insufficient data for statistical analysis (need ≥3 k_shot values)'}

    # Extract mean values for correlation
    metrics = {}
    for metric_name in ['accuracy', 'f1', 'zdr']:
        base_means = [r.get(f'base_{metric_name}_mean', 0) for r in results_list if r.get('status') == 'success']
        ttt_means = [r.get(f'ttt_{metric_name}_mean', 0) for r in results_list if r.get('status') == 'success']

        # Compute Spearman correlation (k_shot vs performance)
        if len(k_shots) == len(base_means) and len(k_shots) > 2:
            base_corr, base_pval = stats.spearmanr(k_shots, base_means)
            ttt_corr, ttt_pval = stats.spearmanr(k_shots, ttt_means)

            metrics[metric_name] = {
                'base_correlation': float(base_corr) if not np.isnan(base_corr) else 0.0,
                'base_pvalue': float(base_pval) if not np.isnan(base_pval) else 1.0,
                'ttt_correlation': float(ttt_corr) if not np.isnan(ttt_corr) else 0.0,
                'ttt_pvalue': float(ttt_pval) if not np.isnan(ttt_pval) else 1.0,
                'interpretation': 'Positive correlation' if ttt_corr > 0.5 and ttt_pval < 0.05 else 'No significant correlation'
            }

    return metrics

def create_summary_table(all_results: List[Dict], output_dir: Path):
    """Create comprehensive summary table with statistics"""
    print(f"\n{'=' * 80}")
    print("GENERATING SUMMARY TABLE")
    print("=" * 80)

    # Filter successful results
    successful = [r for r in all_results if r.get('status') == 'success']

    if not successful:
        print("❌ No successful experiments to summarize")
        return

    # Sort by k_shot
    successful.sort(key=lambda x: x['k_shot'])

    # Compute statistical significance
    stat_tests = compute_statistical_significance(successful)

    # Create summary JSON
    summary = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'k_shot_values': [r['k_shot'] for r in successful],
        'n_episodes_per_kshot': successful[0].get('n_episodes', 0) if successful else 0,
        'total_experiments': len(K_SHOT_VALUES),
        'successful_experiments': len(successful),
        'zero_day_attack': ZERO_DAY_ATTACK,
        'statistical_analysis': stat_tests,
        'results': successful
    }

    summary_file = output_dir / 'kshot_ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_file}")

    # Create LaTeX table
    create_latex_table(successful, output_dir)

    # Create performance plot
    create_performance_plot(successful, output_dir)

    # Print console summary
    print_console_summary(successful, stat_tests)

def create_latex_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table for publication with mean ± std"""
    latex_file = output_dir / 'kshot_ablation_table.tex'

    with open(latex_file, 'w') as f:
        f.write("% K-Shot Ablation Study Results (Multi-Episode)\n")
        f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")

        f.write("\\begin{table*}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{K-Shot Ablation Study: Impact of Shot Count on Model Performance (Mean ± Std over 100 episodes)}\n")
        f.write("\\label{tab:kshot_ablation}\n")
        f.write("\\begin{tabular}{c|cccc|cccc}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{K-Shot} & \\multicolumn{4}{c|}{Base Model} & \\multicolumn{4}{c}{TTT Model} \\\\\n")
        f.write(" & Acc (\\%) & Prec (\\%) & Rec (\\%) & F1 (\\%) & Acc (\\%) & Prec (\\%) & Rec (\\%) & F1 (\\%) \\\\\n")
        f.write("\\hline\n")

        for r in results:
            f.write(f"{r['k_shot']} & ")
            f.write(f"{r.get('base_accuracy_mean', 0)*100:.1f}$\\pm${r.get('base_accuracy_std', 0)*100:.1f} & ")
            f.write(f"{r.get('base_precision_mean', 0)*100:.1f}$\\pm${r.get('base_precision_std', 0)*100:.1f} & ")
            f.write(f"{r.get('base_recall_mean', 0)*100:.1f}$\\pm${r.get('base_recall_std', 0)*100:.1f} & ")
            f.write(f"{r.get('base_f1_mean', 0)*100:.1f}$\\pm${r.get('base_f1_std', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_accuracy_mean', 0)*100:.1f}$\\pm${r.get('ttt_accuracy_std', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_precision_mean', 0)*100:.1f}$\\pm${r.get('ttt_precision_std', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_recall_mean', 0)*100:.1f}$\\pm${r.get('ttt_recall_std', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_f1_mean', 0)*100:.1f}$\\pm${r.get('ttt_f1_std', 0)*100:.1f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    print(f"✅ LaTeX table saved: {latex_file}")

def create_performance_plot(results: List[Dict], output_dir: Path):
    """Create performance visualization with error bars"""
    k_shots = [r['k_shot'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Shot Ablation Study: Performance vs Shot Count (100 Episodes)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Accuracy
    ax = axes[0, 0]
    base_acc = [r.get('base_accuracy_mean', 0)*100 for r in results]
    base_acc_std = [r.get('base_accuracy_std', 0)*100 for r in results]
    ttt_acc = [r.get('ttt_accuracy_mean', 0)*100 for r in results]
    ttt_acc_std = [r.get('ttt_accuracy_std', 0)*100 for r in results]

    ax.errorbar(k_shots, base_acc, yerr=base_acc_std, marker='o', linewidth=2,
                markersize=8, label='Base Model', capsize=5)
    ax.errorbar(k_shots, ttt_acc, yerr=ttt_acc_std, marker='s', linewidth=2,
                markersize=8, label='TTT Model', capsize=5)
    ax.set_xlabel('K-Shot Value', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs K-Shot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(k_shots)
    ax.set_xticklabels(k_shots)

    # Plot 2: F1 Score
    ax = axes[0, 1]
    base_f1 = [r.get('base_f1_mean', 0)*100 for r in results]
    base_f1_std = [r.get('base_f1_std', 0)*100 for r in results]
    ttt_f1 = [r.get('ttt_f1_mean', 0)*100 for r in results]
    ttt_f1_std = [r.get('ttt_f1_std', 0)*100 for r in results]

    ax.errorbar(k_shots, base_f1, yerr=base_f1_std, marker='o', linewidth=2,
                markersize=8, label='Base Model', capsize=5)
    ax.errorbar(k_shots, ttt_f1, yerr=ttt_f1_std, marker='s', linewidth=2,
                markersize=8, label='TTT Model', capsize=5)
    ax.set_xlabel('K-Shot Value', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('F1 Score vs K-Shot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(k_shots)
    ax.set_xticklabels(k_shots)

    # Plot 3: Recall
    ax = axes[1, 0]
    base_rec = [r.get('base_recall_mean', 0)*100 for r in results]
    base_rec_std = [r.get('base_recall_std', 0)*100 for r in results]
    ttt_rec = [r.get('ttt_recall_mean', 0)*100 for r in results]
    ttt_rec_std = [r.get('ttt_recall_std', 0)*100 for r in results]

    ax.errorbar(k_shots, base_rec, yerr=base_rec_std, marker='o', linewidth=2,
                markersize=8, label='Base Model', capsize=5)
    ax.errorbar(k_shots, ttt_rec, yerr=ttt_rec_std, marker='s', linewidth=2,
                markersize=8, label='TTT Model', capsize=5)
    ax.set_xlabel('K-Shot Value', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Recall vs K-Shot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(k_shots)
    ax.set_xticklabels(k_shots)

    # Plot 4: ZDR (Zero-Day Recall)
    ax = axes[1, 1]
    base_zdr = [r.get('base_zdr_mean', 0)*100 for r in results]
    base_zdr_std = [r.get('base_zdr_std', 0)*100 for r in results]
    ttt_zdr = [r.get('ttt_zdr_mean', 0)*100 for r in results]
    ttt_zdr_std = [r.get('ttt_zdr_std', 0)*100 for r in results]

    ax.errorbar(k_shots, base_zdr, yerr=base_zdr_std, marker='o', linewidth=2,
                markersize=8, label='Base Model', color='red', capsize=5)
    ax.errorbar(k_shots, ttt_zdr, yerr=ttt_zdr_std, marker='s', linewidth=2,
                markersize=8, label='TTT Model', color='darkred', capsize=5)
    ax.set_xlabel('K-Shot Value', fontsize=12)
    ax.set_ylabel('Zero-Day Recall (%)', fontsize=12)
    ax.set_title('Zero-Day Detection vs K-Shot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(k_shots)
    ax.set_xticklabels(k_shots)

    plt.tight_layout()

    plot_file = output_dir / 'kshot_performance_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Performance plot saved: {plot_file}")
    plt.close()

def print_console_summary(results: List[Dict], stat_tests: Dict):
    """Print formatted summary to console"""
    print(f"\n{'=' * 80}")
    print("K-SHOT ABLATION STUDY RESULTS (MULTI-EPISODE)")
    print("=" * 80)
    print(f"\nZero-Day Attack: {ZERO_DAY_ATTACK}")
    print(f"Episodes per K-shot: {results[0].get('n_episodes', 0) if results else 0}")
    print(f"Total Experiments: {len(results)}\n")

    print(f"{'K-Shot':<10} {'Base Acc':<18} {'TTT Acc':<18} {'Base ZDR':<18} {'TTT ZDR':<18}")
    print("-" * 90)

    for r in results:
        print(f"{r['k_shot']:<10} "
              f"{r.get('base_accuracy_mean', 0)*100:>6.2f}±{r.get('base_accuracy_std', 0)*100:<6.2f}%  "
              f"{r.get('ttt_accuracy_mean', 0)*100:>6.2f}±{r.get('ttt_accuracy_std', 0)*100:<6.2f}%  "
              f"{r.get('base_zdr_mean', 0)*100:>6.2f}±{r.get('base_zdr_std', 0)*100:<6.2f}%  "
              f"{r.get('ttt_zdr_mean', 0)*100:>6.2f}±{r.get('ttt_zdr_std', 0)*100:<6.2f}%")

    print("=" * 90)

    # Statistical significance summary
    if stat_tests and 'accuracy' in stat_tests:
        print(f"\nSTATISTICAL SIGNIFICANCE (Spearman Correlation):")
        print(f"  Accuracy: r={stat_tests['accuracy']['ttt_correlation']:.3f}, "
              f"p={stat_tests['accuracy']['ttt_pvalue']:.4f} "
              f"({'significant' if stat_tests['accuracy']['ttt_pvalue'] < 0.05 else 'not significant'})")
        print(f"  F1 Score: r={stat_tests['f1']['ttt_correlation']:.3f}, "
              f"p={stat_tests['f1']['ttt_pvalue']:.4f} "
              f"({'significant' if stat_tests['f1']['ttt_pvalue'] < 0.05 else 'not significant'})")
        print(f"  ZDR:      r={stat_tests['zdr']['ttt_correlation']:.3f}, "
              f"p={stat_tests['zdr']['ttt_pvalue']:.4f} "
              f"({'significant' if stat_tests['zdr']['ttt_pvalue'] < 0.05 else 'not significant'})")

def main():
    """Main execution flow"""
    args = parse_args()

    k_shot_values = args.k_shot_values if args.k_shot_values else K_SHOT_VALUES
    n_episodes = args.episodes
    attack = args.attack

    print("\n" + "=" * 80)
    print("K-SHOT ABLATION STUDY (MULTI-EPISODE)")
    print("=" * 80)
    print(f"K-Shot values: {k_shot_values}")
    print(f"Episodes per k_shot: {n_episodes}")
    print(f"Zero-day attack: {attack}")
    print(f"Total experiments: {len(k_shot_values)}")
    print(f"Estimated time: {len(k_shot_values) * n_episodes * 2 / 60:.1f} hours")
    print("=" * 80)

    # Create results directory
    results_dir = Path('ablation_results_multiepisode')
    results_dir.mkdir(exist_ok=True)
    print(f"\n✅ Results directory: {results_dir}")

    # Backup configuration files
    backup_config()

    # Fix asymmetric shot configuration ONCE before all experiments
    fix_symmetric_shots_in_create_meta_tasks()

    all_results = []

    try:
        # Run experiments for each k_shot value
        for k_shot in k_shot_values:
            # Update config for this k_shot
            if not update_k_shot_in_config(k_shot):
                print(f"❌ Failed to update config for k_shot={k_shot}, skipping...")
                continue

            # Run multi-episode evaluation
            result = run_multi_episode_evaluation(k_shot, n_episodes, attack, results_dir)
            all_results.append(result)

            print(f"\n{'=' * 80}")
            print(f"EXPERIMENT SUMMARY: k_shot = {k_shot}")
            print(f"Status: {result.get('status', 'unknown')}")
            if result.get('status') == 'success':
                print(f"Base Accuracy: {result.get('base_accuracy_mean', 0)*100:.2f}% ± {result.get('base_accuracy_std', 0)*100:.2f}%")
                print(f"TTT Accuracy: {result.get('ttt_accuracy_mean', 0)*100:.2f}% ± {result.get('ttt_accuracy_std', 0)*100:.2f}%")
                print(f"Base ZDR: {result.get('base_zdr_mean', 0)*100:.2f}% ± {result.get('base_zdr_std', 0)*100:.2f}%")
                print(f"TTT ZDR: {result.get('ttt_zdr_mean', 0)*100:.2f}% ± {result.get('ttt_zdr_std', 0)*100:.2f}%")
            print("=" * 80)

        # Generate summary
        create_summary_table(all_results, results_dir)

    finally:
        # Restore original configuration
        restore_config()
        restore_asymmetric_shots()
        print(f"\n✅ Ablation study completed!")
        print(f"   Results saved in: {results_dir}")

if __name__ == '__main__':
    main()
