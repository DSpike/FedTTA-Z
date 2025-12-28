"""
K-Shot Ablation Study for Few-Shot Learning
============================================

This script runs a comprehensive ablation study over different k_shot values
to evaluate the impact of shot count on model performance.

Ablation values: k_shot ∈ {5, 10, 20, 50, 100, 152}

Metrics evaluated:
- Base Model: Accuracy, Precision, Recall, F1 Score, ZDR (Zero-Day Recall)
- TTT Model: Accuracy, Precision, Recall, F1 Score, ZDR

FAR (False Alarm Rate) is REMOVED from this evaluation.

Usage:
    python run_kshot_ablation_study.py

Output:
    - Individual results: ablation_results/k_shot_{value}_results.json
    - Comprehensive table: ablation_results/kshot_ablation_summary.json
    - LaTeX table: ablation_results/kshot_ablation_table.tex
    - Visualization: ablation_results/kshot_performance_plot.png
"""

import subprocess
import json
import os
import shutil
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# K-shot values to evaluate
K_SHOT_VALUES = [5, 10, 20, 50, 100, 152]

# Zero-day attack to use for all experiments (consistent evaluation)
ZERO_DAY_ATTACK = "Exploits"  # UNSW-NB15

def backup_config():
    """Backup original config.py"""
    print("=" * 80)
    print("BACKING UP CONFIGURATION")
    print("=" * 80)
    if os.path.exists('config.py'):
        shutil.copy('config.py', 'config.py.ablation_backup')
        print("✅ Backup created: config.py.ablation_backup")
    else:
        print("❌ config.py not found!")
        sys.exit(1)

def restore_config():
    """Restore original config.py"""
    print("\n" + "=" * 80)
    print("RESTORING ORIGINAL CONFIGURATION")
    print("=" * 80)
    if os.path.exists('config.py.ablation_backup'):
        shutil.copy('config.py.ablation_backup', 'config.py')
        print("✅ Config restored from backup")
    else:
        print("⚠️  No backup found, config.py unchanged")

def update_k_shot_in_config(k_shot_value: int):
    """
    Update k_shot value in config.py

    Also updates:
    - Normal shots to match k_shot (symmetric configuration)
    - n_query to maintain reasonable support:query ratio
    """
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
                # Extract indentation
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
    """
    Fix create_meta_tasks() to use symmetric shots (same k_shot for Normal and Attack)

    This modifies transductive_fewshot_model.py to ensure:
    - Normal class uses k_shot (not hardcoded 100)
    - Attack class uses k_shot
    - Both classes have equal shot counts (standard N-way K-shot)
    """
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

    # Find and replace the asymmetric normal shot calculation
    # OLD: normal_shot_target = min(100, max(64, k_shot * 2))
    # NEW: normal_shot_target = k_shot  # Symmetric shots

    old_pattern = 'normal_shot_target = min(100, max(64, k_shot * 2))'
    new_pattern = 'normal_shot_target = k_shot  # ABLATION: Symmetric shots (same as attack class)'

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)

        with open(model_file, 'w') as f:
            f.write(content)

        print(f"✅ Fixed asymmetric shot configuration")
        print(f"   Changed: normal_shot_target = min(100, max(64, k_shot * 2))")
        print(f"   To:      normal_shot_target = k_shot")
        print(f"   Result:  Both Normal and Attack classes now use k_shot samples")
        return True
    else:
        print(f"⚠️  Pattern not found (may already be fixed or code changed)")
        print(f"   Looking for: {old_pattern}")
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

def run_experiment(k_shot_value: int, results_dir: Path) -> Dict:
    """Run experiment for a specific k_shot value"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING EXPERIMENT: k_shot = {k_shot_value}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        # Run main.py
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"⚠️  Warning: main.py exited with code {result.returncode}")
            print(f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}")
            return {
                'k_shot': k_shot_value,
                'status': 'failed',
                'error': result.stderr[-1000:],
                'elapsed_time': elapsed_time
            }

        print(f"✅ Experiment completed in {elapsed_time/60:.1f} minutes")

        # Extract results from the latest evaluation report
        results = extract_results_from_evaluation()

        if results:
            results['k_shot'] = k_shot_value
            results['status'] = 'success'
            results['elapsed_time'] = elapsed_time

            # Save individual result
            result_file = results_dir / f'k_shot_{k_shot_value}_results.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved: {result_file}")

            return results
        else:
            print(f"⚠️  Could not extract results from evaluation reports")
            return {
                'k_shot': k_shot_value,
                'status': 'incomplete',
                'error': 'Failed to extract results',
                'elapsed_time': elapsed_time
            }

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"❌ Experiment timed out after {elapsed_time/60:.1f} minutes")
        return {
            'k_shot': k_shot_value,
            'status': 'timeout',
            'error': 'Experiment exceeded 2 hour timeout',
            'elapsed_time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Experiment failed: {str(e)}")
        return {
            'k_shot': k_shot_value,
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time
        }

def extract_results_from_evaluation() -> Dict:
    """
    Extract performance metrics from the latest evaluation report

    Returns metrics for:
    - Base Model: Accuracy, Precision, Recall, F1, ZDR
    - TTT Model: Accuracy, Precision, Recall, F1, ZDR
    """
    # Look for latest evaluation summary in evaluation_reports/
    eval_dir = Path('evaluation_reports')
    if not eval_dir.exists():
        return None

    # Find most recent evaluation_summary JSON file
    json_files = list(eval_dir.glob('evaluation_summary_*.json'))
    if not json_files:
        return None

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Extract metrics (adapt based on actual structure)
        results = {}

        # Base model metrics
        if 'base_model' in data:
            base = data['base_model']
            results['base_accuracy'] = base.get('accuracy', 0.0)
            results['base_precision'] = base.get('precision', 0.0)
            results['base_recall'] = base.get('recall', 0.0)
            results['base_f1'] = base.get('f1_score', 0.0)
            results['base_zdr'] = base.get('zero_day_recall', 0.0)

        # TTT model metrics
        if 'ttt_model' in data:
            ttt = data['ttt_model']
            results['ttt_accuracy'] = ttt.get('accuracy', 0.0)
            results['ttt_precision'] = ttt.get('precision', 0.0)
            results['ttt_recall'] = ttt.get('recall', 0.0)
            results['ttt_f1'] = ttt.get('f1_score', 0.0)
            results['ttt_zdr'] = ttt.get('zero_day_recall', 0.0)

        # Improvement metrics
        if 'base_accuracy' in results and 'ttt_accuracy' in results:
            results['accuracy_improvement'] = results['ttt_accuracy'] - results['base_accuracy']
            results['f1_improvement'] = results['ttt_f1'] - results['base_f1']
            results['zdr_improvement'] = results['ttt_zdr'] - results['base_zdr']

        return results if results else None

    except Exception as e:
        print(f"Error extracting results: {e}")
        return None

def create_summary_table(all_results: List[Dict], output_dir: Path):
    """Create comprehensive summary table"""
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

    # Create summary JSON
    summary = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'k_shot_values': [r['k_shot'] for r in successful],
        'total_experiments': len(K_SHOT_VALUES),
        'successful_experiments': len(successful),
        'zero_day_attack': ZERO_DAY_ATTACK,
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
    print_console_summary(successful)

def create_latex_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table for publication"""
    latex_file = output_dir / 'kshot_ablation_table.tex'

    with open(latex_file, 'w') as f:
        f.write("% K-Shot Ablation Study Results\n")
        f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")

        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{K-Shot Ablation Study: Impact of Shot Count on Model Performance}\n")
        f.write("\\label{tab:kshot_ablation}\n")
        f.write("\\begin{tabular}{c|cccc|cccc}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{K-Shot} & \\multicolumn{4}{c|}{Base Model} & \\multicolumn{4}{c}{TTT Model} \\\\\n")
        f.write(" & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 \\\\\n")
        f.write("\\hline\n")

        for r in results:
            f.write(f"{r['k_shot']} & ")
            f.write(f"{r.get('base_accuracy', 0)*100:.1f} & ")
            f.write(f"{r.get('base_precision', 0)*100:.1f} & ")
            f.write(f"{r.get('base_recall', 0)*100:.1f} & ")
            f.write(f"{r.get('base_f1', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_accuracy', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_precision', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_recall', 0)*100:.1f} & ")
            f.write(f"{r.get('ttt_f1', 0)*100:.1f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✅ LaTeX table saved: {latex_file}")

def create_performance_plot(results: List[Dict], output_dir: Path):
    """Create performance visualization"""
    k_shots = [r['k_shot'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Shot Ablation Study: Performance vs Shot Count', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy
    ax = axes[0, 0]
    ax.plot(k_shots, [r.get('base_accuracy', 0)*100 for r in results], 'o-', label='Base Model', linewidth=2, markersize=8)
    ax.plot(k_shots, [r.get('ttt_accuracy', 0)*100 for r in results], 's-', label='TTT Model', linewidth=2, markersize=8)
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
    ax.plot(k_shots, [r.get('base_f1', 0)*100 for r in results], 'o-', label='Base Model', linewidth=2, markersize=8)
    ax.plot(k_shots, [r.get('ttt_f1', 0)*100 for r in results], 's-', label='TTT Model', linewidth=2, markersize=8)
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
    ax.plot(k_shots, [r.get('base_recall', 0)*100 for r in results], 'o-', label='Base Model', linewidth=2, markersize=8)
    ax.plot(k_shots, [r.get('ttt_recall', 0)*100 for r in results], 's-', label='TTT Model', linewidth=2, markersize=8)
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
    ax.plot(k_shots, [r.get('base_zdr', 0)*100 for r in results], 'o-', label='Base Model', linewidth=2, markersize=8, color='red')
    ax.plot(k_shots, [r.get('ttt_zdr', 0)*100 for r in results], 's-', label='TTT Model', linewidth=2, markersize=8, color='darkred')
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

def print_console_summary(results: List[Dict]):
    """Print formatted summary to console"""
    print(f"\n{'=' * 80}")
    print("K-SHOT ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"\nZero-Day Attack: {ZERO_DAY_ATTACK}")
    print(f"Total Experiments: {len(results)}\n")

    print(f"{'K-Shot':<10} {'Base Acc':<12} {'TTT Acc':<12} {'Base F1':<12} {'TTT F1':<12} {'Base ZDR':<12} {'TTT ZDR':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['k_shot']:<10} "
              f"{r.get('base_accuracy', 0)*100:>10.2f}%  "
              f"{r.get('ttt_accuracy', 0)*100:>10.2f}%  "
              f"{r.get('base_f1', 0)*100:>10.2f}%  "
              f"{r.get('ttt_f1', 0)*100:>10.2f}%  "
              f"{r.get('base_zdr', 0)*100:>10.2f}%  "
              f"{r.get('ttt_zdr', 0)*100:>10.2f}%")

    print("=" * 80)

def main():
    """Main execution flow"""
    print("\n" + "=" * 80)
    print("K-SHOT ABLATION STUDY")
    print("=" * 80)
    print(f"K-Shot values: {K_SHOT_VALUES}")
    print(f"Zero-day attack: {ZERO_DAY_ATTACK}")
    print(f"Total experiments: {len(K_SHOT_VALUES)}")
    print("=" * 80)

    # Create results directory
    results_dir = Path('ablation_results')
    results_dir.mkdir(exist_ok=True)
    print(f"\n✅ Results directory: {results_dir}")

    # Backup configuration files
    backup_config()

    # Fix asymmetric shot configuration ONCE before all experiments
    fix_symmetric_shots_in_create_meta_tasks()

    all_results = []

    try:
        # Run experiments for each k_shot value
        for k_shot in K_SHOT_VALUES:
            # Update config for this k_shot
            if not update_k_shot_in_config(k_shot):
                print(f"❌ Failed to update config for k_shot={k_shot}, skipping...")
                continue

            # Run experiment
            result = run_experiment(k_shot, results_dir)
            all_results.append(result)

            print(f"\n{'=' * 80}")
            print(f"EXPERIMENT SUMMARY: k_shot = {k_shot}")
            print(f"Status: {result.get('status', 'unknown')}")
            if result.get('status') == 'success':
                print(f"Base Accuracy: {result.get('base_accuracy', 0)*100:.2f}%")
                print(f"TTT Accuracy: {result.get('ttt_accuracy', 0)*100:.2f}%")
                print(f"Base ZDR: {result.get('base_zdr', 0)*100:.2f}%")
                print(f"TTT ZDR: {result.get('ttt_zdr', 0)*100:.2f}%")
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
