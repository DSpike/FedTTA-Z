"""
Monitor K-Shot Ablation Study Progress

Displays real-time progress of the ablation study including:
- Which k_shot value is currently running
- Episode progress (e.g., 45/100)
- Estimated time remaining
- Results summary for completed k_shot values

Usage:
    python monitor_ablation_progress.py
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

def load_results(results_dir):
    """Load all completed k_shot results"""
    results = []
    for k_shot in [5, 10, 20, 50, 100, 152]:
        result_file = results_dir / f'k_shot_{k_shot}_results.json'
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except:
                pass
    return results

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def print_progress():
    """Print current progress"""
    results_dir = Path('ablation_results_multiepisode')

    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 80)
    print("K-SHOT ABLATION STUDY - PROGRESS MONITOR")
    print("=" * 80)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load completed results
    completed_results = load_results(results_dir)

    k_shot_values = [5, 10, 20, 50, 100, 152]
    completed_k_shots = [r['k_shot'] for r in completed_results]

    print(f"Progress: {len(completed_k_shots)}/6 k_shot values completed\n")

    # Show status for each k_shot
    print(f"{'K-Shot':<10} {'Status':<15} {'Base ZDR':<18} {'TTT ZDR':<18} {'Time':<10}")
    print("-" * 80)

    total_time = 0
    for k_shot in k_shot_values:
        if k_shot in completed_k_shots:
            # Get result
            result = [r for r in completed_results if r['k_shot'] == k_shot][0]

            base_zdr = result.get('base_zdr_mean', 0) * 100
            base_zdr_std = result.get('base_zdr_std', 0) * 100
            ttt_zdr = result.get('ttt_zdr_mean', 0) * 100
            ttt_zdr_std = result.get('ttt_zdr_std', 0) * 100
            elapsed = result.get('elapsed_time', 0)
            total_time += elapsed

            print(f"{k_shot:<10} {'✅ Completed':<15} "
                  f"{base_zdr:>5.1f}±{base_zdr_std:<5.1f}%  "
                  f"{ttt_zdr:>5.1f}±{ttt_zdr_std:<5.1f}%  "
                  f"{format_time(elapsed):<10}")
        else:
            # Check if this is currently running
            if len(completed_k_shots) == k_shot_values.index(k_shot):
                print(f"{k_shot:<10} {'🔄 Running...':<15} {'---':<18} {'---':<18} {'---':<10}")
            else:
                print(f"{k_shot:<10} {'⏳ Pending':<15} {'---':<18} {'---':<18} {'---':<10}")

    print("-" * 80)

    # Estimate remaining time
    if len(completed_k_shots) > 0:
        avg_time_per_k = total_time / len(completed_k_shots)
        remaining_k = 6 - len(completed_k_shots)
        estimated_remaining = avg_time_per_k * remaining_k

        print(f"\nTotal elapsed: {format_time(total_time)}")
        print(f"Average per k_shot: {format_time(avg_time_per_k)}")
        print(f"Estimated remaining: {format_time(estimated_remaining)}")

        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        print(f"Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "=" * 80)

    # Show summary stats if any completed
    if completed_results:
        print("\nCOMPLETED RESULTS SUMMARY:")
        print(f"{'K-Shot':<10} {'Base→TTT ZDR':<25} {'ZDR Improvement':<20}")
        print("-" * 55)
        for result in sorted(completed_results, key=lambda x: x['k_shot']):
            base_zdr = result.get('base_zdr_mean', 0) * 100
            ttt_zdr = result.get('ttt_zdr_mean', 0) * 100
            improvement = ttt_zdr - base_zdr

            print(f"{result['k_shot']:<10} "
                  f"{base_zdr:>5.1f}% → {ttt_zdr:>5.1f}%       "
                  f"{improvement:>+5.1f}%  {'✅' if improvement > 5 else '⚠️'}")
        print("=" * 80)

    print("\nPress Ctrl+C to exit monitoring")

def main():
    """Main monitoring loop"""
    print("Starting ablation study progress monitor...")
    print("(Refreshes every 30 seconds)\n")

    try:
        while True:
            print_progress()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    main()
