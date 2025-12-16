#!/usr/bin/env python3
"""
TTT Loss Component Monitor
Specifically tracks TTT adaptation loss components in real-time
"""

import time
import re
from pathlib import Path

def extract_ttt_metrics(line):
    """Extract TTT step metrics from log line"""
    # Pattern: TTT Step 20/83: Loss=1.6013, Entropy=2.7505, Pseudo=0.0033, L2_Reg=1.2525
    pattern = r'TTT Step (\d+)/(\d+): Loss=([\d.]+), Entropy=([\d.]+), Pseudo=([\d.]+), L2_Reg=([\d.]+)'
    match = re.search(pattern, line)

    if match:
        step, total, loss, entropy, pseudo, l2 = match.groups()
        return {
            'step': int(step),
            'total': int(total),
            'loss': float(loss),
            'entropy': float(entropy),
            'pseudo': float(pseudo),
            'l2_reg': float(l2)
        }
    return None

def monitor_ttt_losses(log_file='run_log.txt', check_interval=0.5):
    """Monitor TTT loss components in real-time"""

    log_path = Path(log_file)

    print("="*90)
    print("🔍 TTT LOSS COMPONENT MONITOR - Optimized Parameters Active")
    print("="*90)
    print(f"Watching: {log_path.absolute()}")
    print()
    print("🎯 TARGET METRICS (After Optimization):")
    print("  ✅ Entropy:  ~1.0-1.5  (NOT 2.0+)")
    print("  ✅ Pseudo:   ~0.5-1.5  (NOT 0.006)")
    print("  ✅ L2_Reg:   ~0.01-0.05 (NOT 6.5)")
    print("="*90)
    print()

    seen_lines = 0
    ttt_count = 0
    current_run_metrics = []

    try:
        while True:
            if not log_path.exists():
                print(f"⏳ Waiting for log file...")
                time.sleep(check_interval)
                continue

            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

                if len(lines) > seen_lines:
                    new_lines = lines[seen_lines:]

                    for line in new_lines:
                        # Check for TTT start
                        if 'Starting TTT adaptation' in line or 'TTT adaptation (' in line:
                            ttt_count += 1
                            current_run_metrics = []
                            print(f"\n{'='*90}")
                            print(f"🔄 TTT Run #{ttt_count} Started")
                            print(f"{'='*90}")

                        # Extract TTT metrics
                        metrics = extract_ttt_metrics(line)
                        if metrics:
                            current_run_metrics.append(metrics)

                            # Display every 20 steps
                            if metrics['step'] % 20 == 0:
                                print(f"  Step {metrics['step']:3d}/{metrics['total']:3d}: "
                                      f"Loss={metrics['loss']:.4f}, "
                                      f"Entropy={metrics['entropy']:.4f} "
                                      f"{'✅' if metrics['entropy'] < 1.5 else '⚠️'}, "
                                      f"Pseudo={metrics['pseudo']:.4f} "
                                      f"{'✅' if metrics['pseudo'] > 0.5 else '⚠️'}, "
                                      f"L2={metrics['l2_reg']:.4f} "
                                      f"{'✅' if metrics['l2_reg'] < 1.0 else '⚠️'}")

                        # Check for TTT completion
                        if 'TTT adaptation completed' in line:
                            # Extract final loss
                            match = re.search(r'final loss: ([\d.]+)', line)
                            if match:
                                final_loss = float(match.group(1))
                                print(f"✅ Completed: Final Loss = {final_loss:.4f}")

                            # Show summary if we have metrics
                            if current_run_metrics:
                                print(f"\n  📊 Run Summary:")
                                avg_entropy = sum(m['entropy'] for m in current_run_metrics) / len(current_run_metrics)
                                avg_pseudo = sum(m['pseudo'] for m in current_run_metrics) / len(current_run_metrics)
                                avg_l2 = sum(m['l2_reg'] for m in current_run_metrics) / len(current_run_metrics)

                                print(f"     Avg Entropy:  {avg_entropy:.4f} {'✅ Good' if avg_entropy < 1.5 else '⚠️ High'}")
                                print(f"     Avg Pseudo:   {avg_pseudo:.4f} {'✅ Good' if avg_pseudo > 0.5 else '⚠️ Low'}")
                                print(f"     Avg L2_Reg:   {avg_l2:.4f} {'✅ Good' if avg_l2 < 1.0 else '⚠️ High'}")

                    seen_lines = len(lines)

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print()
        print("="*90)
        print("👋 Monitoring stopped")
        print("="*90)

if __name__ == '__main__':
    monitor_ttt_losses()
