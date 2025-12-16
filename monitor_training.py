#!/usr/bin/env python3
"""
Monitor training progress and extract key metrics
"""
import time
import re
import os

log_file = "run_with_fixes_log.txt"

print("=" * 80)
print("TRAINING MONITOR - Watching for key events")
print("=" * 80)

def tail_file(filename, n=20):
    """Read last n lines of a file"""
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except:
        return []

def check_for_patterns(lines):
    """Check for important patterns in log lines"""
    patterns = {
        'ttt_failed': r'TTT Adaptation FAILED',
        'ttt_start': r'PHASE 2: TTT ADAPTATION',
        'ttt_success': r'TTT Adaptation completed',
        'param_change': r'Parameter change: ([\d\.e\-\+]+)',
        'pred_diff': r'Prediction difference: ([\d\.]+)%',
        'zero_day_samples': r'Zero-day samples: (\d+)',
        'base_accuracy': r'Base Model.*Accuracy: ([\d\.]+)',
        'ttt_accuracy': r'Adapted Model.*Accuracy: ([\d\.]+)',
        'base_zdr': r'Base.*Zero-Day Detection Rate: ([\d\.]+)',
        'ttt_zdr': r'TTT.*Zero-Day Detection Rate: ([\d\.]+)',
        'base_f1': r'Base Model.*F1-Score: ([\d\.]+)',
        'ttt_f1': r'TTT.*F1-Score: ([\d\.]+)',
    }

    results = {}
    for line in lines:
        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                if match.groups():
                    results[key] = match.group(1)
                else:
                    results[key] = True

    return results

last_size = 0
check_count = 0

print("\nMonitoring started... (Ctrl+C to stop)")
print("-" * 80)

try:
    while True:
        if os.path.exists(log_file):
            current_size = os.path.getsize(log_file)
            if current_size != last_size:
                last_size = current_size
                check_count += 1

                lines = tail_file(log_file, 50)
                results = check_for_patterns(lines)

                if results:
                    print(f"\n[Check #{check_count}] Found events:")

                    if 'ttt_start' in results:
                        print("  🔄 TTT Adaptation STARTED")

                    if 'ttt_failed' in results:
                        print("  ❌ TTT Adaptation FAILED!")

                    if 'ttt_success' in results:
                        print("  ✅ TTT Adaptation COMPLETED")

                    if 'param_change' in results:
                        value = float(results['param_change'])
                        if value > 0.001:
                            print(f"  ✅ Parameter change: {value:.6f} (GOOD)")
                        else:
                            print(f"  ⚠️  Parameter change: {value:.6f} (too small)")

                    if 'pred_diff' in results:
                        value = float(results['pred_diff'])
                        if value > 10:
                            print(f"  ✅ Prediction difference: {value}% (GOOD)")
                        else:
                            print(f"  ⚠️  Prediction difference: {value}% (too small)")

                    if 'zero_day_samples' in results:
                        value = int(results['zero_day_samples'])
                        if value > 0:
                            print(f"  ✅ Zero-day samples: {value}")
                        else:
                            print(f"  ❌ Zero-day samples: 0 (PROBLEM!)")

                    if 'base_accuracy' in results:
                        print(f"  📊 Base Model Accuracy: {results['base_accuracy']}")

                    if 'ttt_accuracy' in results:
                        print(f"  📊 TTT Model Accuracy: {results['ttt_accuracy']}")

                    if 'base_f1' in results:
                        print(f"  📊 Base Model F1: {results['base_f1']}")

                    if 'ttt_f1' in results:
                        print(f"  📊 TTT Model F1: {results['ttt_f1']}")

                    if 'base_zdr' in results:
                        print(f"  📊 Base ZDR: {results['base_zdr']}")

                    if 'ttt_zdr' in results:
                        print(f"  📊 TTT ZDR: {results['ttt_zdr']}")

        time.sleep(5)

except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
    print("=" * 80)
