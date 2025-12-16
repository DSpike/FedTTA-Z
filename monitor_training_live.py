#!/usr/bin/env python3
"""
Real-time Training Monitor
Watches run_log.txt and displays key metrics as they appear
"""

import time
import os
from pathlib import Path

def monitor_log(log_file='run_log.txt', check_interval=0.5):
    """Monitor log file in real-time"""

    log_path = Path(log_file)

    print("="*80)
    print("🔍 REAL-TIME TRAINING MONITOR")
    print("="*80)
    print(f"Watching: {log_path.absolute()}")
    print("Press Ctrl+C to stop monitoring")
    print("="*80)
    print()

    # Track what we've seen
    seen_lines = 0

    # Key patterns to highlight
    patterns = {
        'Epoch': '📊',
        'TTT Step': '🔄',
        'TTT adaptation completed': '✅',
        'Loss=': '📉',
        'Accuracy': '🎯',
        'Zero-Day Detection Rate': '🎯',
        'Base Model': '📌',
        'TTT Model': '🔧',
        'ERROR': '❌',
        'WARNING': '⚠️',
        'Device:': '💻',
        'GPU:': '🚀',
        'CUDA': '⚡',
    }

    try:
        while True:
            if not log_path.exists():
                print(f"⏳ Waiting for log file to be created...")
                time.sleep(check_interval)
                continue

            # Read new lines
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

                # Show only new lines
                if len(lines) > seen_lines:
                    new_lines = lines[seen_lines:]

                    for line in new_lines:
                        # Add emoji if pattern found
                        emoji = ''
                        for pattern, icon in patterns.items():
                            if pattern in line:
                                emoji = icon + ' '
                                break

                        # Print with emoji
                        print(emoji + line.rstrip())

                    seen_lines = len(lines)

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print()
        print("="*80)
        print("👋 Monitoring stopped")
        print("="*80)

        # Show summary
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                print("\n📊 QUICK SUMMARY:")
                print("-"*80)

                # Extract key metrics
                if 'Epoch' in content:
                    epochs = [l for l in content.split('\n') if 'Epoch' in l and 'Loss=' in l]
                    if epochs:
                        print(f"Last epoch: {epochs[-1].split('INFO - ')[-1]}")

                if 'Zero-Day Detection Rate' in content:
                    zdr_lines = [l for l in content.split('\n') if 'Zero-Day Detection Rate' in l]
                    if zdr_lines:
                        print(f"Zero-Day Detection: {zdr_lines[-1].split('Rate: ')[-1]}")

                if 'TTT adaptation completed' in content:
                    ttt_lines = [l for l in content.split('\n') if 'TTT adaptation completed' in l]
                    print(f"TTT adaptations: {len(ttt_lines)}")

                print("-"*80)

if __name__ == '__main__':
    monitor_log()
