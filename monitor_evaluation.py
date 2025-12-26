"""
Monitor the progress of the running multi-episode evaluation

Checks the log file and displays:
- Current episode
- Estimated completion time
- Latest metrics
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

def monitor_evaluation():
    """Monitor evaluation progress from log file."""

    # Check for the Exploits evaluation log file (prioritize latest)
    log_file = Path("multi_episode_exploits_threshold_0.85_baseline.log")
    if not log_file.exists():
        log_file = Path("multi_episode_exploits_threshold_0.78.log")
    if not log_file.exists():
        log_file = Path("multi_episode_evaluation_optimized_threshold_exploits.log")
    if not log_file.exists():
        log_file = Path("multi_episode_evaluation_optimized_threshold.log")

    if not log_file.exists():
        print("❌ Log file not found. Evaluation may not have started.")
        return

    # Read log file
    with open(log_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("⏳ Evaluation starting...")
        return

    # Parse progress
    current_episode = None
    total_episodes = 100
    started_at = None
    completed_episodes = []

    for line in lines:
        # Check for episode completion
        match = re.search(r'Episode (\d+)/(\d+)', line)
        if match:
            current_episode = int(match.group(1))
            total_episodes = int(match.group(2))

        # Check for episode results
        if '✅ Episode' in line and 'completed' in line:
            episode_match = re.search(r'Episode (\d+)', line)
            if episode_match:
                completed_episodes.append(int(episode_match.group(1)))

        # Get start time
        if started_at is None:
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    started_at = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                except:
                    pass

    # Display progress
    print("=" * 80)
    print("EVALUATION PROGRESS MONITOR")
    print("=" * 80)

    if started_at:
        elapsed = datetime.now() - started_at
        print(f"\n⏱️  Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Elapsed: {elapsed}")

    completed_count = len(set(completed_episodes))

    if current_episode is not None:
        progress_pct = (current_episode / total_episodes) * 100
        print(f"\n📊 Progress: Episode {current_episode}/{total_episodes} ({progress_pct:.1f}%)")
        print(f"   Completed: {completed_count} episodes")

        # Estimate time remaining
        if completed_count > 0 and started_at:
            avg_time_per_episode = elapsed / completed_count
            remaining_episodes = total_episodes - completed_count
            estimated_remaining = avg_time_per_episode * remaining_episodes
            estimated_completion = datetime.now() + estimated_remaining

            print(f"\n⏳ Estimated Time:")
            print(f"   Remaining: {estimated_remaining}")
            print(f"   Completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n📊 Progress: Initializing...")
        print(f"   Total episodes: {total_episodes}")

    # Show recent metrics if available
    print(f"\n📈 Recent Activity:")
    recent_lines = lines[-20:] if len(lines) > 20 else lines

    for line in recent_lines:
        if any(keyword in line for keyword in ['Episode', 'ZDR', 'FAR', 'Accuracy', '✅', '❌']):
            # Clean up the line
            clean_line = line.strip()
            if clean_line:
                # Truncate long lines
                if len(clean_line) > 100:
                    clean_line = clean_line[:97] + "..."
                print(f"   {clean_line}")

    print("\n" + "=" * 80)
    print("💡 Tip: Run this script again to see updated progress")
    print("=" * 80)

if __name__ == "__main__":
    monitor_evaluation()
