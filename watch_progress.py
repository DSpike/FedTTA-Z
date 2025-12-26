"""
Continuous progress monitor - updates every 30 seconds
Press Ctrl+C to stop
"""

import time
from pathlib import Path
import re
from datetime import datetime

def get_latest_progress():
    """Get latest episode progress from log."""
    log_file = Path("multi_episode_evaluation_optimized_threshold_exploits.log")

    if not log_file.exists():
        return None, None, None

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find latest episode number
    current_episode = None
    started_at = None
    latest_metrics = []

    for line in reversed(lines[-200:]):
        # Episode number
        if current_episode is None:
            match = re.search(r'Episode (\d+)', line)
            if match:
                current_episode = int(match.group(1))

        # Start time
        if started_at is None:
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    started_at = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                except:
                    pass

        # Metrics
        if any(kw in line for kw in ['ZDR:', 'FAR:', 'Accuracy:']):
            latest_metrics.append(line.strip())
            if len(latest_metrics) >= 5:
                break

    return current_episode, started_at, list(reversed(latest_metrics))

def format_time(seconds):
    """Format seconds into readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def main():
    """Main monitoring loop."""
    print("🔍 Starting continuous progress monitor...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            episode, started, metrics = get_latest_progress()

            # Clear screen (Windows compatible)
            print("\033[2J\033[H", end='')

            print("=" * 80)
            print("FAR OPTIMIZATION EVALUATION - LIVE PROGRESS")
            print("=" * 80)
            print()

            if started:
                elapsed = (datetime.now() - started).total_seconds()
                print(f"⏱️  Started: {started.strftime('%H:%M:%S')}")
                print(f"   Elapsed: {format_time(elapsed)}")
                print()

            if episode is not None:
                progress_pct = (episode / 100) * 100
                bar_length = 50
                filled = int(bar_length * episode / 100)
                bar = "█" * filled + "░" * (bar_length - filled)

                print(f"📊 Progress: Episode {episode}/100 ({progress_pct:.1f}%)")
                print(f"   [{bar}]")
                print()

                # Estimate completion
                if episode > 0 and started:
                    avg_time = elapsed / episode
                    remaining = avg_time * (100 - episode)
                    eta = datetime.now().timestamp() + remaining
                    eta_time = datetime.fromtimestamp(eta)

                    print(f"⏳ Estimated:")
                    print(f"   Time per episode: {format_time(avg_time)}")
                    print(f"   Remaining: {format_time(remaining)}")
                    print(f"   Completion: {eta_time.strftime('%H:%M:%S')}")
                    print()
            else:
                print("📊 Progress: Initializing...")
                print()

            if metrics:
                print("📈 Recent Metrics:")
                for metric in metrics[-5:]:
                    # Shorten long lines
                    if len(metric) > 100:
                        metric = metric[:97] + "..."
                    print(f"   {metric}")

            print()
            print("=" * 80)
            print(f"Last update: {datetime.now().strftime('%H:%M:%S')} | Next update in 30s | Ctrl+C to stop")
            print("=" * 80)

            # Wait 30 seconds
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        print("=" * 80)

if __name__ == "__main__":
    main()
