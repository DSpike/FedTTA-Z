"""
Quick test of optimized threshold (0.78) to verify FAR reduction

Runs a smaller evaluation (5 episodes) to quickly validate:
- FAR reduced from ~39.5% to ~18-20%
- ZDR maintained at ~90%+
"""

import sys
import subprocess

# Update config temporarily for quick test
print("=" * 80)
print("TESTING OPTIMIZED THRESHOLD (0.78)")
print("=" * 80)
print("\nExpected Results:")
print("  Current (threshold=0.85): FAR=39.5%, ZDR=95.2%")
print("  Optimized (threshold=0.78): FAR~18.4%, ZDR~90.0%")
print("\nRunning quick test with 5 episodes...")
print("=" * 80)
print()

# Run multi-episode evaluation with 5 episodes
result = subprocess.run([
    sys.executable,
    "multi_episode_evaluation.py",
    "--episodes", "5",
    "--episode-size", "800"
], capture_output=False, text=True)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED - Check multi_episode_results.json for results")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print(f"❌ TEST FAILED with return code {result.returncode}")
    print("=" * 80)
