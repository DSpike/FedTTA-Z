#!/usr/bin/env python3
"""
Verify ROC AUC values in multi-episode results

This script checks if ROC AUC values are being calculated correctly
for both base and TTT models across all episodes.

Usage:
    python verify_roc_auc_values.py --input multi_episode_results/backdoor_100_episodes_phase1.json
"""

import json
import argparse
import numpy as np
from pathlib import Path


def verify_roc_auc(results_file):
    """Verify ROC AUC values in results"""

    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    print("\n" + "="*80)
    print("ROC AUC VERIFICATION")
    print("="*80)

    # Check aggregated results
    print("\n📊 AGGREGATED RESULTS:")
    print("-" * 80)

    if 'base_model' in data and 'roc_auc' in data['base_model']:
        base_roc = data['base_model']['roc_auc']
        print(f"Base Model ROC AUC:  {base_roc['mean']:.4f} ± {base_roc.get('ci_95', 0):.4f}")
        print(f"  Min: {base_roc['min']:.4f}, Max: {base_roc['max']:.4f}")
    else:
        print("⚠️  Base Model ROC AUC not found in aggregated results")

    if 'ttt_model' in data and 'roc_auc' in data['ttt_model']:
        ttt_roc = data['ttt_model']['roc_auc']
        print(f"TTT Model ROC AUC:   {ttt_roc['mean']:.4f} ± {ttt_roc.get('ci_95', 0):.4f}")
        print(f"  Min: {ttt_roc['min']:.4f}, Max: {ttt_roc['max']:.4f}")
    else:
        print("⚠️  TTT Model ROC AUC not found in aggregated results")

    # Check per-episode results
    print("\n📋 PER-EPISODE RESULTS:")
    print("-" * 80)

    if 'per_episode_results' not in data:
        print("⚠️  No per-episode results found")
        return

    episodes = data['per_episode_results']
    print(f"Total episodes: {len(episodes)}\n")

    # Collect ROC AUC values
    base_roc_values = []
    ttt_roc_values = []

    issues = []

    for i, ep in enumerate(episodes):
        ep_id = ep.get('episode_id', i)

        # Base model ROC AUC
        base_roc = ep.get('base_model', {}).get('roc_auc', None)
        if base_roc is None or base_roc == 0.0:
            issues.append(f"Episode {ep_id}: Base ROC AUC is {base_roc}")
        else:
            base_roc_values.append(base_roc)

        # TTT model ROC AUC
        ttt_roc = ep.get('ttt_model', {}).get('roc_auc', None)
        if ttt_roc is None or ttt_roc == 0.0:
            issues.append(f"Episode {ep_id}: TTT ROC AUC is {ttt_roc}")
        else:
            ttt_roc_values.append(ttt_roc)

    # Summary
    print(f"Episodes with valid Base ROC AUC: {len(base_roc_values)}/{len(episodes)}")
    print(f"Episodes with valid TTT ROC AUC:  {len(ttt_roc_values)}/{len(episodes)}")

    if base_roc_values:
        print(f"\nBase ROC AUC statistics:")
        print(f"  Mean: {np.mean(base_roc_values):.4f}")
        print(f"  Std:  {np.std(base_roc_values):.4f}")
        print(f"  Min:  {np.min(base_roc_values):.4f}")
        print(f"  Max:  {np.max(base_roc_values):.4f}")

    if ttt_roc_values:
        print(f"\nTTT ROC AUC statistics:")
        print(f"  Mean: {np.mean(ttt_roc_values):.4f}")
        print(f"  Std:  {np.std(ttt_roc_values):.4f}")
        print(f"  Min:  {np.min(ttt_roc_values):.4f}")
        print(f"  Max:  {np.max(ttt_roc_values):.4f}")

    # Report issues
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        print("-" * 80)
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("\n✅ NO ISSUES FOUND - All episodes have valid ROC AUC values")

    # Check for suspicious patterns
    print("\n🔍 CHECKING FOR SUSPICIOUS PATTERNS:")
    print("-" * 80)

    # Check if all values are the same (suspicious)
    if base_roc_values and len(set(base_roc_values)) == 1:
        print(f"⚠️  WARNING: All Base ROC AUC values are identical: {base_roc_values[0]:.4f}")

    if ttt_roc_values and len(set(ttt_roc_values)) == 1:
        print(f"⚠️  WARNING: All TTT ROC AUC values are identical: {ttt_roc_values[0]:.4f}")

    # Check if values are suspiciously low or high
    if base_roc_values:
        low_count = sum(1 for v in base_roc_values if v < 0.6)
        if low_count > len(base_roc_values) * 0.5:
            print(f"⚠️  WARNING: {low_count}/{len(base_roc_values)} Base ROC AUC values are below 0.6")

    if ttt_roc_values:
        low_count = sum(1 for v in ttt_roc_values if v < 0.6)
        if low_count > len(ttt_roc_values) * 0.5:
            print(f"⚠️  WARNING: {low_count}/{len(ttt_roc_values)} TTT ROC AUC values are below 0.6")

    # Check if TTT is worse than base (suspicious)
    if base_roc_values and ttt_roc_values:
        worse_count = sum(1 for b, t in zip(base_roc_values, ttt_roc_values) if t < b)
        if worse_count > len(base_roc_values) * 0.5:
            print(f"⚠️  WARNING: TTT ROC AUC is worse than Base in {worse_count}/{len(base_roc_values)} episodes")
        else:
            print(f"✅ TTT ROC AUC is better than Base in {len(base_roc_values) - worse_count}/{len(base_roc_values)} episodes")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Verify ROC AUC values in multi-episode results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to the multi-episode results JSON file')

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ Error: File not found: {input_file}")
        return 1

    verify_roc_auc(input_file)
    return 0


if __name__ == "__main__":
    exit(main())
