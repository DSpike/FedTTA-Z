"""
Verification Script: Check if n_query=304 was used during training

This script analyzes the configuration and calculates expected episode structure
to verify if the model was trained with n_query=304 or n_query=16.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config_loader import get_dataset_config

def verify_n_query_configuration():
    """Verify the n_query configuration and calculate expected episode structure"""

    print("=" * 80)
    print("n_query Configuration Verification")
    print("=" * 80)

    # Load configuration
    print("\n📋 Loading configuration...")
    config = get_dataset_config()

    # Extract meta-learning parameters
    n_way = config.n_way
    k_shot = config.k_shot
    n_query = config.n_query
    num_meta_tasks = config.num_meta_tasks

    print(f"\n✅ Configuration loaded successfully")
    print(f"\nMeta-Learning Parameters:")
    print(f"  n_way:           {n_way}")
    print(f"  k_shot:          {k_shot}")
    print(f"  n_query:         {n_query}")
    print(f"  num_meta_tasks:  {num_meta_tasks}")

    # Calculate episode structure
    print(f"\n{'=' * 80}")
    print("Episode Structure Calculation")
    print("=" * 80)

    # Support set calculation
    # Based on code analysis:
    # - Normal: 64-100 shots (typically 100 if k_shot >= 50)
    # - Attack: k_shot samples
    normal_support_estimate = min(100, max(64, k_shot * 2))
    attack_support = k_shot
    total_support = normal_support_estimate + attack_support

    # Query set calculation
    # From line 3175: total_query_samples = n_query * n_way
    total_query = n_query * n_way

    # Total samples per episode
    total_samples_per_episode = total_support + total_query

    print(f"\nSupport Set (per episode):")
    print(f"  Normal samples:  ~{normal_support_estimate} (adaptive: 64-100)")
    print(f"  Attack samples:  {attack_support} (k_shot)")
    print(f"  Total support:   ~{total_support}")

    print(f"\nQuery Set (per episode):")
    print(f"  Formula:         n_query × n_way")
    print(f"  Calculation:     {n_query} × {n_way}")
    print(f"  Total query:     {total_query}")

    print(f"\nTotal Samples per Episode:")
    print(f"  Support + Query: ~{total_support} + {total_query}")
    print(f"  Total:           ~{total_samples_per_episode}")

    # Estimate episodes per epoch
    # Typical CICIDS training set: ~50,000 samples
    estimated_training_samples = 50000
    episodes_per_epoch = estimated_training_samples // total_samples_per_episode

    print(f"\nEstimated Episodes per Epoch:")
    print(f"  Training samples:    ~{estimated_training_samples} (estimated)")
    print(f"  Samples per episode: ~{total_samples_per_episode}")
    print(f"  Episodes per epoch:  ~{episodes_per_epoch}")

    # Comparison with old configuration
    print(f"\n{'=' * 80}")
    print("Comparison: n_query=16 vs n_query=304")
    print("=" * 80)

    # Old configuration
    old_n_query = 16
    old_total_query = old_n_query * n_way
    old_total_samples = total_support + old_total_query
    old_episodes_per_epoch = estimated_training_samples // old_total_samples

    print(f"\n{'Configuration':<20} {'n_query':<10} {'Total Query':<15} {'Total/Episode':<15} {'Episodes/Epoch':<15}")
    print("-" * 80)
    print(f"{'OLD (before)':<20} {old_n_query:<10} {old_total_query:<15} ~{old_total_samples:<14} ~{old_episodes_per_epoch:<14}")
    print(f"{'CURRENT (config)':<20} {n_query:<10} {total_query:<15} ~{total_samples_per_episode:<14} ~{episodes_per_epoch:<14}")

    # Determine which configuration was likely used
    print(f"\n{'=' * 80}")
    print("Verification Result")
    print("=" * 80)

    if n_query == 304:
        print(f"\n✅ Configuration file shows: n_query = {n_query}")
        print(f"\n📊 Expected training characteristics:")
        print(f"   • Support:Query ratio:  1:1 (balanced)")
        print(f"   • Query samples:         {total_query}")
        print(f"   • Episodes per epoch:    ~{episodes_per_epoch}")
        print(f"   • Support vs Query gap:  Should be < 5%")
        print(f"\n⚠️  To confirm n_query=304 was ACTUALLY used during training:")
        print(f"   1. Check training logs for 'Created X meta-learning tasks'")
        print(f"   2. Look for episodes per epoch: ~{episodes_per_epoch} indicates n_query=304")
        print(f"   3. If you see ~{old_episodes_per_epoch} episodes: n_query=16 was used instead")
        print(f"\n💡 Why this matters:")
        print(f"   • n_query=304 → Expected accuracy: 90-95%")
        print(f"   • n_query=16  → Expected accuracy: 65-75%")
        print(f"   • Current single-run: 69.57% (suggests n_query=16 or variance)")
    elif n_query == 16:
        print(f"\n❌ Configuration file shows: n_query = {n_query}")
        print(f"\n⚠️  The configuration was NOT updated correctly!")
        print(f"   Expected: n_query = 304")
        print(f"   Found:    n_query = {n_query}")
    else:
        print(f"\n⚠️  Unexpected n_query value: {n_query}")
        print(f"   Expected: 304")
        print(f"   Found:    {n_query}")

    # Support:Query ratio analysis
    print(f"\n{'=' * 80}")
    print("Support:Query Ratio Analysis")
    print("=" * 80)

    support_query_ratio = total_support / total_query
    old_support_query_ratio = total_support / old_total_query

    print(f"\nSupport:Query Ratios:")
    print(f"  OLD (n_query=16):     {total_support}:{old_total_query} = {old_support_query_ratio:.1f}:1 ❌ IMBALANCED")
    print(f"  CURRENT (n_query={n_query}): {total_support}:{total_query} = {support_query_ratio:.1f}:1", end="")

    if support_query_ratio <= 1.5:
        print(" ✅ BALANCED")
    elif support_query_ratio <= 3.0:
        print(" ⚠️  ACCEPTABLE")
    else:
        print(" ❌ IMBALANCED")

    print(f"\nMeta-Learning Best Practice:")
    print(f"  Recommended ratio: 1:1 to 3:1")
    print(f"  Your ratio:        {support_query_ratio:.1f}:1")

    if support_query_ratio <= 1.5:
        print(f"  Status:            ✅ Excellent - prevents overfitting")
    elif support_query_ratio <= 3.0:
        print(f"  Status:            ⚠️  Acceptable - may have some overfitting")
    else:
        print(f"  Status:            ❌ Poor - likely overfitting")

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print("=" * 80)

    print(f"\n✅ Current config.py setting: n_query = {n_query}")

    if n_query == 304:
        print(f"✅ Config change was applied correctly")
        print(f"\n⚠️  NEXT STEP REQUIRED:")
        print(f"   Run 100-episode validation to verify actual performance:")
        print(f"   python multi_episode_evaluation.py --attack Backdoor --episodes 100")
        print(f"\n📊 Expected results if training used n_query=304:")
        print(f"   Base Model Accuracy: 88-93% (vs 74.86% baseline)")
        print(f"   Base Model F1-Score: 85-90% (vs 78.90% baseline)")
        print(f"\n📊 If results similar to baseline (74-75% accuracy):")
        print(f"   → Training likely used old n_query=16 (cached config?)")
        print(f"   → Need to investigate why new config wasn't loaded")
    else:
        print(f"❌ Config change was NOT applied")
        print(f"   Expected n_query=304, found n_query={n_query}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    verify_n_query_configuration()
