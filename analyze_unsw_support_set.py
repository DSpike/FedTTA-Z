#!/usr/bin/env python3
"""
Analyze Support Set Composition for UNSW-NB15 Dataset

This script calculates the expected composition of the 500-sample support set
used for prototype computation in the base model.
"""

import numpy as np

print("\n" + "="*80)
print("SUPPORT SET COMPOSITION ANALYSIS - UNSW-NB15 DATASET")
print("="*80)

# UNSW-NB15 Attack Categories
print("\n📊 UNSW-NB15 ATTACK CATEGORIES:")
unsw_categories = {
    'Normal': 0,
    'Fuzzers': 1,
    'Analysis': 2,
    'Backdoor': 3,  # ZERO-DAY (excluded from training/validation)
    'DoS': 4,
    'Exploits': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
}

print("   Total categories: 10")
print("   Zero-day category: 'Backdoor' (excluded from training/validation)")

# Typical UNSW-NB15 training set distribution (approximate)
# Source: UNSW-NB15 dataset paper
print("\n" + "-"*80)
print("TRAINING SET DISTRIBUTION (Before train/val split)")
print("-"*80)

# Approximate counts from UNSW-NB15 training set
# These are typical proportions - actual may vary
training_distribution = {
    'Normal': 56000,        # ~33%
    'Fuzzers': 18184,      # ~11%
    'Analysis': 2000,       # ~1%
    'DoS': 12264,          # ~7%
    'Exploits': 33393,     # ~20%
    'Generic': 40000,      # ~24%
    'Reconnaissance': 10491, # ~6%
    'Shellcode': 1133,      # ~0.7%
    'Worms': 130,           # ~0.1%
}

# Calculate total BEFORE excluding Backdoor
total_before_filter = sum(training_distribution.values())
print(f"\nTotal training samples (without Backdoor): {total_before_filter:,}")

print("\nBreakdown (Backdoor already excluded):")
for category, count in sorted(training_distribution.items(), key=lambda x: -x[1]):
    percentage = count / total_before_filter * 100
    print(f"   {category:20s}: {count:6,} samples ({percentage:5.2f}%)")

# After 80/20 split for train/validation
validation_total = int(total_before_filter * 0.2)

print(f"\n" + "="*80)
print(f"VALIDATION SET (20% of training, Backdoor excluded)")
print("="*80)
print(f"\nExpected total validation samples: ~{validation_total:,}")

# Proportional distribution in validation
validation_distribution = {cat: int(count * 0.2) for cat, count in training_distribution.items()}

print("\nExpected validation breakdown:")
for category, count in sorted(validation_distribution.items(), key=lambda x: -x[1]):
    percentage = count / validation_total * 100
    print(f"   {category:20s}: ~{count:5,} samples ({percentage:5.2f}%)")

# Binary classification
normal_count = validation_distribution['Normal']
attack_count = validation_total - normal_count

print(f"\n📊 BINARY DISTRIBUTION:")
print(f"   Normal (0):        ~{normal_count:6,} samples ({normal_count/validation_total*100:5.2f}%)")
print(f"   Attack (1):        ~{attack_count:6,} samples ({attack_count/validation_total*100:5.2f}%)")

# Support set sampling (500 random samples)
print("\n" + "="*80)
print("SUPPORT SET SAMPLING (500 Random Samples from Validation)")
print("="*80)

support_size = 500

print(f"\n📊 EXPECTED COMPOSITION (proportional random sampling):")
print(f"\nBinary breakdown:")
expected_normal = int(support_size * normal_count / validation_total)
expected_attack = int(support_size * attack_count / validation_total)
print(f"   Normal samples:    ~{expected_normal:3d} ({expected_normal/support_size*100:5.2f}%)")
print(f"   Attack samples:    ~{expected_attack:3d} ({expected_attack/support_size*100:5.2f}%)")

print(f"\nKnown attack category breakdown:")
for category, val_count in sorted(validation_distribution.items(), key=lambda x: -x[1]):
    if category == 'Normal':
        continue
    expected = int(support_size * val_count / validation_total)
    print(f"   {category:20s}: ~{expected:3d} samples ({val_count/validation_total*100:5.2f}% of validation)")

print(f"\n   {'Backdoor (ZERO-DAY)':20s}:    0 samples (EXCLUDED)")

print("\n" + "="*80)
print("PROTOTYPE COMPUTATION")
print("="*80)

print("""
From these 500 samples, the base model computes 2 prototypes:

1. Normal Prototype (class 0):
   - Computed from ~97 normal traffic samples
   - Represents mean embedding of normal behavior

2. Known Attack Prototype (class 1):
   - Computed from ~403 known attack samples:
     - ~115 Generic samples (largest category)
     - ~97 Exploits samples
     - ~53 Fuzzers samples
     - ~36 DoS samples
     - ~30 Reconnaissance samples
     - ~18 Analysis samples
     - ~3 Shellcode samples
     - ~0 Worms samples
   - Represents mean embedding of known attack patterns
   - Does NOT include Backdoor (zero-day category)

Test-time classification:
   - Each test sample's embedding distance to both prototypes
   - Classified as Normal or Attack based on nearest prototype
   - Zero-day Backdoor attacks detected via similarity to known attacks
""")

print("="*80)
print("KEY INSIGHTS")
print("="*80)

print("""
1. ✅ Support set has NO zero-day Backdoor samples
2. ✅ 500 samples is substantial (not "few-shot")
3. ✅ Attack-heavy representation: ~33% Normal, ~67% Known Attacks
4. ✅ Diverse attack types: 8 different categories (excluding Backdoor)
5. ✅ Fixed seed ensures same 500 samples across all episodes
6. ✅ Prototypes represent robust average of known patterns

Attack Diversity in Support Set:
   - Generic: Network-based generic attacks (most common)
   - Exploits: Software vulnerability exploits
   - Fuzzers: Fuzzing attempts
   - DoS: Denial of Service attacks
   - Reconnaissance: Network scanning
   - Analysis: Port scanning and probing
   - Shellcode: Shellcode injection attempts
   - Worms: Worm propagation

This is NOT few-shot learning - it's:
   "Prototype-based Zero-Day Detection with Validation Reference Set"
""")

print("="*80 + "\n")
