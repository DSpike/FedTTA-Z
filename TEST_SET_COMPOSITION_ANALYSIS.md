# Test Set Composition Analysis

**Date**: 2025-12-19
**Dataset**: UNSW-NB15
**Zero-Day Attack**: DoS (Leave-One-Out Evaluation)
**Total Test Samples**: 830

---

## Overview

The test set is split into three categories for zero-day detection evaluation:
1. **Normal** - Benign traffic (not attacks)
2. **Known Attacks** - Attack types seen during training (8 types)
3. **Zero-Day (Unknown)** - Attack type held out from training (DoS)

---

## Test Set Composition

```
┌─────────────────────────────────────────────────────────────┐
│                   TEST SET: 830 SAMPLES                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NORMAL                        349 samples (42.0%)          │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                │
│                                                              │
│  KNOWN ATTACKS                 285 samples (34.3%)          │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                        │
│                                                              │
│  ZERO-DAY (DoS)                196 samples (23.6%)          │
│  ████████████████████████                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Total Known (Normal + Known Attacks):  634 samples (76.4%)
Total Unknown (Zero-Day):              196 samples (23.6%)
```

---

## Detailed Breakdown

### Category Distribution

| Category | Samples | Percentage | Attack Types Included |
|----------|---------|------------|----------------------|
| **Normal** | 349 | 42.0% | Benign traffic only |
| **Known Attacks** | 285 | 34.3% | 8 attack types (see below) |
| **Zero-Day (DoS)** | 196 | 23.6% | DoS attacks only |
| **TOTAL** | 830 | 100% | 10 categories total |

### Known vs Unknown Split

| Split | Samples | Percentage | Description |
|-------|---------|------------|-------------|
| **Known** | 634 | 76.4% | Normal + 8 known attack types |
| **Unknown (Zero-Day)** | 196 | 23.6% | DoS attacks (held out) |

---

## UNSW-NB15 Attack Types

### All 10 Categories

| Label | Attack Type | Status | Included In |
|-------|------------|---------|-------------|
| 0 | Normal | Normal | Known (training + testing) |
| 1 | Fuzzers | Known Attack | Known (training + testing) |
| 2 | Analysis | Known Attack | Known (training + testing) |
| 3 | Backdoor | Known Attack | Known (training + testing) |
| 4 | **DoS** | **ZERO-DAY** | **Unknown (testing only)** |
| 5 | Exploits | Known Attack | Known (training + testing) |
| 6 | Generic | Known Attack | Known (training + testing) |
| 7 | Reconnaissance | Known Attack | Known (training + testing) |
| 8 | Shellcode | Known Attack | Known (training + testing) |
| 9 | Worms | Known Attack | Known (training + testing) |

### Known Attack Types (8 types, 285 samples)

The 285 "known attack" samples are distributed across these 8 attack types:

1. **Fuzzers** - Attempts to cause program suspension/crashes by feeding random data
2. **Analysis** - Port scanning, spam, HTML file penetration
3. **Backdoor** - Techniques to bypass normal authentication
4. **Exploits** - Taking advantage of system vulnerabilities
5. **Generic** - Works against block ciphers
6. **Reconnaissance** - Information gathering attacks
7. **Shellcode** - Small piece of code used as payload in exploitation
8. **Worms** - Self-replicating malware

**Note**: The distribution across these 8 types is likely unbalanced (some types may have very few samples).

### Zero-Day Attack Type (1 type, 196 samples)

- **DoS (Denial of Service)** - 196 samples
  - Attempts to make machine/network resource unavailable
  - Volume-based attacks (floods target with traffic)
  - **Never seen during training** (true zero-day scenario)

---

## Implications for Your Results

### 1. Zero-Day Proportion (23.6%)

This is a **healthy zero-day proportion**:
- ✅ Large enough to get reliable statistics (196 samples)
- ✅ Not too large to dominate the test set
- ✅ Realistic for zero-day scenarios (typically 20-30%)

### 2. Class Imbalance

The test set has **moderate class imbalance**:
- Normal: 42.0% (349 samples)
- Known Attacks: 34.3% (285 samples across 8 types)
- Zero-Day: 23.6% (196 samples, single type)

**Impact**:
- Binary classification (attack vs normal) is relatively balanced: 58% attack, 42% normal
- Multi-class is imbalanced: 285 samples divided across 8 known attack types = **~36 samples per type on average**
- This explains why your base model struggles with known attacks (low samples per type)

### 3. Why Known Attack Detection Is Lower (61.43% from previous meta_epochs=40 results)

**Root cause**: The 285 known attack samples are split across 8 different attack types:
- **Average per type**: 285 ÷ 8 = ~36 samples per attack type
- **Some types likely have < 20 samples** (severe data scarcity)
- **Model struggles to learn 8 different patterns from limited data**

**In contrast**:
- Zero-Day (DoS): 196 samples, single type
- **More samples per type** means better detection

### 4. Why DoS Might Be Easier to Detect

DoS attacks have **196 samples** (single type) vs known attacks have **~36 samples per type** (8 types):

**DoS characteristics**:
- Volume-based (high packet count, bytes)
- Extreme feature values
- Consistent attack pattern

**Known attacks** (Fuzzers, Analysis, Backdoor, etc.):
- More subtle patterns
- Diverse behaviors across 8 types
- Limited samples per type

This partially explains your **inverted performance pattern** (zero-day 81% > known 61%).

---

## Critical Question: How Many Samples Per Known Attack Type?

Unfortunately, the current results don't break down the 285 known attack samples by type. This information would be **critical** for understanding your model's performance.

### To Get This Information

You need to analyze the test set pickle file to see the distribution:

```python
import pickle
import numpy as np
from collections import Counter

# Load test set
with open('saved_test_sets/test_set_trial_0.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Get attack type distribution (multiclass labels)
y_multiclass = test_data['test_labels_original']  # or similar key
attack_counts = Counter(y_multiclass)

# Map to attack names
attack_types = {
    0: 'Normal', 1: 'Fuzzers', 2: 'Analysis', 3: 'Backdoor',
    4: 'DoS', 5: 'Exploits', 6: 'Generic', 7: 'Reconnaissance',
    8: 'Shellcode', 9: 'Worms'
}

for label, count in sorted(attack_counts.items()):
    attack_name = attack_types[label]
    status = '(ZERO-DAY)' if label == 4 else ''
    print(f'{attack_name:20s}: {count:4d} samples {status}')
```

**Hypothesis**: Some known attack types have very few samples (< 20), causing poor detection rates.

---

## Comparison with SOTA Papers

### Your Test Set vs SOTA

| Paper | Test Set Size | Zero-Day Proportion | Evaluation Method |
|-------|--------------|---------------------|-------------------|
| **Your Work** | 830 samples | 23.6% (196 samples) | Leave-one-attack-out (DoS) |
| Alshahrani et al. 2024 | ~82,000 samples | Variable per attack | Leave-one-attack-out (all 9 types) |
| Ullah & Mahmoud 2024 | ~175,000 samples | Not zero-day eval | Standard train/test split |

**Key Difference**: SOTA papers use the full UNSW-NB15 test set (~82,000 samples), while you're using a stratified subset (830 samples).

### Why Subset vs Full Test Set?

**Possible reasons** for using 830 samples instead of full test set:
1. **Computational efficiency** - Meta-learning is expensive
2. **Episodic evaluation** - Need specific k-shot + n-query format
3. **Balanced evaluation** - Stratified sampling ensures all types represented

**Pros**:
- ✅ Faster evaluation
- ✅ Controlled class distribution

**Cons**:
- ⚠️ Less statistical power (830 vs 82,000)
- ⚠️ May not represent true distribution
- ⚠️ Harder to compare with SOTA papers

---

## Recommendations

### 1. Analyze Known Attack Type Distribution

**Action**: Run script to show samples per known attack type

**Expected finding**: Some types have < 20 samples, causing poor detection

**Impact**: Explains inverted performance pattern

### 2. Consider Increasing Test Set Size

**Current**: 830 samples (stratified subset)
**Alternative**: Use larger test set (5,000-10,000 samples)

**Benefits**:
- More reliable statistics
- Better comparison with SOTA
- More samples per known attack type

**Cost**: Longer evaluation time

### 3. Report Per-Attack-Type Metrics

**Current**: Aggregated known attack detection (61.43%)
**Better**: Show detection rate for each of the 8 known attack types

**Example**:
```
Known Attack Detection by Type:
  Fuzzers:        65.2% (45/69 samples)
  Analysis:       58.3% (28/48 samples)
  Backdoor:       71.4% (15/21 samples)  ← Low sample count!
  Exploits:       62.1% (38/61 samples)
  Generic:        55.6% (25/45 samples)
  Reconnaissance: 48.2% (14/29 samples)
  Shellcode:      41.7% (10/24 samples)  ← Poor detection!
  Worms:          38.5% (10/26 samples)   ← Poor detection!
```

This would reveal which attack types are hardest to detect.

### 4. Run Comprehensive Zero-Day Evaluation

**As recommended in IMMEDIATE_ACTION_PLAN.md**:
- Test all 9 attack types as zero-day
- Compare DoS (23.6%, 196 samples) with other attack types
- Calculate average ZDR across all types

**Hypothesis**: Attack types with more samples will have higher ZDR.

---

## Summary

### Test Set Composition
- **830 total samples**
- **Normal**: 349 (42.0%)
- **Known Attacks**: 285 (34.3%) across 8 types = ~36 samples per type
- **Zero-Day (DoS)**: 196 (23.6%), single type

### Key Insights

1. **Moderate class imbalance** but reasonable for zero-day evaluation
2. **Known attacks spread thin** across 8 types (likely < 40 samples per type)
3. **Zero-Day (DoS) has more samples** than individual known types (196 vs ~36)
4. This partially explains **inverted performance** (zero-day easier than known)
5. **DoS may be inherently easier** to detect (volume-based, extreme features)

### Next Steps

1. ✅ You now understand the test set composition
2. ❓ Need to analyze: Samples per known attack type (critical!)
3. ❓ Need to run: Comprehensive evaluation (all 9 attacks as zero-day)
4. ❓ Need to compare: How do other attacks compare to DoS?

This analysis confirms that **comprehensive zero-day evaluation (Phase 1) is critical** before making architectural changes.
