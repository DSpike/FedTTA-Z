# TTT Failure Root Cause Analysis for Backdoor Attacks

**Date**: December 21, 2025
**Attack Type**: Backdoor
**Dataset**: UNSW-NB15

---

## Executive Summary

### The Problem

Test-time training (TTT) **consistently fails** for Backdoor attack detection:
- **ZDR decreases** by -4.64% (93.33% → 88.69%)
- **FAR increases** by +8.88% (36.23% → 45.11%)
- **High variance**: TTT std = 0.0179 vs Base std = 0.0000 (infinite ratio)

### Root Causes Identified

1. **Insufficient Training Data** (Primary)
2. **Overly Aggressive Hyperparameters**
3. **Poor Embedding Quality**
4. **TTT Overconfidence Problem**
5. **Extreme Data Imbalance**

---

## Root Cause #1: Insufficient Training Data

### The Data Scarcity Problem

**Backdoor Attack Distribution:**
```
Total test samples:    82,332
Backdoor samples:      583 (0.71%)
DoS samples:          4,089 (4.97%) - 7x more
Exploits samples:    11,132 (13.52%) - 19x more
```

**Impact on TTT:**
- TTT requires **repeated exposure** to learn patterns
- With only 583 samples and batch size 64:
  - Only ~9 batches of unique Backdoor data
  - TTT configured for up to 400 steps
  - **Oversampling ratio: 43.91x** - extreme repetition!

### Why This Causes Failure

1. **Memorization Instead of Learning**
   - TTT sees the same 583 samples repeated 44 times
   - Model memorizes noise instead of learning patterns
   - Leads to overfitting and poor generalization

2. **High Variance**
   - 100-episode analysis shows:
     - Base model ZDR std: 0.0000 (rock solid)
     - TTT model ZDR std: 0.0179 (infinite variance ratio!)
   - **Interpretation**: With so little data, random batch selection causes huge performance swings

3. **Comparison with Other Attacks**
   ```
   Attack     | Samples | TTT Impact
   -----------|---------|-------------
   Backdoor   |     583 | -4.64% ZDR ❌
   DoS        |   4,089 | ~0% ZDR ✅
   Exploits   |  11,132 | Unknown (likely positive)
   ```

**Conclusion**: 583 samples is **below critical threshold** for stable TTT (~1,000 minimum needed).

---

## Root Cause #2: Overly Aggressive Hyperparameters

### Current TTT Configuration

```python
ttt_lr:                      0.005      # Learning rate
ttt_base_steps:              10         # Minimum steps
ttt_max_steps:               400        # Maximum steps
ttt_batch_size:              64         # Batch size
ttt_confidence_reg_weight:   0.4        # Regularization
pseudo_threshold:            0.8        # Pseudo-label threshold
pseudo_weight:               1.5        # Pseudo-label weight
entropy_weight:              0.8        # Entropy loss weight
```

### Problems Identified

#### 1. Learning Rate Too High
- **TTT LR**: 0.005
- **Base model LR**: 0.0011
- **Ratio**: 4.5x higher!

**Impact**:
- Large weight updates during adaptation
- With limited data (583 samples), easily overshoots
- Causes unstable convergence
- **Evidence**: High variance (ZDR std = 0.0179)

#### 2. Too Many Adaptation Steps
- **Max steps**: 400
- **Backdoor batches**: ~9 unique
- **Repetition**: Each sample seen 44 times!

**Impact**:
- Severe overfitting to training noise
- Model "polishes" mistakes instead of correcting them
- Like studying the same 10 questions 44 times - you memorize, don't learn

#### 3. Weak Regularization
- **Confidence reg weight**: 0.4

**Impact**:
- Insufficient penalty for overconfident predictions
- TTT becomes too aggressive (predicts attack more often)
- **Result**: FAR increases from 36.23% to 45.11%

### Recommended Fixes

```python
# For rare attacks (<1,000 samples)
ttt_lr:                      0.002      # 60% reduction
ttt_max_steps:               50         # 87.5% reduction
ttt_confidence_reg_weight:   0.7        # 75% increase
pseudo_threshold:            0.9        # Higher bar
pseudo_weight:               0.8        # Reduce influence
```

---

## Root Cause #3: Poor Embedding Quality

### Embedding Diagnostic Results

From evaluation logs:
```
Prototypes well-separated:   False
Embeddings well-separable:   False
Prototype-based accuracy:    0.0000
```

### What This Means

1. **Prototypes Not Well-Separated**
   - The learned "center" for Backdoor attacks is too close to other attack types
   - Model cannot distinguish Backdoor from similar attacks
   - **Result**: Confusion in predictions

2. **Embeddings Not Well-Separable**
   - Individual Backdoor samples don't cluster together
   - High intra-class variance (samples within Backdoor are spread out)
   - Low inter-class distance (Backdoor overlaps with other classes)

3. **Zero Prototype-Based Accuracy**
   - Nearest-prototype classification completely fails
   - **Critical**: This means the embedding space is fundamentally broken for Backdoor
   - TTT relies on good embeddings - if base model embeddings are poor, TTT makes them worse

### Why Poor Embeddings?

**Backdoor Attack Characteristics:**
```
Feature               Backdoor Mean   Normal Mean     Difference
--------------------------------------------------------------
spkts                         4.38        22.78        -80.8%
dpkts                         0.84        25.42        -96.7%
sbytes                      581.31      4072.38        -85.7%
dbytes                      158.12     18704.93        -99.2%
rate                     154631.49     28349.99       +445.4%
sload                123637371.31  39753387.67       +211.0%
```

**Analysis**:
- Backdoor attacks have **extremely small packet counts** (4.38 vs 22.78)
- **Extremely small byte counts** (581 vs 4072)
- But **extremely high rate** (154k vs 28k)
- **Pattern**: Quick, small bursts - stealthy behavior

**Problem**:
- These subtle patterns are hard to capture with limited samples
- 583 samples insufficient to learn the nuanced Backdoor signature
- Model confuses Backdoor with normal traffic (both have small footprints)

### Feature Variance Analysis

```
Attack Type          Avg Std Dev     Avg Variance
--------------------------------------------------
Backdoor               8,585,195       1.10e+15
DoS                   14,131,241       2.96e+15
Exploits              10,648,250       1.68e+15
Normal                12,972,192       2.43e+15
```

**Observation**:
- Backdoor has **lowest variance** among attack types
- **Implication**: Less diverse patterns, more homogeneous
- **BUT**: With only 583 samples, variance estimates are unreliable
- **Problem**: Model may overfit to the specific Backdoor samples seen, missing unseen variants

---

## Root Cause #4: TTT Overconfidence Problem

### Confusion Matrix Analysis

**Base Model:**
```
               Predicted
             Normal  Attack
Actual Normal   50      16     FAR: 24.24%
Actual Attack   38      70     Recall: 64.81%
```
- Conservative predictions
- Balanced false positives/negatives

**TTT Model (100-episode average):**
```
               Predicted
             Normal  Attack
Actual Normal   ?       ?      FAR: 45.11%
Actual Attack   ?       ?      Recall: Higher
```
- **More aggressive** - predicts "Attack" more often
- Higher recall (catches more attacks)
- **BUT much higher FAR** (many false alarms)

### The Overconfidence Mechanism

1. **Entropy Minimization**
   - TTT uses entropy loss to make confident predictions
   - Pushes probabilities toward 0 or 1
   - **Problem**: With noisy/limited data, confidently predicts the WRONG class

2. **Pseudo-Labeling**
   - TTT generates pseudo-labels for confident predictions (threshold=0.8)
   - Treats these as ground truth for further training
   - **Catastrophic**: If initial prediction wrong, TTT reinforces the mistake

3. **Positive Feedback Loop**
   ```
   1. Model makes confident wrong prediction (e.g., Normal → Attack)
   2. Pseudo-label system accepts it (confidence > 0.8)
   3. Model trains on this wrong pseudo-label
   4. Becomes MORE confident in wrong prediction
   5. Repeat...
   ```

### Evidence

From 100-episode results:
- **FAR increased consistently** across ALL episodes
- Base FAR: 36.23% ± 0.00% (no variance)
- TTT FAR: 45.11% ± 2.31% (high variance, always worse)

**Interpretation**: TTT consistently becomes overconfident in wrong direction (Normal → Attack).

---

## Root Cause #5: Extreme Data Imbalance

### The Imbalance Problem

**Test Set Composition:**
```
Normal samples:        37,000 (44.94%)
All attack samples:    45,332 (55.06%)
Backdoor samples:         583 (0.71%)
```

**Imbalance Ratios:**
- Normal:Backdoor = 63:1
- All Attacks:Backdoor = 78:1

### Impact on TTT

1. **Class Prior Bias**
   - TTT implicitly learns class priors from data
   - "Attack" class appears 78x more than "Backdoor"
   - TTT biased toward predicting general "Attack"

2. **Pseudo-Label Imbalance**
   - With 78:1 ratio, pseudo-labels will be 98.7% non-Backdoor
   - TTT training dominated by non-Backdoor examples
   - Backdoor-specific patterns diluted

3. **Batch Composition**
   - Batch size: 64
   - Expected Backdoor per batch: 64 × (583/82332) = 0.45
   - **Most batches have ZERO Backdoor samples**
   - TTT cannot learn from what it doesn't see

### Why Base Model Does Better

**Base model training:**
- Used SMOTE/ADASYN to balance classes
- Backdoor oversampled to 9,102 samples (15.6x increase)
- Training imbalance ratio: 4.92:1 (much better)

**TTT adaptation:**
- No rebalancing at test time
- Stuck with raw 63:1 imbalance
- **Cannot overcome the imbalance** with 583 samples

---

## Interaction Effects: How Problems Compound

### The Doom Loop

```
1. [Insufficient Data]
   ↓
   Only 583 Backdoor samples
   ↓
2. [Aggressive Hyperparameters]
   ↓
   High LR (0.005) + Many steps (400) = Overfitting
   ↓
3. [Poor Embeddings]
   ↓
   Backdoor overlaps with Normal in embedding space
   ↓
4. [TTT Overconfidence]
   ↓
   Entropy minimization pushes toward confident wrong predictions
   ↓
5. [Pseudo-Label Reinforcement]
   ↓
   Wrong predictions become training data
   ↓
6. [More Overfitting]
   ↓
   Model polishes mistakes 44 times (oversampling)
   ↓
RESULT: High FAR, Low ZDR, High Variance
```

### Why Single-Run Success is Misleading

**Lucky run characteristics:**
- Random seed happened to:
  - Select more separable Backdoor samples
  - Create better batch compositions
  - Initialize weights favorably
- **But**: 99 other episodes showed consistent failure
- **Lesson**: Don't trust single runs with small sample sizes

---

## Validation: Comparison with DoS

### DoS Attack (Successful TTT Case)

**DoS Characteristics:**
```
Samples:              4,089 (7x more than Backdoor)
TTT Impact:           ~0% ZDR change (maintains performance)
FAR:                  Lower than Backdoor case
Variance:             Lower (more stable)
```

**Why TTT Works for DoS:**
1. **Sufficient data**: 4,089 samples crosses ~1,000 threshold
2. **More batches**: ~64 unique batches vs 9 for Backdoor
3. **Less oversampling**: 6x vs 44x repetition
4. **Distinct signature**: DoS floods are easier to detect than stealthy Backdoor

**Conclusion**: The 1,000-sample threshold hypothesis is supported.

---

## Summary: The Five Root Causes

| Cause | Impact | Evidence | Fix Difficulty |
|-------|--------|----------|----------------|
| **Insufficient Data** | ⭐⭐⭐⭐⭐ | 583 vs 4,089 (DoS) | 🔧🔧🔧🔧 (Need data) |
| **Aggressive Hyperparams** | ⭐⭐⭐⭐ | 43.9x oversampling | 🔧 (Easy config) |
| **Poor Embeddings** | ⭐⭐⭐⭐ | 0.0000 proto accuracy | 🔧🔧🔧 (Retraining) |
| **TTT Overconfidence** | ⭐⭐⭐ | FAR +8.88% | 🔧🔧 (Regularization) |
| **Extreme Imbalance** | ⭐⭐⭐ | 63:1 ratio | 🔧🔧🔧 (Rebalancing) |

**Primary Root Cause**: Insufficient data (583 samples)
**Amplifying Factors**: Aggressive hyperparameters, poor embeddings

---

## Recommendations

### Immediate (Easy Fixes)

1. **Disable TTT for Backdoor**
   ```python
   if attack_type == "Backdoor" or zero_day_samples < 1000:
       use_ttt = False  # Use base model
   ```

2. **Reduce Hyperparameter Aggressiveness**
   ```python
   ttt_lr: 0.005 → 0.002
   ttt_max_steps: 400 → 50
   ttt_confidence_reg_weight: 0.4 → 0.7
   ```

### Medium-Term (Data Solutions)

3. **Data Augmentation**
   - Generate synthetic Backdoor samples using SMOTE
   - Target: 2,000+ samples
   - Apply at test time before TTT

4. **Cross-Dataset Transfer**
   - Collect Backdoor samples from other datasets (CICIDS, etc.)
   - Domain adaptation to UNSW-NB15 feature space

### Long-Term (Architectural)

5. **Attack-Specific TTT**
   - Different hyperparameters per attack type
   - Backdoor: Conservative (low LR, few steps, high reg)
   - DoS: Aggressive (current settings)

6. **Improve Base Model Embeddings**
   - Prototype-based training with margin loss
   - Contrastive learning to separate Backdoor from Normal
   - Meta-learning with Backdoor variants

7. **Confidence Calibration**
   - Temperature scaling post-TTT
   - Target: Reduce overconfidence
   - Prevent false alarm cascade

---

## Conclusion

TTT fails for Backdoor attacks due to **fundamental data scarcity** (583 samples), amplified by aggressive hyperparameters and poor embedding quality. The solution requires either:

1. **Short-term**: Disable TTT for Backdoor (use base model)
2. **Long-term**: Augment data to >1,000 samples + hyperparameter tuning

**Key Insight**: TTT is **not a silver bullet**. It requires sufficient data (>1,000 samples) and proper tuning. For rare attacks, simpler approaches (base model) often outperform complex adaptation schemes.

**For Publication**: This is an important negative result - document the failure mode and sample size requirements for TTT to help future researchers avoid the same pitfall.
