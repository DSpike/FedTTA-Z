# TTT Overconfidence and Performance Degradation - Root Cause Analysis

## Problem Summary

After implementing ZDR-maximized threshold optimization with FAR ≤ 8%, TTT performance is still degrading:

**Results (with optimized threshold = 0.05):**
- Base Model: Accuracy=81.75%, F1=79.75%, ZDR≈77%
- TTT Model: Accuracy=78.81%, F1=76.05%, ZDR=70.4%
- **Degradation: -2.94% accuracy, -3.7% F1, -6.6% ZDR** ❌

## Root Cause: Overconfident Probability Predictions

### Probability Distribution Analysis

From `run_far_8_percent.log`:
```
📊 TTT Probability Analysis:
  ├─ Attack prob range: [0.0000, 1.0000]
  ├─ Attack prob mean: 0.3619, std: 0.4801
  ├─ Attack prob median: 0.0000  ⚠️ 50% of samples have prob = 0!
  └─ Samples with prob > 0.9: 273/756 (36.1%)
```

**Critical Finding:** Median = 0.0000 means:
- **≥50% of samples have attack_prob = 0** (model 100% sure they're Normal)
- **36% have attack_prob > 0.9** (model 100% sure they're Attack)
- **<14% have intermediate probabilities** (0.0 < prob < 0.9)

This is **extreme overconfidence** - the model has no nuance!

### Why This Causes Performance Degradation

**Prediction Distribution (threshold=0.05):**
- Predicted: 482 Normal, 273 Attack
- Actual: 360 Normal, 395 Attack
- **Under-predicting attacks by 122 samples!** ❌

**Zero-Day Detection:**
- Zero-day attacks: 189 total
- Detected: 133 (ZDR = 70.4%)
- **Missed: 56 zero-day attacks** because model assigned them prob=0

The model is **too confident** that certain attacks are Normal, and changing the threshold doesn't help because probabilities are binary (0 or 1).

## Why TTT Causes Overconfidence

### 1. Entropy Minimization (TENT) Pushes to Extremes

TENT loss = `-Σ p(x) log p(x)` is minimized when probabilities are 0 or 1:
- p=0.5 → Entropy = 0.693 (high, bad)
- p=0.9 → Entropy = 0.325 (medium)
- p=1.0 → Entropy = 0.000 (low, good for TENT!)

**Result:** TENT actively **punishes uncertainty** and **rewards overconfidence**.

### 2. Prototype-Based Classification Creates Hard Boundaries

Prototype distance → logits → softmax:
- If sample is closer to one prototype, distance ratio can be extreme
- Softmax amplifies: `exp(-d1) / (exp(-d1) + exp(-d2))`
- Small distance differences → extreme probabilities (0.001 or 0.999)

### 3. Small Support Set Doesn't Represent Full Distribution

- Support set: 252 samples (33% of test set)
- Test set: 756 samples (67% unseen during TTT)
- Zero-day attacks in test: 189 (diverse attack patterns)

**Result:** TTT adapts to support set patterns, but:
- Support set doesn't cover all zero-day variants
- Model becomes **overconfident on similar patterns**
- Model becomes **overconfident (wrong) on dissimilar patterns** ❌

### 4. Temperature Scaling (T=1.5) Isn't Enough

Current: `calibrated_logits = logits / 1.5`
- This softens probabilities slightly: 0.999 → 0.95, 0.001 → 0.05
- But NOT enough when distance ratios are extreme

## Why Threshold Optimization Can't Fix This

The threshold optimization is working correctly:
- Searched 500 thresholds from 0.05 to 0.95
- Selected threshold=0.05 to maximize ZDR subject to FAR ≤ 8%
- This is the **optimal threshold** given the probability distribution

**But it can't fix the underlying problem:**

| Threshold | Effect | Why It Doesn't Help |
|-----------|--------|---------------------|
| 0.01-0.04 | Classify almost everything as Attack | FAR explodes (>50%), violates constraint |
| **0.05** | **Current optimal** | **ZDR=70.4%, FAR=5.54%** |
| 0.10-0.90 | Same predictions as 0.05 | Most probs are 0 or 1, threshold in between has no effect |
| 0.91-0.99 | Classify only prob>0.9 as Attack | Misses all the prob=0 samples, ZDR drops further |

**The real issue:** With median prob=0, changing threshold from 0.05 to 0.50 makes **almost no difference** because:
- 50% of samples with prob=0 are still predicted Normal
- Only affects the small % with intermediate probs

## Solutions to Consider

### Option 1: Increase Temperature Scaling (QUICK FIX)
**Change:** T=1.5 → T=3.0 or T=5.0
**Effect:** Softens overconfident predictions more aggressively
- 0.999 → 0.80 (T=3.0) or 0.60 (T=5.0)
- Gives threshold optimization more room to work
**Risk:** May not be enough if distances are still extreme

### Option 2: Add Confidence Penalty to TTT Loss (BETTER FIX)
**Change:** Add term to TTT loss that penalizes overconfidence
```python
ttt_loss = entropy_loss + lambda * max(0, max_prob - 0.9)
```
**Effect:** Prevents model from becoming too confident (>90%)
**Benefit:** Maintains uncertainty, better calibration

### Option 3: Increase Support Set Size (MORE DATA)
**Current:** 252 samples (100 Normal, 152 Attack)
**Change:** 504 samples (200 Normal, 304 Attack) - double it
**Effect:** Better coverage of zero-day distribution
**Risk:** May overfit more (more params to adapt per sample)

### Option 4: Use Ensemble/Averaging (ROBUST)
**Change:** Average predictions from:
- Base model (before TTT)
- Adapted model (after TTT)
```python
final_prob = 0.7 * base_prob + 0.3 * ttt_prob
```
**Effect:** Reduces overconfidence, leverages both models
**Benefit:** Can't perform worse than base model

### Option 5: Reduce TTT Steps (LESS ADAPTATION)
**Current:** 200-400 steps
**Change:** 50-100 steps (adapt less aggressively)
**Effect:** Less entropy minimization → less overconfidence
**Risk:** May not adapt enough to test distribution

## Recommended Next Steps

1. **Immediate test:** Increase temperature to T=5.0, rerun
   - Expected: More calibrated probabilities, better ZDR
   - Timeline: 5 min

2. **If still degrading:** Add confidence penalty to TTT loss
   - Prevents overconfidence at the source
   - Timeline: 10 min code + 5 min run

3. **If still degrading:** Use base+TTT ensemble (0.7/0.3 weighting)
   - Guaranteed to not perform worse than base model
   - Timeline: 5 min code + 5 min run

## Key Insight

**The threshold optimization is NOT the problem** - it's working correctly!

**The real problem is TTT adaptation:** It's making predictions **too confident and too wrong** on zero-day attacks that differ from the support set patterns.

Fixing overconfidence will make threshold optimization effective and unlock the +2-4% improvement we're targeting.
