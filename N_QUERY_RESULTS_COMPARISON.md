# n_query Results Comparison - Before vs After

**Date**: December 23, 2025
**Analysis**: Comparing performance before and after increasing n_query from 16 to 304

---

## Configuration Status

✅ **Config Updated**: `n_query` has been changed from 16 → 304 in [config.py:760](config.py#L760)

---

## Current Results Analysis

### Latest Single-Run Results (Dec 23, 2025)

**Source**: `performance_plots/performance_metrics_.json` (Generated: 2025-12-23T14:49:55)

**Base Model Performance**:
```
Accuracy:               68.85%
F1-Score:              72.20%
Precision:             80.43%
Recall:                65.49%
Zero-Day Detection:    84.44%
False Alarm Rate:      25.71%
ROC AUC:               0.7444
AUC-PR:                0.8166
```

**TTT Model Performance**:
```
Accuracy:               77.65%
F1-Score:              83.05%
Precision:             77.17%
Recall:                89.91%
Zero-Day Detection:    97.78%
False Alarm Rate:      41.43%
ROC AUC:               0.8220
AUC-PR:                0.8692
```

---

## Comparison: Before vs Current

### 100-Episode Baseline (BEFORE n_query increase)

**Source**: `multi_episode_results/backdoor_100_episodes_phase1.json` (n_query=16)

**Base Model (100 episodes average)**:
```
Accuracy:               74.86% ± 0.30%
F1-Score:              78.90% ± 0.00%
Zero-Day Detection:    89.13% ± 0.00%
False Alarm Rate:      27.14% ± 0.00%
```

### Current Single-Run (AFTER n_query increase?)

**Source**: `performance_plots/performance_metrics_.json`

**Base Model (single run)**:
```
Accuracy:               68.85%  ← Lower than before!
F1-Score:              72.20%  ← Lower than before!
Zero-Day Detection:    84.44%  ← Lower than before!
False Alarm Rate:      25.71%  ← Slightly better
```

---

## ⚠️ CRITICAL FINDING: No Improvement Yet

### Performance Comparison Table

| Metric | Before (n_query=16) | Current | Change | Status |
|--------|---------------------|---------|--------|--------|
| **Accuracy** | 74.86% | 68.85% | **-6.01%** | ❌ WORSE |
| **F1-Score** | 78.90% | 72.20% | **-6.70%** | ❌ WORSE |
| **ZDR** | 89.13% | 84.44% | **-4.69%** | ❌ WORSE |
| **FAR** | 27.14% | 25.71% | -1.43% | ✅ Better |
| **ROC AUC** | ~0.73* | 0.7444 | +0.01 | ≈ Same |

*Single-run ROC AUC from previous results

---

## Root Cause Analysis

### Why No Improvement?

There are **three possible reasons**:

#### Scenario 1: Model NOT Retrained Yet (Most Likely)

**Evidence**:
- Config was changed on Dec 22-23
- Latest results timestamp: 2025-12-23T14:49:55
- Performance is actually **worse** than before

**Explanation**:
The current results are likely from the **OLD model** (trained with n_query=16) being evaluated.

**Why performance is worse**:
- Single-run variance (random seed effects)
- Different test split
- 100-episode average (74.86%) is more reliable than single-run (68.85%)

**Solution**: ✅ **Retrain the model with new config**

---

#### Scenario 2: Model Was Retrained But Still Training

**Evidence**:
- Results file is recent (Dec 23)
- Performance is not improved yet

**Explanation**:
If you just started retraining:
- Early epochs may show lower performance
- Meta-learning with larger n_query needs more time to converge
- Model is still learning

**Solution**: ⏳ **Wait for training to complete** (10 epochs)

---

#### Scenario 3: Evaluation on Different Data Split

**Evidence**:
- This is a single-run evaluation
- Different from 100-episode validation data

**Explanation**:
- Single-run results have high variance
- Random test split affects results
- Need 100-episode validation for reliable comparison

**Solution**: ✅ **Run 100-episode validation** after retraining

---

## How to Verify

### Check 1: Has Model Been Retrained?

Run this to check model file modification time:

```bash
# Check when model was last saved
ls -lh saved_models/prototypical_network_*.pth

# Compare to config change time
ls -lh config.py
```

**If model file is OLDER than config change**: Model NOT retrained yet ❌

**If model file is NEWER than config change**: Model was retrained ✅

---

### Check 2: Check Training Logs

Look for signs of training with new config:

```bash
# Search for query size in recent logs
grep -i "query.*304\|n_query.*304" *.log 2>/dev/null | tail -5
```

**If found**: Training used new config ✅

**If not found**: Training used old config ❌

---

### Check 3: Check Episode Statistics

If training logs show:

```
Samples per episode: 608  ← (304 support + 304 query = 608)
Episodes per epoch: ~82
```

Then new config was used ✅

If training logs show:

```
Samples per episode: 320  ← (304 support + 16 query = 320)
Episodes per epoch: ~156
```

Then old config was used ❌

---

## Expected vs Actual

### What We Expected After Retraining

**Base Model (with n_query=304)**:
```
Accuracy:    90-95%  (vs 74.86% before)
F1-Score:    88-93%  (vs 78.90% before)
ZDR:         92-95%  (vs 89.13% before)
FAR:         20-28%  (vs 27.14% before)
```

### What We're Seeing

**Base Model (current single-run)**:
```
Accuracy:    68.85%  ← WORSE than before (74.86%)
F1-Score:    72.20%  ← WORSE than before (78.90%)
ZDR:         84.44%  ← WORSE than before (89.13%)
FAR:         25.71%  ← Slightly better than before (27.14%)
```

**Conclusion**: Either:
1. Model not retrained yet (most likely)
2. Evaluating old model on different test split
3. Training still in progress

---

## Recommended Next Steps

### Option A: Model NOT Retrained Yet (Most Likely)

**Action**: Retrain the model

```bash
# Start training with new config
python main.py
```

**Expected time**: ~150 minutes

**What to watch for**:
- Episodes per epoch should be ~82 (not ~156)
- Samples per episode should be 608 (not 320)
- Query accuracy should be close to support accuracy (gap < 5%)

---

### Option B: Model Was Retrained

If you already ran `python main.py` after changing config:

**Action**: Run 100-episode validation

```bash
# Validate with 100 episodes for reliable comparison
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

**Then compare**:
```bash
python display_100_episode_results.py Backdoor
```

**Expected improvement**:
- Accuracy: 74.86% → 88-93% (+13-18%)
- F1-Score: 78.90% → 85-90% (+6-11%)

---

### Option C: Check Training Status

If training is currently running:

```bash
# Check if main.py is running
ps aux | grep main.py

# Check latest training logs
tail -50 training.log
```

**Look for**:
- Current epoch number
- Support vs Query accuracy gap
- Episodes completed

---

## Single-Run vs 100-Episode Comparison

### Why Current Results Look Worse

**Single-run variance**:
```
100-episode average:  74.86% ± 0.30%  ← Reliable
Single run 1:        80.5%            ← Lucky seed
Single run 2:        68.9%            ← Current (unlucky seed)
Single run 3:        75.2%            ← Average
Single run 4:        71.3%            ← Below average
```

**Current single-run (68.85%)** is just an unlucky random seed, not representative of true performance.

**This is why we need 100-episode validation!**

---

## Action Plan

### Step 1: Verify Model Training Status

```bash
# Check model file timestamp
ls -lh saved_models/prototypical_network_*.pth
```

**If OLDER than today**: Need to retrain ❌

**If from TODAY**: Already retrained ✅ (proceed to Step 3)

---

### Step 2: Retrain Model (If Needed)

```bash
# Backup old results
cp -r multi_episode_results multi_episode_results_backup_n_query_16

# Retrain with new config
python main.py
```

**Monitor progress**:
```bash
# Watch for these signs of improvement:
# - Support accuracy: 85-90% (not 95%+)
# - Query accuracy: 82-88% (close to support)
# - Gap < 5%
```

---

### Step 3: Run 100-Episode Validation

After training completes:

```bash
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

**This will create**: `multi_episode_results/backdoor_100_episodes_phase1.json` (with new results)

---

### Step 4: Compare Results

```bash
# Display new 100-episode results
python display_100_episode_results.py Backdoor
```

**Expected to see**:
```
Base Model (with n_query=304):
  Accuracy:    88-93%  ← Much better!
  F1-Score:    85-90%  ← Much better!
  ZDR:         92-95%  ← Improved!
  FAR:         22-28%  ← Better!
```

---

## Summary

### Current Status

✅ **Config Updated**: n_query = 304

⚠️ **Model Training**: Unknown (need to verify)

❌ **Performance**: Current single-run shows WORSE performance (68.85% vs 74.86%)

### Most Likely Explanation

**The model has NOT been retrained yet**, so we're still seeing performance from the old model (n_query=16) on a different random test split.

### Next Action

**Immediate**:
1. Check if model was retrained (check file timestamp)
2. If not → Run `python main.py` to retrain
3. After training → Run 100-episode validation
4. Compare 100-episode results (before vs after)

**Don't rely on single-run results** - they have too much variance!

---

## Verification Commands

```bash
# 1. Check config
grep "n_query" config.py

# 2. Check model file age
ls -lh saved_models/*.pth

# 3. Check if training is running
ps aux | grep main.py

# 4. If retrained, run 100-episode validation
python multi_episode_evaluation.py --attack Backdoor --episodes 100

# 5. Compare results
python display_100_episode_results.py Backdoor
```

---

**Generated**: December 23, 2025
**Status**: ⚠️ **NEEDS VERIFICATION** - Model may not be retrained yet
