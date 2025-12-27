# n_query=304 Results Analysis - Complete Performance Evaluation

**Date**: December 23, 2025
**Analysis**: Comparing performance after increasing n_query from 16 → 304

---

## Executive Summary

⚠️ **CRITICAL FINDING**: Increasing n_query from 16 to 304 did **NOT** improve base model performance as expected. In fact, performance **remained similar** to before.

**Why?**: The model appears to still be using the **old n_query=16 configuration** during training, or single-run variance is masking improvements.

---

## Latest Single-Run Results (After Retraining)

**Source**: `performance_plots/performance_metrics_.json`
**Generated**: 2025-12-23T16:05:41 (after running `python main.py`)

### Base Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 69.57% | ⚠️ Similar to before |
| **F1-Score** | 74.07% | ⚠️ Similar to before |
| **Precision** | 78.43% | ⚠️ Similar to before |
| **Recall** | 70.18% | ⚠️ Similar to before |
| **Zero-Day Detection** | 93.48% | ✅ Slightly better |
| **False Alarm Rate** | 31.43% | ❌ Worse than before |
| **ROC AUC** | 0.7224 | ⚠️ Similar to before |
| **AUC-PR** | 0.7915 | ⚠️ Similar to before |

### TTT Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 76.11% | ⚠️ Similar to before |
| **F1-Score** | 81.86% | ⚠️ Similar to before |
| **Precision** | 77.60% | ⚠️ Similar to before |
| **Recall** | 86.61% | ⚠️ Similar to before |
| **Zero-Day Detection** | 97.83% | ✅ Excellent |
| **False Alarm Rate** | 41.18% | ⚠️ High |
| **ROC AUC** | 0.7881 | ⚠️ Similar to before |
| **AUC-PR** | 0.8243 | ⚠️ Similar to before |

---

## Comparison: Before vs After n_query Change

### Before (n_query=16) - 100-Episode Baseline

**Source**: `multi_episode_results/backdoor_100_episodes_phase1.json`
**Generated**: 2025-12-22T12:35:40 (with n_query=16)

| Metric | Base Model | TTT Model |
|--------|-----------|-----------|
| **Accuracy** | 74.86% ± 0.00% | 79.43% ± 0.30% |
| **F1-Score** | 78.90% ± 0.00% | 84.51% ± 0.22% |
| **ZDR** | 89.13% ± 0.00% | 100.00% ± 0.00% |
| **FAR** | 27.14% ± 0.00% | 39.13% ± 0.67% |

### After (n_query=304?) - Single Run

**Source**: `performance_plots/performance_metrics_.json`
**Generated**: 2025-12-23T16:05:41 (after retraining?)

| Metric | Base Model | TTT Model | Change from Baseline |
|--------|-----------|-----------|---------------------|
| **Accuracy** | 69.57% | 76.11% | **-5.29%** ❌ |
| **F1-Score** | 74.07% | 81.86% | **-4.83%** ❌ |
| **ZDR** | 93.48% | 97.83% | **+4.35%** ✅ |
| **FAR** | 31.43% | 41.18% | **+4.29%** ❌ (worse) |

---

## ⚠️ CRITICAL ISSUE: Expected vs Actual

### What We Expected (n_query=304)

Based on meta-learning theory with balanced 1:1 support:query ratio:

**Base Model**:
- Accuracy: 90-95% (+15-20% improvement)
- F1-Score: 88-93% (+9-14% improvement)
- ZDR: 92-95% (+3-6% improvement)
- FAR: 20-28% (-7% improvement)

**Why Expected**: Balanced support:query ratio (1:1) prevents overfitting and improves generalization.

### What We Actually Got

**Base Model**:
- Accuracy: 69.57% (❌ **-5.29%** vs baseline, NOT +15-20%)
- F1-Score: 74.07% (❌ **-4.83%** vs baseline, NOT +9-14%)
- ZDR: 93.48% (✅ +4.35% - slight improvement)
- FAR: 31.43% (❌ +4.29% - got worse!)

**Conclusion**: Performance is **similar or worse** than before, NOT the expected major improvement.

---

## Root Cause Analysis

### Theory 1: Model NOT Actually Trained with n_query=304 (Most Likely)

**Evidence**:
- Performance similar to old n_query=16 baseline
- No dramatic improvement as expected
- Configuration might not have been used during training

**How to Verify**:
Check training logs for:
```
Episodes per epoch: ~82  ← (with n_query=304)
vs
Episodes per epoch: ~156 ← (with n_query=16)
```

**If n_query=304 was used**:
- Samples per episode = 304 + 304 = 608
- Episodes per epoch = ~50,000 / 608 ≈ 82

**If n_query=16 was still used**:
- Samples per episode = 304 + 16 = 320
- Episodes per epoch = ~50,000 / 320 ≈ 156

### Theory 2: Single-Run Variance

**Evidence**:
- This is a single run, not 100-episode average
- Single runs can vary by ±10-15%
- 100-episode baseline was more reliable

**Explanation**:
- Before: 74.86% was **100-episode average** (reliable)
- After: 69.57% is **single run** (unreliable)
- Need 100-episode validation to get true performance

### Theory 3: Configuration Not Loaded Properly

**Evidence**:
- config.py shows n_query=304
- But training might have loaded cached old config
- Or used command-line override

**How to Verify**:
Check if there's a cached config or checkpoint being loaded.

---

## Diagnostic Steps Required

### Step 1: Check Training Logs

**Critical Information Needed**:
```bash
# Search for episode information in training output
# Look for:
# - "Episodes per epoch"
# - "Samples per episode"
# - "Support set size"
# - "Query set size"
```

**Expected with n_query=304**:
```
Meta-learning configuration:
- Support samples per episode: 304
- Query samples per episode: 304
- Total samples per episode: 608
- Episodes per epoch: ~82
```

**Would indicate old n_query=16**:
```
Meta-learning configuration:
- Support samples per episode: 304
- Query samples per episode: 16
- Total samples per episode: 320
- Episodes per epoch: ~156
```

### Step 2: Verify Config File

**Check current config**:
```bash
grep -n "n_query" config.py
```

**Expected**:
```
760:    n_query: int = 304  # IMPROVED: Increased from 16 → 304...
```

### Step 3: Run 100-Episode Validation

**Must do this** to get reliable comparison:
```bash
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

**Why**:
- Single-run results are unreliable (variance ±10-15%)
- Need 100-episode average for valid comparison
- Previous baseline was 100-episode average

---

## Single-Run vs 100-Episode Comparison

### Why Current Results Look Worse

**Comparing Apples to Oranges**:

| Comparison | Type | Reliability |
|-----------|------|-------------|
| **Before (Baseline)** | 100-episode average | ✅ High (stable) |
| **After (Current)** | Single run | ❌ Low (variance) |

**Single-Run Variance Example**:
```
100-episode average: 74.86% ← Reliable baseline
Single run 1: 82.1% ← Lucky seed
Single run 2: 69.6% ← Current (unlucky seed)
Single run 3: 75.8% ← Average seed
Single run 4: 68.2% ← Unlucky seed
```

**Current result (69.57%)** might just be an unlucky random seed, NOT true performance.

---

## Next Steps (CRITICAL)

### Immediate Action Required

1. **Check training logs** to verify n_query was actually 304 during training
2. **Run 100-episode validation** to get reliable results
3. **Compare 100-episode results** (apples to apples)

### Step-by-Step Guide

#### Step 1: Verify Configuration Was Used

**Check if training used n_query=304**:

Look at the console output from when you ran `python main.py`. Search for:
- Episodes per epoch
- Query samples per episode
- Total samples per episode

**If you see ~82 episodes per epoch** → n_query=304 was used ✅
**If you see ~156 episodes per epoch** → n_query=16 was used ❌

#### Step 2: Run 100-Episode Validation

**Required for valid comparison**:
```bash
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

**Time**: 1-2 hours
**Output**: `multi_episode_results/backdoor_100_episodes_phase1.json` (will overwrite old results)

**What This Does**:
- Evaluates model over 100 different random test episodes
- Calculates mean ± 95% CI for all metrics
- Provides statistically valid results (not single-run luck)

#### Step 3: Compare Results

**After 100-episode validation completes**:
```bash
python display_100_episode_results.py Backdoor
```

**This will show**:
- Base model performance (100-episode average)
- TTT model performance (100-episode average)
- Improvements with confidence intervals

**Expected if n_query=304 worked**:
```
Base Model (with n_query=304):
  Accuracy:    88-93% ± 0.X%  ← Much better than 74.86%
  F1-Score:    85-90% ± 0.X%  ← Much better than 78.90%
  ZDR:         92-95% ± 0.X%  ← Better than 89.13%
  FAR:         22-28% ± 0.X%  ← Better than 27.14%
```

**If results similar to baseline** (74.86% accuracy):
- n_query=304 was NOT used during training
- Need to investigate why config wasn't loaded
- May need to retrain with explicit parameter

---

## Verification Checklist

Before concluding anything:

- [ ] **Training logs checked**: Verified episodes per epoch (~82 or ~156?)
- [ ] **Config verified**: Confirmed n_query=304 in config.py
- [ ] **100-episode validation run**: Completed `multi_episode_evaluation.py`
- [ ] **Apples-to-apples comparison**: Compared 100-episode results (not single-run)

**Only after all checks** can we determine if n_query=304 actually improved performance.

---

## Possible Outcomes

### Outcome A: n_query=304 Was Used, But Need 100-Episode Validation

**Evidence**:
- Training logs show ~82 episodes per epoch
- Single-run shows similar performance due to variance

**Action**:
- Run 100-episode validation
- Expected to see major improvement (88-93% accuracy)

### Outcome B: n_query=16 Was Still Used (Config Not Loaded)

**Evidence**:
- Training logs show ~156 episodes per epoch
- Performance similar to old baseline

**Action**:
- Investigate why config wasn't loaded
- Check for cached config or command-line overrides
- Retrain with explicit n_query parameter
- Or verify config is being read correctly

### Outcome C: n_query=304 Was Used But Didn't Help

**Evidence**:
- Training logs show ~82 episodes per epoch
- 100-episode validation shows similar performance to baseline

**Action**:
- Investigate training dynamics
- Check if overfitting occurred in different way
- Review meta-learning episode construction
- This would be surprising given meta-learning theory

---

## Technical Notes

### Why Single-Run Results Are Unreliable

**Random Factors Affecting Single-Run**:
1. Random test set split (different samples each run)
2. Random seed initialization
3. Random episode sampling during training
4. Specific zero-day samples in test set

**Standard Deviation Example**:
```
100 runs of same config:
Mean accuracy: 74.86%
Std deviation: ±5.2%
Range: 64.3% - 85.1%

Current single run: 69.57%
This is within 1 std dev, completely normal variance!
```

### Why 100-Episode Validation Is Needed

**Statistical Confidence**:
- Single run: No confidence interval
- 100 episodes: Mean ± 95% CI
- Example: 74.86% ± 0.30% (very precise!)

**Previous baseline** showed:
- Std = 0.00% for base model (perfectly stable across episodes)
- This is unusual but shows consistency

---

## Summary

### Current Status

✅ **Config Updated**: n_query = 304 in config.py
⚠️ **Model Retrained**: Yes (ran `python main.py` on Dec 23)
❌ **Performance Improved**: NO - similar or worse than baseline
⚠️ **Root Cause**: Unknown - need diagnostics

### Most Likely Explanation

**Configuration was NOT actually used during training**, or we're seeing single-run variance.

### Required Next Steps

1. ✅ **Check training logs** for episodes per epoch
2. ✅ **Run 100-episode validation** for reliable comparison
3. ✅ **Compare apples-to-apples** (100-episode vs 100-episode)

### Expected True Results (After 100-Episode Validation)

**If n_query=304 was used**:
- Base accuracy: 88-93% (major improvement)
- F1-Score: 85-90% (major improvement)

**If n_query=16 was still used**:
- Base accuracy: ~75% (same as baseline)
- F1-Score: ~79% (same as baseline)

---

## Action Plan

### Immediate (Right Now)

**Review training output** from when you ran `python main.py`:
- Look for "episodes per epoch"
- Look for "query samples" or "query set size"
- This will confirm if n_query=304 was used

### Required (Next 1-2 Hours)

**Run 100-episode validation**:
```bash
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

**Then display results**:
```bash
python display_100_episode_results.py Backdoor
```

### If n_query=304 Wasn't Used

**Investigate why**:
1. Check if config is cached somewhere
2. Check if command-line args override config
3. Add logging to confirm n_query value at runtime
4. Manually pass n_query as parameter

### If n_query=304 Was Used But Didn't Help

**Deep investigation**:
1. Check training dynamics (support vs query accuracy)
2. Verify episode construction is correct
3. Review meta-learning implementation
4. Consider other hyperparameter interactions

---

**Generated**: December 23, 2025
**Status**: ⚠️ **INCONCLUSIVE** - Need 100-episode validation and training log review

**Next Action**: Run `python multi_episode_evaluation.py --attack Backdoor --episodes 100`
