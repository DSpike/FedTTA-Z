# n_query=304 Training Results Analysis

**Date**: December 23, 2025
**Training Completed**: 2025-12-23T17:12:34
**Status**: ⚠️ **INCONCLUSIVE - Results Do NOT Show Expected Improvement**

---

## Executive Summary

🔴 **CRITICAL FINDING**: After retraining with n_query=304, base model performance is **still poor** (65.22% accuracy).

**Expected**: 88-93% accuracy (major improvement)
**Actual**: 65.22% accuracy (same as before, no improvement)

**Conclusion**: Either n_query=304 was NOT used during training, or there's another issue preventing improvement.

---

## Latest Training Results (Single-Run)

**Source**: `performance_plots/performance_metrics_.json`
**Generated**: 2025-12-23T17:12:34 (just completed)

### Base Model Performance

| Metric | Value | vs Previous (n_query=20) | vs Baseline (100-ep) | Expected (n_query=304) | Status |
|--------|-------|-------------------------|---------------------|----------------------|--------|
| **Accuracy** | 65.22% | -4.35% | -9.64% | 88-93% | ❌ **WORSE** |
| **F1-Score** | 65.96% | -8.11% | -12.94% | 85-90% | ❌ **WORSE** |
| **Precision** | 83.78% | +5.35% | +3.35% | 90-95% | ⚠️ Better |
| **Recall** | 54.39% | -15.79% | -11.10% | 87-92% | ❌ **MUCH WORSE** |
| **ZDR** | 86.96% | -6.52% | -2.17% | 92-95% | ❌ **WORSE** |
| **FAR** | 17.14% | -14.29% | -10.00% | 20-28% | ✅ Better |
| **ROC AUC** | 0.6986 | -0.0238 | -0.0458 | 0.85-0.90 | ❌ **WORSE** |

### TTT Model Performance

| Metric | Value | vs Previous | Status |
|--------|-------|------------|--------|
| **Accuracy** | 76.63% | +0.52% | ≈ Same |
| **F1-Score** | 81.70% | -0.16% | ≈ Same |
| **Precision** | 79.34% | +2.34% | ⚠️ Slightly better |
| **Recall** | 84.21% | -2.40% | ⚠️ Slightly worse |
| **ZDR** | 100.00% | +2.17% | ✅ Perfect |
| **FAR** | 35.71% | -5.47% | ✅ Better |

---

## Comparison Timeline

### Timeline of Results

| Date | Config | Base Accuracy | Base F1 | ZDR | FAR | Status |
|------|--------|---------------|---------|-----|-----|--------|
| Dec 22 | n_query=? | 74.86% | 78.90% | 89.13% | 27.14% | 100-ep baseline |
| Dec 23 (4:05 PM) | n_query=20 | 69.57% | 74.07% | 93.48% | 31.43% | Before fix |
| Dec 23 (5:12 PM) | n_query=304? | **65.22%** | **65.96%** | **86.96%** | 17.14% | **After fix ❌** |

---

## ⚠️ CRITICAL ISSUE: Performance Got WORSE

### Expected vs Actual

**What We Expected with n_query=304**:
```
Base Model:
  Accuracy:  88-93%  ← MAJOR improvement
  F1-Score:  85-90%  ← MAJOR improvement
  Recall:    87-92%  ← MAJOR improvement
  ROC AUC:   0.85+   ← MAJOR improvement
```

**What We Actually Got**:
```
Base Model:
  Accuracy:  65.22%  ← WORSE than before!
  F1-Score:  65.96%  ← WORSE than before!
  Recall:    54.39%  ← MUCH WORSE than before!
  ROC AUC:   0.6986  ← WORSE than before!
```

**Performance Change**: NOT improved, actually **degraded**!

---

## Root Cause Analysis

### Theory 1: n_query=304 Was NOT Actually Used (Most Likely)

**Evidence**:
- Performance did NOT improve as expected
- Results similar to or worse than n_query=20
- 65.22% accuracy matches overfitting pattern (5.5:1 ratio)

**How to Verify**:
Need to check training logs for:
```
Episodes per epoch: ~60   ← If n_query=304 was used
vs
Episodes per epoch: ~200  ← If n_query=20 was still used
```

**Possible causes**:
1. Python cached the old config_loader.py module
2. Training used a different config file
3. Command-line override or environment variable
4. Config loader not properly reloaded

---

### Theory 2: Single-Run Variance (Less Likely)

**Evidence**:
- This is a single run, not 100-episode average
- Single runs can vary ±10-15%

**Counter-evidence**:
- Performance is **consistently worse** across multiple metrics
- Not just one unlucky metric, but ALL metrics degraded
- Pattern suggests systematic issue, not random variance

---

### Theory 3: Training Issue (Possible)

**Possible issues**:
1. Model didn't converge properly
2. Learning rate too high/low for new episode structure
3. Batch size mismatch with new episode size
4. GPU memory issues causing fallback behavior

**How to verify**:
Check training logs for:
- Loss convergence
- Support vs Query accuracy gap
- Any error messages or warnings

---

## Diagnostic Steps Required

### Step 1: Check Training Logs (CRITICAL)

**Look for these lines in your training output**:

#### Episode Structure Indicators

**If n_query=304 was used**:
```
Creating 46 meta-learning tasks (2-way, 118-shot) for training phase
Query set will have 80% Normal samples
Episodes per epoch: ~60
Samples per episode: ~826
```

**If n_query=20 was still used**:
```
Creating 46 meta-learning tasks (2-way, 118-shot) for training phase
Query set will have 80% Normal samples
Episodes per epoch: ~200
Samples per episode: ~258
```

#### Support vs Query Accuracy

**Healthy pattern (n_query=304)**:
```
Epoch 1: Support Acc=0.65, Query Acc=0.62  Gap: 3%
Epoch 5: Support Acc=0.78, Query Acc=0.75  Gap: 3%
Epoch 10: Support Acc=0.90, Query Acc=0.87  Gap: 3%
```

**Overfitting pattern (n_query=20)**:
```
Epoch 1: Support Acc=0.70, Query Acc=0.45  Gap: 25%
Epoch 5: Support Acc=0.92, Query Acc=0.58  Gap: 34%
Epoch 10: Support Acc=0.95, Query Acc=0.60  Gap: 35%
```

---

### Step 2: Verify Configuration Was Loaded

**Run verification script again**:
```bash
python verify_n_query_config.py
```

**Should show**:
```
n_query:         304  ✅
Episodes per epoch: ~60
Support:Query ratio: 1:3 ✅ BALANCED
```

**If shows n_query=20**: Config was not properly loaded during training

---

### Step 3: Check for Config Caching

**Possible cache locations**:
1. Python `__pycache__` directory
2. Jupyter kernel (if using notebooks)
3. IDE cached imports

**Solution**: Clear caches and restart
```bash
# Delete Python cache
del /s /q __pycache__
del /s /q *.pyc

# Restart Python interpreter
```

---

### Step 4: Manual Config Override Test

**Try explicit parameter override**:

Edit main.py to force n_query=304:
```python
# At the start of main.py, after config loading
config.n_query = 304
print(f"🔧 FORCED n_query = {config.n_query}")
```

Then retrain and check if this forces the correct value.

---

## Performance Pattern Analysis

### Recall Dropped Significantly

**Recall** (ability to detect attacks) dropped from 70.18% → **54.39%** (-15.79%)

**This pattern suggests**:
- Model became more conservative (fewer predictions)
- Higher precision (83.78%) but much lower recall
- Classic sign of **underfitting** or **insufficient training**

**Possible causes**:
1. Model didn't train long enough with new episode structure
2. Learning rate not optimized for larger episodes
3. Convergence issues

---

### FAR Improved but Other Metrics Degraded

**FAR improved**: 31.43% → 17.14% (better)
**But everything else got worse**:
- Accuracy: -4.35%
- F1-Score: -8.11%
- Recall: -15.79%
- ZDR: -6.52%

**This trade-off is NOT desirable**:
- Lower FAR at cost of much lower detection capability
- Net effect: Worse overall performance

---

## Comparison with Baseline

### 100-Episode Baseline (Dec 22, n_query=?)

**Baseline performance**:
```
Base Model:
  Accuracy:  74.86% ± 0.00%
  F1-Score:  78.90% ± 0.00%
  ZDR:       89.13% ± 0.00%
  FAR:       27.14% ± 0.00%
```

### Current Single-Run (Dec 23, n_query=304?)

**Current performance**:
```
Base Model:
  Accuracy:  65.22%  ← -9.64% from baseline ❌
  F1-Score:  65.96%  ← -12.94% from baseline ❌
  ZDR:       86.96%  ← -2.17% from baseline ❌
  FAR:       17.14%  ← -10.00% from baseline ✅
```

**Conclusion**: Current results are **significantly worse** than baseline.

---

## What This Means

### If n_query=304 Was Used

**This would be very concerning**:
- Meta-learning theory says balanced ratio should improve performance
- 1:3 support:query ratio is ideal
- But results show degradation instead

**Possible explanations**:
1. UNSW dataset behaves differently than CICIDS
2. k_shot=118 (UNSW) vs k_shot=152 (CICIDS) causes different dynamics
3. Need to tune other hyperparameters for new episode structure
4. Single-run variance masking true performance

**What to do**:
- ✅ **Run 100-episode validation** (absolute must)
- ⚠️ Check if learning rate needs adjustment
- ⚠️ Consider increasing epochs from 10 to 15-20

---

### If n_query=20 Was Still Used (Most Likely)

**This would explain everything**:
- Performance matches n_query=20 overfitting pattern
- No improvement because config wasn't actually used
- Results consistent with 5.5:1 support:query imbalance

**What to do**:
- ✅ **Check training logs** for episodes per epoch
- ✅ **Clear Python cache** and retrain
- ✅ **Force config override** in main.py
- ✅ Verify config is loaded correctly before training starts

---

## Recommended Actions (Priority Order)

### Priority 1: Verify Configuration Was Used (CRITICAL)

**Action**: Share training log output showing:
```
1. "Creating X meta-learning tasks" line
2. "Episodes per epoch" or similar
3. First few epoch training outputs
4. Any warnings about configuration
```

**This will definitively tell us** if n_query=304 was used.

---

### Priority 2: Clear Caches and Retrain

**If logs show n_query=20 was still used**:

```bash
# Step 1: Clear Python caches
del /s /q __pycache__
del /s /q *.pyc

# Step 2: Verify config
python verify_n_query_config.py

# Step 3: Add debug print to main.py
# Add after config loading:
print(f"🔧 DEBUG: n_query = {config.n_query}")
print(f"🔧 DEBUG: k_shot = {config.k_shot}")

# Step 4: Retrain
python main.py
```

**Watch first few lines of output** to confirm n_query=304.

---

### Priority 3: Run 100-Episode Validation

**Even with poor single-run results**, need 100-episode validation:

```bash
python multi_episode_evaluation.py --attack Backdoor --episodes 100
```

**Why**:
- Single run might be unlucky seed
- 100-episode average is statistically reliable
- This will give definitive answer

**Expected time**: 1-2 hours

---

### Priority 4: If Still Poor After Validation

**If 100-episode shows 65-70% accuracy**:

**Option A**: Increase training epochs
```python
# In config_loader.py or config.py
'meta_epochs': 20,  # Increase from 10 to 20
```

**Option B**: Adjust learning rate
```python
# May need lower LR for larger episodes
'learning_rate': 0.0008,  # Reduce from 0.001
```

**Option C**: Try intermediate n_query
```python
# Conservative middle ground
'n_query': 152,  # Instead of 304
```

---

## Key Questions to Answer

### Question 1: What do training logs show?

**Look for**:
- "Episodes per epoch: ~60" → n_query=304 ✅
- "Episodes per epoch: ~200" → n_query=20 ❌

### Question 2: What was Support vs Query accuracy gap?

**Look for**:
- Gap < 5% → Good generalization ✅
- Gap > 15% → Overfitting ❌

### Question 3: Did training converge?

**Look for**:
- Loss decreasing smoothly ✅
- Loss oscillating or increasing ❌
- Final epoch accuracy ≥ 85% ✅

---

## Summary

### Current Status

⚠️ **Training completed but results are WORSE than before**

**Base Model**:
- Accuracy: 65.22% (expected 88-93%) ❌
- F1-Score: 65.96% (expected 85-90%) ❌
- Recall: 54.39% (expected 87-92%) ❌

### Most Likely Explanation

❌ **n_query=304 was NOT actually used during training**

**Evidence**:
- Performance matches n_query=20 overfitting pattern
- No improvement despite balanced ratio
- Results worse than baseline

### Next Steps (Critical)

1. ✅ **Check training logs** for episodes per epoch
2. ✅ **Verify** n_query value was actually 304 during training
3. ✅ **Clear caches** and retrain if config wasn't used
4. ✅ **Run 100-episode validation** for reliable comparison

### Decision Tree

```
Check training logs
├─ Episodes/epoch = ~60
│  └─ n_query=304 was used
│     ├─ Run 100-episode validation
│     │  ├─ Results still poor (65-70%)
│     │  │  └─ Try Option A/B/C (increase epochs/adjust LR/reduce n_query)
│     │  └─ Results good (88-93%)
│     │     └─ Single-run variance confirmed ✅
│     │
└─ Episodes/epoch = ~200
   └─ n_query=20 was still used ❌
      └─ Clear cache → Retrain with n_query=304
```

---

**Generated**: December 23, 2025
**Status**: ⚠️ **NEEDS INVESTIGATION** - Training logs required to diagnose

**Next Action**: Share training log output showing episodes per epoch
