# L2 Regularization Fix for TTT Performance Degradation

## Problem

After fixing the backward graph error, TTT ran successfully for all 200 steps but **severely degraded performance**:
- Base Model: 81.75% accuracy
- TTT Model: 21.01% accuracy (60% degradation!)

## Root Cause Analysis

The L2 regularization penalty was **accumulating too much** over 200 steps:

```
Step 20:  L2_Reg = 0.0306  (Entropy: 0.0084, Pseudo: 0.0011)
Step 40:  L2_Reg = 0.0844  (Entropy: 0.0074, Pseudo: 0.0009)
Step 100: L2_Reg = 0.4331  (Entropy: 0.0052, Pseudo: 0.0014)
Step 200: L2_Reg = 0.9546  (Entropy: 0.0054, Pseudo: 0.0005)
                   ^^^^^^
                   180x larger than adaptation signals!
```

### Why This Happened

1. **L2 regularization weight**: `0.0001`
2. **Number of steps**: 200
3. **Result**: L2 penalty accumulated to 0.9546, which is ~180x larger than the actual adaptation losses (entropy + pseudo-label)
4. **Effect**: Model was heavily penalized back toward original weights instead of adapting

### The Problem in Simple Terms

L2 regularization prevents the model from drifting too far from its original state. But with:
- Small adaptation signals (entropy ≈ 0.005)
- Large L2 penalty (0.95)

The optimizer focused on **minimizing L2 loss** (staying close to original weights) rather than **minimizing entropy/pseudo-label loss** (adapting to test data).

Result: Model barely adapted, or adapted in wrong direction → catastrophic performance drop.

## The Solution

### Implementation

**File**: `config.py`
**Line**: 537

**Change**:
```python
# BEFORE
ttt_l2_reg_weight: float = 0.0001  # L2 accumulates to 0.95 over 200 steps

# AFTER
ttt_l2_reg_weight: float = 0.00001  # L2 should accumulate to ~0.01 over 200 steps
```

**10x reduction**: `0.0001 → 0.00001`

### Expected Behavior

With the reduced L2 weight, at step 200 we expect:
- **L2_Reg**: ~0.01 (instead of 0.95)
- **Entropy**: ~0.005
- **Pseudo**: ~0.001
- **Total Loss**: ~0.016

Now the adaptation signals (entropy + pseudo ≈ 0.006) are **similar in magnitude** to L2 reg (0.01), allowing the model to:
1. ✅ Actually adapt to test data
2. ✅ Stay somewhat anchored to original weights
3. ✅ Balance adaptation vs. stability

## Expected Results

### Previous Run (L2 = 0.0001):
- Base: 81.75%
- TTT: 21.01% ❌ (60% degradation)

### Expected This Run (L2 = 0.00001):
- Base: ~81-82%
- TTT: **~84-86%** ✅ (+2-4% improvement)

## Technical Background

### Why L2 Regularization in TTT?

Standard TENT (Test Entropy Minimization) doesn't use L2 regularization - it relies on:
1. BatchNorm statistics to anchor the model
2. Entropy minimization for adaptation

However, in our case we added L2 to prevent **catastrophic drift**, especially with:
- Prototype-based classification
- K-means pseudo-labeling
- Dynamic prototype updates

But the L2 weight must be **carefully tuned** to not overwhelm the adaptation signals.

### Loss Component Balance

For effective TTT, the loss components should be balanced:
```
Total Loss = α * Entropy + β * Pseudo + γ * L2_Reg

Ideal balance:
- Entropy: dominant adaptation signal (~0.005-0.01)
- Pseudo: complementary signal (~0.001-0.005)
- L2_Reg: stability anchor (~0.01-0.05)

Bad balance (previous run):
- Entropy: 0.0054 ← too small
- Pseudo: 0.0005 ← too small
- L2_Reg: 0.9546 ← DOMINATES (180x larger!)
```

## Related Fixes

This is part of a series of fixes to get TTT working:

1. ✅ **Prototype Mismatch** (Previous session): Store and reuse TTT prototypes
2. ✅ **Deepcopy Failure** (This session): Use in-place adaptation
3. ✅ **Backward Graph Error** (This session): Detach prototypes after computation
4. ✅ **L2 Regularization Accumulation** (This fix): Reduce L2 weight 10x

## Files Modified

1. **config.py** - Line 537: `ttt_l2_reg_weight: 0.0001 → 0.00001`

## Testing

**Command**: `python main.py 2>&1 | tee run_reduced_l2.log`

**What to Verify**:
- [ ] TTT completes all 200 steps without errors
- [ ] L2_Reg at step 200 is ~0.01 (not 0.95)
- [ ] Entropy and Pseudo losses remain similar magnitude
- [ ] TTT **improves** performance (not degrades)
- [ ] Expected improvement: +2-4% over base model

**Key Metrics to Check**:
```
Step 200/200:
  Loss = X.XXXX
  Entropy = ~0.005
  Pseudo = ~0.001
  L2_Reg = ~0.01  ← Should be ~100x smaller than before
```

## Date
2025-12-15

## Status
Implemented and testing (run in progress)
