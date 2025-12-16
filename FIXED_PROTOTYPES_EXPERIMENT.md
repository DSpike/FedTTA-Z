# Fixed Prototypes Experiment

## Hypothesis

**Problem**: TTT with prototype updates shows -1.37% degradation instead of +2-4% improvement.

**Hypothesis**: Prototype updates during TTT introduce noise via k-means clustering, which degrades performance.

**Solution**: Disable prototype updates, keep only the initial anchor-based prototypes from training data.

## Rationale

### Why Updates May Hurt

**During TTT adaptation**:
1. **Step 0**: Initialize prototypes using anchor-based assignment from training data ✓ (GOOD)
2. **Step 10**: Update prototypes using k-means on test embeddings (MAY BE BAD)
3. **Step 20**: Update prototypes using k-means on test embeddings (MAY BE BAD)
4. ... repeat every 10 steps

**The problem with updates**:
- K-means on test data may produce noisy pseudo-labels
- Noisy labels → wrong prototypes → wrong classifications
- Initial anchors from training data are more trustworthy

### Evidence from Previous Run

**With prototype updates (every 10 steps)**:
```
Logs show:
  → Prototypes updated using k-means (step 10)
  → Prototypes updated using k-means (step 20)
  ...
  → Prototypes updated using k-means (step 200)

Result: -1.37% degradation
```

**What we're testing now**:
- Keep initial anchor-based prototypes (from training data)
- BatchNorm still adapts over 200 steps
- Prototypes stay fixed (no k-means updates)

## Implementation

**File**: `coordinators/centralized_coordinator.py`
**Line**: 468

**Change**:
```python
# BEFORE:
prototype_update_interval = 10  # Recompute every 10 steps

# AFTER:
prototype_update_interval = 999999  # DISABLED - keep initial anchor-based prototypes
```

**Effect**:
```python
if use_prototype_based and (step + 1) % prototype_update_interval == 0:
    # This will NEVER trigger (step < 200, interval = 999999)
    # Prototypes stay fixed at initial values
```

## Expected Behavior

### What Should Happen

**TTT Adaptation**:
1. ✅ Initialize prototypes with anchor-based assignment (step 0)
2. ✅ Adapt BatchNorm parameters (all 200 steps)
3. ❌ **NO** prototype updates (stay fixed)
4. ✅ Use same prototypes for classification throughout

**Logs Should Show**:
```
✅ Generated anchor prototypes from training data for KMeans alignment
✅ Using Nearest Anchor assignment (preserves class imbalance)
   Initial prototypes shape: torch.Size([2, 256]), num_classes: 2
   Class distribution: [69, 31]

[NO "Prototypes updated" messages during TTT steps]

TTT Step 10/200: ...
TTT Step 20/200: ...
...
TTT Step 200/200: ...
```

### What Should NOT Happen

**Should NOT see**:
```
→ Prototypes updated using k-means (step 10)  ← Should be GONE
→ Prototypes updated at step 10 (num_classes: 2)  ← Should be GONE
```

## Predicted Outcomes

### Best Case ✅
- Removing noisy updates fixes the -1.37% degradation
- TTT shows positive improvement: +1% to +3%
- Closer to target +2-4% improvement

### Moderate Case ⚠️
- Performance stays similar: -1.37% → -0.5% or +0.5%
- Small improvement but not reaching target
- Indicates other factors limiting TTT

### Worst Case ❌
- Performance gets worse: -1.37% → -2% or more
- Suggests prototype updates were actually helping
- Need to re-enable updates with improvements

## Comparison Matrix

| Configuration | Initialization | Updates | Result |
|--------------|----------------|---------|--------|
| **Run 1** | Basic k-means | Every 10 steps | **-60%** (catastrophic) |
| **Run 2** | Anchor-based | Every 10 steps | **-1.37%** (minor degradation) |
| **Run 3** | Anchor-based | **DISABLED** | **?** (testing now) |

**Goal**: Reach positive improvement (+2-4%)

## Technical Details

### What Still Adapts

**With fixed prototypes, TTT still adapts**:
1. ✅ BatchNorm scale parameters (γ)
2. ✅ BatchNorm shift parameters (β)
3. ✅ BatchNorm running statistics (mean, variance)

**Total**: 896 parameters still being updated

### What Stays Fixed

**With fixed prototypes**:
1. ❌ Prototype positions (stay at initial anchor values)
2. ❌ TCN weights (always frozen in TENT)
3. ❌ Projection layers (always frozen in TENT)

### Why This Might Work

**Theory**:
- **Initial prototypes** from training data anchors are high-quality
- **BatchNorm adaptation** helps embeddings match test distribution
- **Fixed prototypes** provide stable reference points
- Together: Embeddings adapt, prototypes stay trustworthy

**Analogy**:
```
Training:
  Embeddings: Learn where to place samples
  Prototypes: Learn where class centers are

TTT (with updates):
  Embeddings: Shift slightly (BatchNorm)
  Prototypes: Also shift (k-means)
  Problem: Both shifting → unstable

TTT (fixed prototypes):
  Embeddings: Shift slightly (BatchNorm)
  Prototypes: Stay fixed (trustworthy anchors)
  Benefit: Stable reference points
```

## Alternative Approaches (If This Doesn't Work)

If fixed prototypes don't help, we can try:

### Plan B: Update Less Frequently
```python
prototype_update_interval = 50  # Update every 50 steps instead of 10
```

### Plan C: Confidence-Based Updates
```python
# Only update prototypes from high-confidence samples
if confidence.mean() > 0.9:
    update_prototypes()
```

### Plan D: Exponential Moving Average
```python
# Blend old and new prototypes
prototypes_new = compute_prototypes(support_x, support_y)
prototypes = 0.9 * prototypes_old + 0.1 * prototypes_new  # EMA
```

### Plan E: Larger Support Set
```python
support_size = 100 → 300  # Use more samples for better estimates
```

## Files Modified

1. `coordinators/centralized_coordinator.py`
   - Line 468: `prototype_update_interval = 10 → 999999`

## Current Run

**Command**: `python main.py 2>&1 | tee run_fixed_prototypes.log`
**Status**: Running
**Expected Duration**: ~2-3 minutes

## Date
2025-12-15

## Status
⏳ Testing in progress
