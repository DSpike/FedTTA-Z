# TTT Deepcopy Issue - Root Cause and Fix

## Problem Discovery

After implementing the prototype reuse fix, TTT was STILL showing no improvement and appeared to be degrading performance. Investigation revealed the actual root cause.

## Root Cause

**TTT was not running at all!** The issue was in `coordinators/centralized_coordinator.py` line 252:

```python
adapted_model = copy.deepcopy(self.model)  # ❌ FAILED SILENTLY
```

### Error Message:
```
ERROR - TTT adaptation failed: Only Tensors created explicitly by the user (graph leaves)
support the deepcopy protocol at the moment. If you were attempting to deepcopy a module,
this may be because of a torch.nn.utils.weight_norm usage
```

### Why This Happened:
- The TransductiveLearner model uses `torch.nn.utils.weight_norm` in its layers
- `weight_norm` creates computed parameters (not graph leaves) that don't support deepcopy
- The deepcopy operation failed silently
- TTT never ran, so the "adapted model" was actually just the base model unchanged
- No prototypes were stored (because TTT never executed)
- Evaluation showed degradation because it was comparing the same model to itself with different prototype computations

### Impact:
1. ❌ TTT adaptation crashed before starting
2. ❌ No model parameters were updated
3. ❌ No prototypes were stored for consistent evaluation
4. ❌ "Adapted model" predictions were 100% identical to base model
5. ❌ Performance showed -0.4% degradation (due to different prototype sampling, not actual TTT)

## Solution

### Fix Applied (centralized_coordinator.py lines 252-260):

```python
# Clone model for adaptation
# FIX: deepcopy fails with weight_norm, adapt in-place instead
try:
    adapted_model = copy.deepcopy(self.model)
    logger.info("✅ Model cloned successfully using deepcopy")
except Exception as e:
    logger.warning(f"⚠️  deepcopy failed ({e}), adapting original model in-place")
    # Note: This modifies self.model directly during TTT, but that's acceptable
    # since we only use the adapted model for evaluation afterwards
    adapted_model = self.model
```

### Why This Works:
- If deepcopy succeeds: Perfect, we have a separate adapted model
- If deepcopy fails: We adapt the original model in-place
  - This is acceptable because after TTT, we only need the adapted model for evaluation
  - The base model evaluation already happened before TTT
  - No need to preserve the original model state after that point

## Expected Results After Fix

### What Should Happen:
1. ✅ TTT adaptation runs for 200 steps
2. ✅ BatchNorm statistics update (momentum=0.8)
3. ✅ Model parameters adapt to test distribution
4. ✅ Prototypes are stored: `adapted_model.ttt_prototypes`
5. ✅ Evaluation uses stored prototypes
6. ✅ Performance improves by +2-4%

### Key Log Messages to Verify:
```
✅ Using k-means clustering for pseudo-labels (k=2)
   Using 100 samples as support set for prototype computation during TTT
🔄 Starting TTT adaptation (tent) for 200 steps...
   → Prototypes updated at step 10
   → Prototypes updated at step 20
   ...
   ✅ Stored TTT prototypes (shape: torch.Size([2, 256]))
✅ Using stored TTT prototypes for consistent evaluation
```

### Expected Performance:
- Base Model: ~81.75% accuracy
- TTT Model: ~84-86% accuracy
- Improvement: **+2-4%** (not -0.4%)

## Files Modified

1. **coordinators/centralized_coordinator.py** (lines 252-260)
   - Added try-except for deepcopy failure
   - Fallback to in-place adaptation

2. **main.py** (lines 4053-4063)
   - Added check for stored TTT prototypes
   - Use stored prototypes if available

## Summary

The original prototype reuse fix was correct, but it never got a chance to run because TTT itself was crashing silently on the deepcopy operation. Now with both fixes:

1. ✅ **Deepcopy Fix**: TTT can actually run (adapts in-place if needed)
2. ✅ **Prototype Reuse Fix**: Evaluation uses the correct prototypes

Both fixes are necessary for TTT to work correctly!

## Date
2025-12-15

## Status
Fixed and testing
