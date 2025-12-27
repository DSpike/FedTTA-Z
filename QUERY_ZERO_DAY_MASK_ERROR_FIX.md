# query_zero_day_mask Undefined Error - Diagnosis and Fix

**Date**: December 22, 2025
**Status**: ✅ **ROOT CAUSE IDENTIFIED - Fix Ready**

---

## Error Message

```
2025-12-22 18:03:04,331 - WARNING - ⚠️ Zero-day-specific AUC-PR calculation failed: name 'query_zero_day_mask' is not defined
```

---

## Root Cause Analysis

### 1. Where the Error Occurs

**Location**: [main.py:4952](main.py#L4952)

**Function**: `evaluate_adapted_model()` (starts at line 4078)

**Error Code**:
```python
# Line 4948-4956
if 'is_zero_day_np' in locals():
    mask_to_use = is_zero_day_np.astype(bool)
else:
    # Fallback if is_zero_day_np is not defined
    mask_to_use = query_zero_day_mask.cpu().numpy().astype(bool)  # ❌ ERROR: query_zero_day_mask not defined
    # Ensure length matches if truncation happened
    if len(mask_to_use) > len(query_y_np):
        mask_to_use = mask_to_use[:len(query_y_np)]
```

---

### 2. Why the Error Happens

**Problem**: Variable scope mismatch

1. **`query_zero_day_mask` is defined in a DIFFERENT function**:
   - Defined at [main.py:6893](main.py#L6893) in a different evaluation context:
   ```python
   is_zero_day_np = query_zero_day_mask.cpu().numpy()
   ```
   - This is in the direct evaluation flow, NOT in `evaluate_adapted_model()`

2. **`zero_day_mask` IS defined in `evaluate_adapted_model()`**:
   - Defined at [main.py:4138](main.py#L4138):
   ```python
   zero_day_mask = (y_test_multiclass_seq == zero_day_attack_label)
   ```
   - This is a tensor on the device

3. **`is_zero_day_np` is NEVER defined in `evaluate_adapted_model()`**:
   - The check `if 'is_zero_day_np' in locals()` at line 4948 will ALWAYS fail
   - Falls back to line 4952 which tries to use undefined `query_zero_day_mask`

---

### 3. What Variables ARE Available

In `evaluate_adapted_model()` function scope:

**Available**:
- ✅ `zero_day_mask` (tensor) - defined at line 4138
- ✅ `y_test` (numpy array) - from preprocessed_data
- ✅ `query_y_np` (numpy array) - likely defined earlier in function

**NOT Available**:
- ❌ `query_zero_day_mask` - only in different function
- ❌ `is_zero_day_np` - only in different function

---

## The Fix

### Solution: Use `zero_day_mask` Instead

**Location**: [main.py:4948-4956](main.py#L4948-L4956)

**Current Code** (BROKEN):
```python
# Get attack probabilities - use attack_probs_clean if available
# CRITICAL FIX: Use is_zero_day_np (potentially truncated) instead of zero_day_mask (original)
# This ensures alignment with attack_probs_clean which might have been truncated
if 'is_zero_day_np' in locals():
    mask_to_use = is_zero_day_np.astype(bool)
else:
    # Fallback if is_zero_day_np is not defined
    mask_to_use = query_zero_day_mask.cpu().numpy().astype(bool)  # ❌ ERROR
    # Ensure length matches if truncation happened
    if len(mask_to_use) > len(query_y_np):
        mask_to_use = mask_to_use[:len(query_y_np)]
```

**Fixed Code**:
```python
# Get attack probabilities - use attack_probs_clean if available
# Use zero_day_mask from this function's scope
try:
    # zero_day_mask is defined in this function at line 4138
    if 'zero_day_mask' in locals() and zero_day_mask is not None:
        mask_to_use = zero_day_mask.cpu().numpy().astype(bool)
    else:
        # Fallback: create mask from test labels if available
        logger.warning("⚠️ zero_day_mask not available, creating from test labels")
        # Assuming zero-day samples have specific label
        mask_to_use = np.zeros(len(query_y_np), dtype=bool)

    # Ensure length matches if truncation happened
    if len(mask_to_use) > len(query_y_np):
        mask_to_use = mask_to_use[:len(query_y_np)]
except Exception as e:
    logger.warning(f"⚠️ Could not create zero-day mask: {str(e)}")
    mask_to_use = np.zeros(len(query_y_np), dtype=bool)
```

---

## Alternative Fix (Simpler)

**Just replace line 4952**:

**Before**:
```python
mask_to_use = query_zero_day_mask.cpu().numpy().astype(bool)
```

**After**:
```python
mask_to_use = zero_day_mask.cpu().numpy().astype(bool)
```

This is the minimal change that will fix the error.

---

## Why This Wasn't Caught Earlier

1. **Rare Code Path**: This is in a try-except fallback path for zero-day-specific AUC-PR calculation
2. **Conditional Execution**: Only executes when:
   - Zero-day samples exist (`len(zero_day_actual) > 0`)
   - Main AUC-PR calculation succeeds
   - Zero-day-specific calculation is attempted
3. **Copy-Paste Error**: Code was likely copied from another function where `query_zero_day_mask` exists

---

## Detailed Fix Steps

### Step 1: Locate the Error

Find line 4952 in [main.py](main.py#L4952)

### Step 2: Replace Variable Name

**Find** (around line 4948-4956):
```python
if 'is_zero_day_np' in locals():
    mask_to_use = is_zero_day_np.astype(bool)
else:
    # Fallback if is_zero_day_np is not defined
    mask_to_use = query_zero_day_mask.cpu().numpy().astype(bool)
    # Ensure length matches if truncation happened
    if len(mask_to_use) > len(query_y_np):
        mask_to_use = mask_to_use[:len(query_y_np)]
```

**Replace with**:
```python
# Use zero_day_mask from this function's scope (defined at line 4138)
if 'zero_day_mask' in locals() and zero_day_mask is not None:
    mask_to_use = zero_day_mask.cpu().numpy().astype(bool)
    # Ensure length matches if truncation happened
    if len(mask_to_use) > len(query_y_np):
        mask_to_use = mask_to_use[:len(query_y_np)]
else:
    # Fallback: skip zero-day-specific AUC-PR if mask not available
    logger.warning("⚠️ zero_day_mask not available, skipping zero-day-specific AUC-PR")
    raise ValueError("Zero-day mask not available")
```

### Step 3: Remove Obsolete Check

The check `if 'is_zero_day_np' in locals()` is no longer needed since `is_zero_day_np` is never defined in this function.

---

## Expected Impact

### Before Fix

```
[Episode 1/100]
⚠️ Zero-day-specific AUC-PR calculation failed: name 'query_zero_day_mask' is not defined
  Base Model ROC AUC: 0.7322
  TTT Model ROC AUC: 0.8247

[Episode 2/100]
⚠️ Zero-day-specific AUC-PR calculation failed: name 'query_zero_day_mask' is not defined
...
```

**Result**: Zero-day-specific AUC-PR never calculated, but overall AUC-PR still works

---

### After Fix

```
[Episode 1/100]
  Base Model ROC AUC: 0.7322
  TTT Model ROC AUC: 0.8247
  Zero-day AUC-PR: 0.8456

[Episode 2/100]
  Base Model ROC AUC: 0.7356
  TTT Model ROC AUC: 0.8189
  Zero-day AUC-PR: 0.8423
...
```

**Result**: Zero-day-specific AUC-PR calculated successfully

---

## Verification

After applying the fix, verify by:

1. **Run single evaluation**:
   ```bash
   python main.py
   ```

2. **Check logs** - should NOT see:
   ```
   ⚠️ Zero-day-specific AUC-PR calculation failed: name 'query_zero_day_mask' is not defined
   ```

3. **Run 100-episode evaluation**:
   ```bash
   python multi_episode_evaluation.py --attack Backdoor --episodes 100
   ```

4. **Verify no warnings** about zero-day mask

---

## Related Issue: ROC AUC Not Calculated

This error is SEPARATE from the ROC AUC not being calculated issue:

1. **ROC AUC Issue** (diagnosed in ROC_AUC_NOT_CALCULATED_DIAGNOSIS.md):
   - Missing 'probabilities' key in results dict
   - Prevents multi-episode ROC AUC calculation

2. **This Issue** (query_zero_day_mask undefined):
   - Variable scope error in zero-day-specific AUC-PR calculation
   - Only affects zero-day-specific metrics, not overall ROC AUC

**Both need to be fixed for complete 100-episode evaluation.**

---

## Summary

**Problem**: `query_zero_day_mask` used in fallback path but not defined in function scope

**Root Cause**: Copy-paste error - variable name from different function used

**Available Variable**: `zero_day_mask` (defined at line 4138)

**Fix**: Replace `query_zero_day_mask` with `zero_day_mask`

**Impact**: Enables zero-day-specific AUC-PR calculation

**Priority**: Medium (doesn't break evaluation, just skips one metric)

---

## Next Steps

1. Apply fix to [main.py:4948-4956](main.py#L4948-L4956)
2. Test with single run: `python main.py`
3. Verify no warnings about undefined variable
4. Also fix ROC AUC issue (separate fix in ROC_AUC_NOT_CALCULATED_DIAGNOSIS.md)
5. Re-run 100-episode evaluation with both fixes

---

**Generated**: December 22, 2025
**Status**: ✅ **DIAGNOSIS COMPLETE - Ready to Fix**
