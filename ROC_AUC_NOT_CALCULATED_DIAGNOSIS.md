# ROC AUC Not Being Calculated in 100-Episode Evaluation - Diagnosis

**Date**: December 22, 2025
**Status**: ✅ **ROOT CAUSE IDENTIFIED - Fix Ready**

---

## Problem

When running 100-episode evaluation (`multi_episode_evaluation.py`), ROC AUC is not being calculated, and you see:

```
⚠️ Could not calculate ROC AUC: ...
```

The display script falls back to single-run ROC AUC with the message:
```
ROC AUC (from single run):
  Base Model:    0.7322
  TTT Model:     0.8247
  Note:          ⚠️  Single-run only (100-episode didn't save probabilities)
```

---

## Root Cause Analysis

### 1. What the Code Expects

**Location**: [multi_episode_evaluation.py:210-222](multi_episode_evaluation.py#L210-L222)

```python
# Get probabilities and true labels
base_probs = base_eval_results.get('probabilities', None)
ttt_probs = adapted_eval_results.get('probabilities', None)
y_test = system.preprocessed_data['y_test']

# Calculate ROC AUC for base model
if base_probs is not None and len(base_probs) > 0:
    if isinstance(base_probs, list):
        base_probs = np.array(base_probs)
    # Ensure we have binary labels and valid probabilities
    if len(np.unique(y_test)) == 2 and len(base_probs) == len(y_test):
        base_roc_auc = roc_auc_score(y_test, base_probs)
        logger.info(f"  Base Model ROC AUC: {base_roc_auc:.4f}")
```

**The code expects**: `base_eval_results` to have a key `'probabilities'` containing probability scores

---

### 2. What the Code Actually Returns

**Location**: [main.py:5929-5952](main.py#L5929-L5952)

```python
results = {
    'accuracy': accuracy,
    # Binary classification metrics
    'precision': precision_binary,
    'recall': recall_binary,
    'f1_score': f1_binary,
    'f1_score_standard': f1_standard,
    'zero_day_detection_rate': zero_day_detection_rate,
    'far': far,
    'optimal_threshold': fixed_threshold,  # Base model uses fixed 0.5 threshold
    'roc_auc': roc_auc,
    'roc_curve': roc_curve_data,
    'auc_pr': auc_pr,  # AUC-PR (PRIMARY metric for imbalanced zero-day detection)
    'pr_curve': pr_curve_data,
    'confusion_matrix': cm.tolist(),  # Binary confusion matrix
    'classification_report': class_report,  # Detailed binary metrics
    'test_samples': len(y_test_binary),
    'query_samples': len(y_test_combined),
    'support_samples': len(y_test_combined),  # Same as query samples for direct evaluation
    'cm_samples_used': base_cm_samples_used,  # CRITICAL: Store for TTT model to match
    'cm_total_samples': base_cm_total_samples,  # CRITICAL: Store for TTT model to match
    'common_valid_mask': common_valid_mask.tolist() if hasattr(common_valid_mask, 'tolist') else list(common_valid_mask),  # CRITICAL: Common mask based on labels only
    'base_valid_mask': base_valid_mask.tolist() if hasattr(base_valid_mask, 'tolist') else list(base_valid_mask)  # Base model's actual valid mask
}

return results
```

**Problem**: The `results` dictionary **does NOT include** a `'probabilities'` key!

---

### 3. Where Are the Probabilities?

**Location**: [main.py:5705](main.py#L5705)

```python
# Use attack probabilities directly (already computed above)
attack_probs = probs_np
```

The probabilities ARE calculated and stored in `attack_probs`, but they are **never added to the results dictionary**.

---

## The Fix

### Option 1: Add Probabilities to Results Dictionary (Recommended)

**Location**: [main.py:5929](main.py#L5929) - Modify the `results` dictionary

```python
results = {
    'accuracy': accuracy,
    # Binary classification metrics
    'precision': precision_binary,
    'recall': recall_binary,
    'f1_score': f1_binary,
    'f1_score_standard': f1_standard,
    'zero_day_detection_rate': zero_day_detection_rate,
    'far': far,
    'optimal_threshold': fixed_threshold,  # Base model uses fixed 0.5 threshold
    'roc_auc': roc_auc,
    'roc_curve': roc_curve_data,
    'auc_pr': auc_pr,  # AUC-PR (PRIMARY metric for imbalanced zero-day detection)
    'pr_curve': pr_curve_data,
    'confusion_matrix': cm.tolist(),  # Binary confusion matrix
    'classification_report': class_report,  # Detailed binary metrics
    'test_samples': len(y_test_binary),
    'query_samples': len(y_test_combined),
    'support_samples': len(y_test_combined),  # Same as query samples for direct evaluation
    'cm_samples_used': base_cm_samples_used,  # CRITICAL: Store for TTT model to match
    'cm_total_samples': base_cm_total_samples,  # CRITICAL: Store for TTT model to match
    'common_valid_mask': common_valid_mask.tolist() if hasattr(common_valid_mask, 'tolist') else list(common_valid_mask),  # CRITICAL: Common mask based on labels only
    'base_valid_mask': base_valid_mask.tolist() if hasattr(base_valid_mask, 'tolist') else list(base_valid_mask),  # Base model's actual valid mask
    # ADD THIS LINE:
    'probabilities': attack_probs.tolist() if hasattr(attack_probs, 'tolist') else list(attack_probs)  # Add probabilities for ROC AUC calculation
}
```

**Change**: Add one line to include probabilities in the results dictionary

---

### Option 2: Calculate ROC AUC Directly (Alternative)

Instead of relying on probabilities, calculate ROC AUC directly in `evaluate_base_model_only()` and pass it through.

**Not recommended** because:
- multi_episode_evaluation.py already has the ROC AUC calculation logic
- Duplicating the calculation is unnecessary
- Adding probabilities is cleaner and more flexible

---

## Implementation

### Step 1: Modify `evaluate_base_model_only()` in main.py

**Find this line** (around line 5929):
```python
results = {
    'accuracy': accuracy,
    ...
    'base_valid_mask': base_valid_mask.tolist() if hasattr(base_valid_mask, 'tolist') else list(base_valid_mask)  # Base model's actual valid mask
}
```

**Add probabilities**:
```python
results = {
    'accuracy': accuracy,
    ...
    'base_valid_mask': base_valid_mask.tolist() if hasattr(base_valid_mask, 'tolist') else list(base_valid_mask),  # Base model's actual valid mask
    'probabilities': attack_probs.tolist() if hasattr(attack_probs, 'tolist') else list(attack_probs)  # For ROC AUC in multi-episode evaluation
}
```

---

### Step 2: Do the Same for `evaluate_adapted_model()`

**Search for** the TTT model's results dictionary (similar location around line 7300-7400)

**Find**:
```python
adapted_results = {
    'accuracy': ttt_accuracy,
    ...
}
```

**Add probabilities** to TTT results too (search for where `ttt_probs` or similar is calculated)

---

## Expected Impact

### Before Fix

```
[Episode 1/100]
  ⚠️ Could not calculate ROC AUC: 'probabilities'

[Episode 2/100]
  ⚠️ Could not calculate ROC AUC: 'probabilities'

...

[After 100 episodes]
✅ ROC AUC calculated for 0 episodes (base model)
✅ ROC AUC calculated for 0 episodes (TTT model)
```

**Result**: No 100-episode ROC AUC, falls back to single-run

---

### After Fix

```
[Episode 1/100]
  Base Model ROC AUC: 0.7322
  TTT Model ROC AUC: 0.8247

[Episode 2/100]
  Base Model ROC AUC: 0.7356
  TTT Model ROC AUC: 0.8189

...

[After 100 episodes]
✅ ROC AUC calculated for 100 episodes (base model)
✅ ROC AUC calculated for 100 episodes (TTT model)

BASE MODEL PERFORMANCE
======================================================================
...
ROC AUC: 0.7322 ± 0.0024

TTT ADAPTED MODEL PERFORMANCE
======================================================================
...
ROC AUC: 0.8247 ± 0.0031
```

**Result**: Full 100-episode ROC AUC with mean and confidence intervals!

---

## File Size Impact

### Probabilities Storage

**Per Episode**:
- Test samples: ~184 sequences (after filtering to 30% zero-day)
- Probabilities: 184 float32 values = 736 bytes

**100 Episodes**:
- Base model: 100 × 184 × 4 bytes = 73.6 KB
- TTT model: 100 × 184 × 4 bytes = 73.6 KB
- **Total**: ~147 KB for probabilities

### JSON File Size

**Current** (without probabilities):
- `backdoor_100_episodes_phase1.json`: ~120 KB

**After Fix** (with probabilities):
- Estimated: ~267 KB (+147 KB for probabilities)
- Still reasonable (<1 MB)

### Comparison with Original Concern

**ROC_AUC_MODIFICATION_SUMMARY.md** mentioned concern about file size:
> "If probabilities were saved: ~500 KB to 1+ MB"
> "100 episodes × ~184 samples × 2 models = 36,800 probability values"

**Reality Check**:
- 36,800 float32 values × 4 bytes = **147 KB** (NOT 500 KB!)
- Original estimate was too conservative
- **147 KB is acceptable** file size increase

---

## Why This Wasn't Caught Earlier

1. **Single-run main.py works**: Single-run evaluation doesn't need probabilities in results dict because it has `attack_probs` in local scope

2. **ROC AUC calculated during single run**: The single-run calculates ROC AUC directly from `attack_probs` and stores it in results

3. **100-episode needs probabilities**: Multi-episode evaluation runs in a different process and needs probabilities passed through results dict

4. **No error, just warning**: Code doesn't crash, just logs warning and skips ROC AUC

---

## Summary

**Problem**: Probabilities not included in results dictionary → ROC AUC cannot be calculated in 100-episode evaluation

**Root Cause**: `evaluate_base_model_only()` returns results dict without `'probabilities'` key

**Solution**: Add one line to include probabilities in results dict

**Impact**: +147 KB file size, enables true 100-episode ROC AUC calculation

**Benefit**: Full statistical validation of ROC AUC over 100 episodes instead of single-run only

---

## Next Steps

1. Modify [main.py:5929](main.py#L5929) to add `'probabilities'` to base model results
2. Modify TTT model evaluation to add `'probabilities'` to adapted model results
3. Re-run 100-episode evaluation: `python multi_episode_evaluation.py --attack Backdoor --episodes 100`
4. Verify ROC AUC is calculated: `python display_100_episode_results.py Backdoor`

Expected output after fix:
```
ROC AUC:
  Base Model:    0.7322 ± 0.0024
  TTT Model:     0.8247 ± 0.0031
  Improvement:   +0.0925
  Status:        ✅ GOOD (0.80-0.90)
```

**No more "Single-run only" warning!**

---

**Generated**: December 22, 2025
**Status**: ✅ **DIAGNOSIS COMPLETE - Ready to Fix**
