# Evaluation Flaws Analysis

## 🔴 Critical Issues Found

### 1. **"Test Samples Evaluated: 0"** - Test Sample Tracking Error

**Location**: `main.py:6175` (final evaluation logging)
**Issue**: The final evaluation shows `Test Samples Evaluated: 0` despite evaluating 332 samples.

**Root Cause**:

```python
# Line 1827: final_results gets test_samples from base_results
'test_samples': base_results.get('test_samples', 0),

# Line 6175: final_evaluation gets test_samples from final_results
logger.info(f"Test Samples Evaluated: {final_evaluation.get('test_samples', 0)}")
```

**Problem**: `base_results` comes from `evaluate_zero_day_detection()` which may not properly set `test_samples` in the returned dictionary, or it's being overwritten somewhere.

**Fix Required**:

- Verify `base_results['test_samples']` is set in `_evaluate_base_model()` method
- Ensure `test_samples` is correctly passed through the evaluation chain
- Add logging to track where `test_samples` gets lost

---

### 2. **PR-Based Threshold Optimization Failed: Shape Mismatch**

**Location**: `main.py:4672-4677` (TTT evaluation threshold optimization)
**Error**: `operands could not be broadcast together with shapes (312,) (313,)`

**Root Cause**:

```python
pr_threshold, pr_auc, pr_precision, pr_recall, pr_thresh = find_optimal_threshold_pr(
    query_y_binary.cpu().numpy(),  # Shape: (N,)
    attack_probabilities.cpu().numpy(),  # Shape: (N+1,?) - MISMATCH!
    method='f1',
    min_recall=0.3
)
```

**Problem**: `query_y_binary` and `attack_probabilities` have different lengths (312 vs 313), indicating:

- Possible filtering of samples that wasn't applied consistently
- Sampling issue that creates length mismatch
- Index misalignment between labels and probabilities

**Fix Required**:

```python
# Add length validation before threshold optimization
assert len(query_y_binary) == len(attack_probabilities), \
    f"Length mismatch: y_true={len(query_y_binary)}, y_scores={len(attack_probabilities)}"

# Or ensure consistent filtering:
common_indices = torch.arange(min(len(query_y_binary), len(attack_probabilities)))
query_y_binary = query_y_binary[common_indices]
attack_probabilities = attack_probabilities[common_indices]
```

---

### 3. **K-Fold CV Error: 2D Array Instead of 1D**

**Location**: `main.py:4254` (k-fold evaluation)
**Error**: `y should be a 1d array, got an array of shape (66, 2) instead`

**Root Cause**:

```python
# Somewhere in k-fold evaluation, labels are being passed as 2D arrays (one-hot encoded?)
# Instead of 1D arrays (class indices)
```

**Problem**: Labels are one-hot encoded (shape `(batch, num_classes)`) instead of class indices (shape `(batch,)`).

**Fix Required**:

```python
# Convert one-hot to class indices before sklearn metrics
if y.ndim > 1 and y.shape[1] > 1:
    y = np.argmax(y, axis=1)  # Convert one-hot to class indices
```

---

### 4. **ZDR Calculation Verification**

**Location**: `main.py:4935-4950` (TTT ZDR calculation)
**Status**: ✅ **CORRECT** (verified)

**Current Implementation**:

```python
zero_day_predictions = ttt_predictions_np[is_zero_day_np]
zero_day_actual = query_y_np[is_zero_day_np]
zero_day_tp = ((zero_day_predictions == 1) & (zero_day_actual == 1)).sum()
zero_day_fn = ((zero_day_predictions == 0) & (zero_day_actual == 1)).sum()
zero_day_detection_rate = zero_day_tp / (zero_day_tp + zero_day_fn) if (zero_day_tp + zero_day_fn) > 0 else 0.0
```

**Analysis**: ✅ Correctly calculates recall on zero-day samples only (TP/(TP+FN))

**Issue**: ZDR = 0.4252 (42.52%) despite AUC-PR = 0.8673 (86.73%)

- **Explanation**: AUC-PR is calculated on ALL test samples (zero-day + non-zero-day)
- **ZDR** is calculated ONLY on zero-day samples (127 samples, 38.3% of test set)
- **Mismatch is expected**: AUC-PR includes non-zero-day samples which may have higher accuracy

---

## 📊 Summary of Issues

| Issue                       | Severity  | Status      | Impact                                |
| --------------------------- | --------- | ----------- | ------------------------------------- |
| Test Samples = 0            | ⚠️ Medium | 🔴 Active   | Misleading logging                    |
| PR Threshold Shape Mismatch | 🔴 High   | 🔴 Active   | Threshold optimization fails silently |
| K-Fold CV 2D Array Error    | 🔴 High   | 🔴 Active   | K-fold evaluation partially broken    |
| ZDR Calculation             | ✅ OK     | ✅ Verified | Correct implementation                |

---

## 🛠️ Recommended Fixes

### Fix 1: Test Sample Tracking

```python
# In _evaluate_base_model(), ensure test_samples is set:
base_results = {
    ...
    'test_samples': len(X_test),  # Explicitly set
    ...
}

# In evaluate_zero_day_detection(), verify:
assert 'test_samples' in base_results, "test_samples missing from base_results"
```

### Fix 2: PR Threshold Shape Validation

```python
# Before find_optimal_threshold_pr():
query_y_np = query_y_binary.cpu().numpy()
attack_prob_np = attack_probabilities.cpu().numpy()

if len(query_y_np) != len(attack_prob_np):
    min_len = min(len(query_y_np), len(attack_prob_np))
    logger.warning(f"Shape mismatch: y_true={len(query_y_np)}, y_scores={len(attack_prob_np)}, using {min_len} samples")
    query_y_np = query_y_np[:min_len]
    attack_prob_np = attack_prob_np[:min_len]
```

### Fix 3: K-Fold CV Label Conversion

```python
# In k-fold evaluation, convert one-hot to class indices:
def ensure_1d_labels(y):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    return y
```

---

## 🔍 Additional Investigation Needed

1. **Why shape mismatch occurs**: Trace through the evaluation pipeline to find where filtering/sampling creates length mismatches
2. **Where test_samples gets lost**: Add debug logging throughout evaluation chain
3. **K-fold CV label format**: Verify how labels are passed to sklearn metrics in k-fold evaluation









