# Fair Comparison: Sample Size Verification

## ✅ **Current Implementation Status**

The system **DOES attempt** to ensure both base and TTT models are evaluated on the same number of samples, but there are **potential issues** that need verification.

---

## 🔍 **Current Implementation**

### **1. Base Model Evaluation**

**Location**: `main.py` `_evaluate_base_model()` (lines 5285-5364)

**Process**:

1. Uses full test set: `X_test`, `y_test` from `preprocessed_data`
2. Applies confidence-based rejection (filters low-confidence predictions)
3. Stores:
   - `base_cm_samples_used`: Number of samples used for confusion matrix
   - `base_valid_mask`: Boolean mask of valid samples
   - `common_valid_mask`: Mask based on labels only (not predictions)

**Code**:

```python
# Confidence-based rejection
confidences, _ = base_probabilities.max(dim=1)
uncertain_mask = confidences < confidence_threshold
base_predictions[uncertain_mask] = -1  # Mark as rejected

# Store valid mask
valid_mask = base_predictions != -1
base_cm_samples_used = valid_mask.sum()
base_valid_mask = valid_mask.copy()
```

---

### **2. TTT Model Evaluation**

**Location**: `main.py` `_evaluate_ttt_model()` (lines 5688-6500)

**Process**:

1. Uses full test set: `X_test`, `y_test` from `preprocessed_data` (same as base)
2. Receives `base_cm_samples_used` and `base_valid_mask` from base model
3. **Attempts to use same samples** via `base_valid_mask`
4. Applies confidence-based rejection (may filter different samples)

**Code**:

```python
# TTT also applies confidence-based rejection
confidences, _ = torch.max(adapted_probabilities, dim=1)
uncertain_mask = confidences_np < confidence_threshold
adapted_predictions_binary[uncertain_mask] = -1

# Try to match base model's sample count
if base_cm_samples_used is not None and ttt_cm_samples_used != base_cm_samples_used:
    # Force TTT to use same samples as base model
    if base_valid_mask is not None:
        # Use base_valid_mask directly
```

---

## ⚠️ **Potential Issues**

### **Issue 1: Confidence-Based Rejection May Filter Different Samples**

**Problem**:

- Base model rejects samples with confidence < threshold
- TTT model also rejects samples with confidence < threshold
- **Different models may have different confidence distributions**
- Result: Different numbers of samples may be rejected

**Example**:

```
Base model: 287/332 samples valid (45 rejected)
TTT model:  304/332 samples valid (28 rejected)
→ Different sample sizes! ⚠️
```

---

### **Issue 2: TTT May Not Be Able to Match Base Samples**

**Problem**:

- If TTT predictions are invalid at `base_valid_mask` indices
- TTT cannot use those exact samples
- Falls back to using available valid TTT samples
- Result: Different sample sets

**Code** (lines 6441-6464):

```python
if len(valid_ttt_indices) >= base_cm_samples_used:
    # Can match
else:
    # Cannot match - uses fewer samples
    logger.warning(f"⚠️ Using {ttt_cm_samples_used} samples (less than base model's {base_cm_samples_used})")
```

---

### **Issue 3: Metrics Calculated on Different Sample Sets**

**Problem**:

- Confusion matrix tries to match sample counts
- But other metrics (accuracy, F1, etc.) may use different samples
- Result: Metrics not directly comparable

---

## ✅ **Recommended Fixes**

### **Fix 1: Use Base Model's Valid Mask for ALL Metrics**

**Current**: Only confusion matrix tries to match samples  
**Fix**: Use `base_valid_mask` for ALL metrics (accuracy, F1, precision, recall, etc.)

**Implementation**:

```python
# In TTT evaluation, use base_valid_mask for ALL metrics
if base_valid_mask is not None:
    # Use base_valid_mask for ALL metric calculations
    ttt_accuracy = accuracy_score(
        query_y_np[base_valid_mask],
        ttt_predictions_np[base_valid_mask]
    )
    # Same for all other metrics
```

---

### **Fix 2: Disable Confidence Rejection for Fair Comparison**

**Option A**: Disable confidence rejection entirely  
**Option B**: Use same confidence threshold and apply to both models identically

**Implementation**:

```python
# Option: Disable confidence rejection for fair comparison
# Or: Use same threshold and apply identically
confidence_threshold = getattr(self.config, 'confidence_rejection_threshold', 0.7)
# Apply to both models with same threshold
```

---

### **Fix 3: Store and Use Exact Sample Indices**

**Current**: Uses boolean masks  
**Fix**: Store exact sample indices and use them for both models

**Implementation**:

```python
# Base model stores indices
base_valid_indices = np.where(base_valid_mask)[0]

# TTT model uses same indices
ttt_predictions_at_base_indices = ttt_predictions_np[base_valid_indices]
query_y_at_base_indices = query_y_np[base_valid_indices]

# Calculate metrics on exact same samples
ttt_accuracy = accuracy_score(query_y_at_base_indices, ttt_predictions_at_base_indices)
```

---

## 🔍 **Verification Checklist**

To ensure fair comparison, verify:

1. ✅ **Same test set source**: Both use `preprocessed_data['X_test']` and `preprocessed_data['y_test']`
2. ⚠️ **Same sample filtering**: Check if confidence rejection filters same samples
3. ⚠️ **Same valid mask**: Check if `base_valid_mask` is used for TTT metrics
4. ⚠️ **Same sample count**: Check logs for "BASE MODEL CM FINAL COUNT" vs "TTT MODEL CM FINAL COUNT"
5. ⚠️ **All metrics use same samples**: Not just confusion matrix

---

## 📊 **Current Status**

| Component                        | Base Model                    | TTT Model                      | Match?        |
| -------------------------------- | ----------------------------- | ------------------------------ | ------------- |
| **Test Set Source**              | `preprocessed_data['X_test']` | `preprocessed_data['X_test']`  | ✅ Yes        |
| **Initial Sample Count**         | Full test set                 | Full test set                  | ✅ Yes        |
| **Confidence Rejection**         | Applied                       | Applied                        | ⚠️ May differ |
| **Valid Mask for CM**            | `base_valid_mask`             | Tries to use `base_valid_mask` | ⚠️ May differ |
| **Valid Mask for Other Metrics** | `base_valid_mask`             | May use different mask         | ❌ **NO**     |

---

## 🎯 **Recommendation**

**Apply Fix 1**: Use `base_valid_mask` for ALL TTT metrics (not just confusion matrix) to ensure fair comparison across all performance metrics.



