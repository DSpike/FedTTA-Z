# Implementation Plan: Fix Zero-Day Mask and Evaluation

## 🎯 **Goal**

Fix the incorrect zero-day mask and provide **separate, accurate evaluation** for:

- Zero-day attacks only (unseen during training)
- Non-zero-day test samples (seen during training)
- Overall performance

---

## 📋 **Proposed Solution (Simple & Effective)**

### **Approach: Use Attack Label Instead of Indices**

Since `zero_day_indices` are broken after sequence creation, we'll use the **zero-day attack label** directly from `y_test`.

---

## 🔧 **Implementation Steps**

### **Step 1: Fix Zero-Day Mask Creation** ⚠️ **CRITICAL**

**File**: `main.py`  
**Method**: `evaluate_base_model_only()` (lines 1988-1997)

**Current Code** (WRONG):

```python
if len(zero_day_indices) == 0:
    zero_day_mask = torch.ones(len(y_test), dtype=torch.bool)
else:
    zero_day_mask = torch.ones(len(y_test), dtype=torch.bool)  # ❌ ALL samples
```

**Fixed Code**:

```python
# Get zero-day attack label from preprocessed_data
zero_day_attack = self.preprocessed_data.get('zero_day_attack', 'Generic')
attack_types = self.preprocessed_data.get('attack_types', {})
zero_day_attack_label = attack_types.get(zero_day_attack, 1)  # Default to label 1

# Create zero-day mask: mark samples where label == zero_day_attack_label
# Note: For sequences, the label comes from the last timestep
zero_day_mask = (y_test_tensor == zero_day_attack_label)

logger.info(f"Zero-day mask created: {zero_day_mask.sum().item()}/{len(zero_day_mask)} samples")
logger.info(f"Zero-day attack: {zero_day_attack}, label: {zero_day_attack_label}")
logger.info(f"Test label distribution: {torch.bincount(y_test_tensor)}")
```

---

### **Step 2: Add Separate Performance Metrics** 📊

**File**: `main.py`  
**Method**: `evaluate_base_model_only()` (after line 2044)

**Add Code**:

```python
# Calculate metrics separately for zero-day and non-zero-day samples
zero_day_predictions = base_predictions[zero_day_mask]
zero_day_actual = y_test_tensor[zero_day_mask]

non_zero_day_mask = ~zero_day_mask
non_zero_day_predictions = base_predictions[non_zero_day_mask]
non_zero_day_actual = y_test_tensor[non_zero_day_mask]

# Zero-day only metrics
if len(zero_day_actual) > 0:
    zero_day_accuracy = (zero_day_predictions == zero_day_actual).float().mean().item()
    zero_day_y_true_bin = (zero_day_actual.cpu().numpy() != 0).astype(int)
    zero_day_y_pred_bin = (zero_day_predictions.cpu().numpy() != 0).astype(int)
    zero_day_precision = _prec(zero_day_y_true_bin, zero_day_y_pred_bin, zero_division=0)
    zero_day_recall = _rec(zero_day_y_true_bin, zero_day_y_pred_bin, zero_division=0)
    zero_day_f1 = _f1(zero_day_y_true_bin, zero_day_y_pred_bin, zero_division=0)
    zero_day_cm = confusion_matrix(zero_day_y_true_bin, zero_day_y_pred_bin)
else:
mes    zero_day_accuracy = 0.0
    zero_day_precision = 0.0
    zero_day_recall = 0.0
    zero_day_f1 = 0.0
    zero_day_cm = [[0, 0], [0, 0]]

# Non-zero-day metrics
if len(non_zero_day_actual) > 0:
    non_zero_day_accuracy = (non_zero_day_predictions == non_zero_day_actual).float().mean().item()
    non_zero_day_y_true_bin = (non_zero_day_actual.cpu().numpy() != 0).astype(int)
    non_zero_day_y_pred_bin = (non_zero_day_predictions.cpu().numpy() != 0).astype(int)
    non_zero_day_precision = _prec(non_zero_day_y_true_bin, non_zero_day_y_pred_bin, zero_division=0)
    non_zero_day_recall = _rec(non_zero_day_y_true_bin, non_zero_day_y_pred_bin, zero_division=0)
    non_zero_day_f1 = _f1(non_zero_day_y_true_bin, non_zero_day_y_pred_bin, zero_division=0)
    non_zero_day_cm = confusion_matrix(non_zero_day_y_true_bin, non_zero_day_y_pred_bin)
else:
    non_zero_day_accuracy = 0.0
    non_zero_day_precision = 0.0
    non_zero_day_recall = 0.0
    non_zero_day_f1 = 0.0
    non_zero_day_cm = [[0, 0], [0, 0]]
```

---

### **Step 3: Update Results Dictionary** 📝

**File**: `main.py`  
**Method**: `evaluate_base_model_only()` (update `base_results` dictionary around line 2051)

**Add to `base_results`**:

```python
base_results = {
    # ... existing fields ...

    # Separate metrics for zero-day attacks
    'zero_day_only': {
        'accuracy': zero_day_accuracy,
        'precision': zero_day_precision,
        'recall': zero_day_recall,
        'f1_score': zero_day_f1,
        'confusion_matrix': zero_day_cm.tolist() if isinstance(zero_day_cm, np.ndarray) else zero_day_cm,
        'num_samples': len(zero_day_actual)
    },

    # Separate metrics for non-zero-day samples
    'non_zero_day': {
        'accuracy': non_zero_day_accuracy,
        'precision': non_zero_day_precision,
        'recall': non_zero_day_recall,
        'f1_score': non_zero_day_f1,
        'confusion_matrix': non_zero_day_cm.tolist() if isinstance(non_zero_day_cm, np.ndarray) else non_zero_day_cm,
        'num_samples': len(non_zero_day_actual)
    }
}
```

---

### **Step 4: Update Logging** 📊

**File**: `main.py`  
**Method**: `evaluate_base_model_only()` (update logging around line 2072)

**Add**:

```python
logger.info(f"✅ Base Model Results:")
logger.info(f"   Overall Accuracy: {base_accuracy:.4f}")
logger.info(f"   Overall F1-Score: {base_f1_conventional:.4f}")
logger.info(f"\n   🔴 Zero-Day Attacks Only ({len(zero_day_actual)} samples):")
logger.info(f"      Accuracy: {zero_day_accuracy:.4f}")
logger.info(f"      F1-Score: {zero_day_f1:.4f}")
logger.info(f"      Precision: {zero_day_precision:.4f}")
logger.info(f"      Recall: {zero_day_recall:.4f}")
logger.info(f"\n   🟢 Non-Zero-Day Samples ({len(non_zero_day_actual)} samples):")
logger.info(f"      Accuracy: {non_zero_day_accuracy:.4f}")
logger.info(f"      F1-Score: {non_zero_day_f1:.4f}")
logger.info(f"      Precision: {non_zero_day_precision:.4f}")
logger.info(f"      Recall: {non_zero_day_recall:.4f}")
```

---

### **Step 5: Apply Same Fix to TTT Evaluation** 🔄

**File**: `main.py`  
**Method**: `evaluate_adapted_model_only()` (similar changes around line 2150)

Apply the same fixes:

1. Fix zero-day mask creation
2. Add separate metrics calculation
3. Update results dictionary
4. Update logging

---

### **Step 6: Update Visualization** 📈

**File**: `visualization/performance_visualization.py`

Add function to plot separate zero-day vs non-zero-day performance:

```python
def plot_zero_day_vs_non_zero_day(base_results, ttt_results, save=True):
    """Plot separate performance for zero-day and non-zero-day samples"""
    # Implementation here
```

---

## ✅ **Validation Checklist**

After implementation, verify:

1. ✅ Zero-day mask correctly identifies zero-day samples

   - Check: `zero_day_mask.sum() < len(y_test)` (not all samples)
   - Expected: ~30-40% of test samples are zero-day

2. ✅ Zero-day-only accuracy is **LOWER** than overall

   - Expected: Zero-day accuracy ~50-70%
   - Overall accuracy ~70-80%

3. ✅ Non-zero-day accuracy is **HIGHER** than overall

   - Expected: Non-zero-day accuracy ~85-95%
   - This confirms model performs well on seen attacks

4. ✅ Logs show correct breakdown:
   - Zero-day samples count
   - Non-zero-day samples count
   - Separate metrics for each

---

## 🚀 **Quick Implementation Order**

1. **Start with Step 1** (fix mask) - **MOST CRITICAL**
2. **Then Step 2-4** (separate metrics) - **HIGH PRIORITY**
3. **Then Step 5** (apply to TTT) - **HIGH PRIORITY**
4. **Finally Step 6** (visualization) - **MEDIUM PRIORITY**

---

## 📊 **Expected Results After Fix**

### **Base Model**:

- **Zero-day-only accuracy**: 50-70% ⬇️ (realistic for unseen attacks)
- **Non-zero-day accuracy**: 85-95% ✅ (expected - model has seen these)
- **Overall 🠀 accuracy**: 70-80% (weighted average)

### **TTT Model**:

- **Zero-day-only accuracy**: 60-80% ⬆️ (improvement from adaptation)
- **Non-zero-day accuracy**: 85-95% (similar to base)
- **Overall accuracy**: 75-85% (weighted average)

This will show:

- ✅ Base model **struggles** with zero-day attacks (as expected)
- ✅ TTT adaptation **improves** zero-day detection (proves its value)
- ✅ Both models perform well on known attacks (validation)

---

## 🎯 **Summary**

**Problem**: Zero-day mask marks ALL samples as zero-day, inflating performance.

**Solution**:

1. Use attack label to identify zero-day samples
2. Calculate separate metrics for zero-day vs non-zero-day
3. Report all three: zero-day, non-zero-day, overall

**Impact**:

- Accurate zero-day detection evaluation
- Clear demonstration of TTT value
- Proper scientific evaluation methodology

