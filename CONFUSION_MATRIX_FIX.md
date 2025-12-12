# ✅ Confusion Matrix Fix - Handles Rejected Predictions (-1)

## 🎯 **Problem**

The confusion matrix was showing empty or no values because:

1. **Confidence-based rejection** marks uncertain predictions as `-1` (Unknown class)
2. **sklearn's `confusion_matrix()`** only accepts valid class labels (0, 1, 2, ...), not `-1`
3. When confusion matrices were calculated with predictions containing `-1`, they either:
   - Failed silently
   - Produced incorrect results
   - Showed empty/zero values

## ✅ **Solution**

**Filter out `-1` values before calculating confusion matrices:**

All confusion matrix calculations now:
1. **Filter out rejected predictions** (`-1`) before calculation
2. **Only use valid predictions** (>= 0) for confusion matrices
3. **Handle edge cases** where all predictions are rejected

---

## 📋 **Fixed Locations**

### **1. Base Model Overall Confusion Matrix** (Line ~3010)

**Before:**
```python
base_cm = confusion_matrix(y_test_filtered.cpu().numpy(), base_predictions.cpu().numpy())
```

**After:**
```python
# Filter out rejected predictions (-1) before confusion matrix calculation
valid_mask_cm = base_predictions.cpu().numpy() != -1
if valid_mask_cm.sum() > 0:
    base_cm = confusion_matrix(y_test_filtered.cpu().numpy()[valid_mask_cm], base_predictions.cpu().numpy()[valid_mask_cm])
else:
    base_cm = np.array([[0, 0], [0, 0]])  # Empty confusion matrix if all rejected
```

---

### **2. Base Model Zero-Day Confusion Matrix** (Line ~3040-3050)

**Before:**
```python
zero_day_y_pred_bin = (zero_day_predictions.cpu().numpy() != 0).astype(int)
zero_day_cm = confusion_matrix(zero_day_y_true_bin, zero_day_y_pred_bin)
```

**After:**
```python
# Filter out rejected predictions (-1) before calculating metrics
zero_day_valid_mask = (zero_day_predictions.cpu().numpy() != -1)
if zero_day_valid_mask.sum() > 0:
    zero_day_predictions_valid = zero_day_predictions.cpu().numpy()[zero_day_valid_mask]
    zero_day_actual_valid = zero_day_actual.cpu().numpy()[zero_day_valid_mask]
    zero_day_y_true_bin = (zero_day_actual_valid != 0).astype(int)
    zero_day_y_pred_bin = (zero_day_predictions_valid != 0).astype(int)
    zero_day_cm = confusion_matrix(zero_day_y_true_bin, zero_day_y_pred_bin)
else:
    zero_day_cm = np.array([[0, 0], [0, 0]])  # All rejected
```

---

### **3. Base Model Non-Zero-Day Confusion Matrix** (Line ~3174-3181)

**Similar fix applied** - filters out `-1` before calculating confusion matrix.

---

### **4. TTT Model Overall Confusion Matrix** (Line ~3965-3966)

**Before:**
```python
adapted_cm = confusion_matrix(y_test_tensor.cpu().numpy(), adapted_predictions.cpu().numpy())
adapted_cm_binary = confusion_matrix(y_test_binary, adapted_predictions_binary)
```

**After:**
```python
# Filter out rejected predictions (-1) before confusion matrix calculation
adapted_valid_mask_cm = adapted_predictions.cpu().numpy() != -1
if adapted_valid_mask_cm.sum() > 0:
    adapted_cm = confusion_matrix(y_test_tensor.cpu().numpy()[adapted_valid_mask_cm], adapted_predictions.cpu().numpy()[adapted_valid_mask_cm])
else:
    adapted_cm = np.array([[0, 0], [0, 0]])

# For binary confusion matrix, filter out rejected from adapted_predictions_binary
adapted_binary_valid_mask = adapted_predictions_binary != -1
if adapted_binary_valid_mask.sum() > 0:
    adapted_cm_binary = confusion_matrix(y_test_binary[adapted_binary_valid_mask], adapted_predictions_binary[adapted_binary_valid_mask])
else:
    adapted_cm_binary = np.array([[0, 0], [0, 0]])
```

---

### **5. TTT Model Zero-Day Confusion Matrix** (Line ~3970-3980)

**Similar fix applied** - filters out `-1` before calculating confusion matrix.

---

### **6. TTT Model Non-Zero-Day Confusion Matrix** (Line ~4091-4098)

**Similar fix applied** - filters out `-1` before calculating confusion matrix.

---

## 🔍 **Why This Was Needed**

### **The Issue:**
- `confusion_matrix(y_true, y_pred)` expects valid class labels
- When `y_pred` contains `-1`, sklearn either:
  - Treats `-1` as an invalid label (causing errors)
  - Produces unexpected/empty confusion matrices
  - Fails to calculate properly

### **The Fix:**
1. **Filter before calculation**: Remove `-1` from predictions and labels
2. **Synchronize filtering**: Keep predictions and labels aligned
3. **Handle edge cases**: Provide empty confusion matrix if all rejected

---

## 💡 **How It Works Now**

```python
# Example: Predictions contain -1 (rejected)
predictions = [0, 1, -1, 0, 1, -1]  # Some rejected
labels = [0, 1, 1, 0, 1, 1]

# Step 1: Filter out -1
valid_mask = predictions != -1  # [True, True, False, True, True, False]
valid_predictions = predictions[valid_mask]  # [0, 1, 0, 1]
valid_labels = labels[valid_mask]  # [0, 1, 0, 1]

# Step 2: Calculate confusion matrix on valid data only
cm = confusion_matrix(valid_labels, valid_predictions)
# Result: Valid 2x2 confusion matrix
```

---

## ✅ **Status**

- ✅ All 6 confusion matrix calculations fixed
- ✅ Rejected predictions (`-1`) filtered out before calculation
- ✅ Edge cases handled (all predictions rejected)
- ✅ Confusion matrices now show correct values
- ✅ No linter errors

**Implementation Complete!** ✅

---

## 🎯 **Expected Result**

Confusion matrices should now display:
- **Correct values** (TN, FP, FN, TP)
- **Only valid predictions** (excludes rejected `-1`)
- **Proper 2x2 matrices** for binary classification
- **Meaningful metrics** derived from confusion matrices

The confusion matrix will reflect performance only on **confident predictions** (those that passed the confidence threshold), which aligns with the confidence-based rejection strategy! 🚀






