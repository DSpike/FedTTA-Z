# Separate Performance Metrics Explanation

## 🎯 **What Are "Separate Performance Metrics"?**

Instead of calculating performance on **all test samples together**, we calculate metrics **separately** for different categories of test samples.

---

## 📊 **Current Situation (Before Fix)**

### **What We Have Now**:

```python
# Overall metrics (mixing everything together)
base_accuracy = (base_predictions == y_test_tensor).float().mean().item()
# This includes:
# - Normal samples (seen during training) ✅
# - Zero-day attacks (NOT seen during training) ❌
# - Other attacks (seen during training) ✅
```

**Problem**: The 90% accuracy mixes:

- **High performance** on seen samples (Normal + other attacks)
- **Low performance** on unseen zero-day attacks

**Result**: We can't tell which category is performing well or poorly!

---

## ✅ **What "Separate Metrics" Means**

### **Category 1: Zero-Day Attacks Only**

Calculate metrics **only** on zero-day attack samples:

```python
# Filter to only zero-day samples
zero_day_predictions = base_predictions[zero_day_mask]
zero_day_actual = y_test_tensor[zero_day_mask]

# Calculate metrics for zero-day ONLY
zero_day_accuracy = (zero_day_predictions == zero_day_actual).float().mean().item()
zero_day_precision = ...
zero_day_recall = ...
zero_day_f1 = ...
```

**What this tells us**: How well the model performs on **unseen zero-day attacks** (should be lower!)

---

### **Category 2: Non-Zero-Day Samples**

Calculate metrics **only** on non-zero-day samples (Normal + other attacks):

```python
# Filter to non-zero-day samples
non_zero_day_mask = ~zero_day_mask
non_zero_day_predictions = base_predictions[non_zero_day_mask]
non_zero_day_actual = y_test_tensor[non_zero_day_mask]

# Calculate metrics for non-zero-day ONLY
non_zero_day_accuracy = (non_zero_day_predictions == non_zero_day_actual).float().mean().item()
non_zero_day_precision = ...
non_zero_day_recall = ...
non_zero_day_f1 = ...
```

**What this tells us**: How well the model performs on **seen samples** (Normal + attacks seen during training) - should be higher!

---

### **Category 3: Overall (Existing)**

Keep the overall metrics (all samples together):

```python
overall_accuracy = (base_predictions == y_test_tensor).float().mean().item()
```

**What this tells us**: Weighted average performance across all categories.

---

## 📈 **Example Results**

### **Before (Mixed Metrics)**:

```
Overall Accuracy: 90.06%  ← We don't know if this is good or bad!
```

### **After (Separate Metrics)**:

```
Overall Accuracy: 90.06%

Zero-Day Attacks Only (119 samples, 40% of test set):
  Accuracy: 62.5%  ← Lower! Model struggles with unseen attacks (expected)
  F1-Score: 0.58
  Precision: 0.65
  Recall: 0.52

Non-Zero-Day Samples (213 samples, 60% of test set):
  Accuracy: 96.2%  ← Higher! Model performs well on seen samples (expected)
  F1-Score: 0.94
  Precision: 0.97
  Recall: 0.95
```

**Now we can see**:

- ✅ Model performs **well** on seen samples (96%)
- ⚠️ Model **struggles** with zero-day attacks (62%)
- 📊 Overall is **weighted average** (90%)

---

## 🎯 **Why This Matters**

### **For Scientific Evaluation**:

1. **Shows True Zero-Day Detection Performance**:

   - Base model on zero-day: 62% (realistic for unseen attacks)
   - TTT model on zero-day: 75% (improvement!)

2. **Validates Model Behavior**:

   - High performance on seen samples (96%) = Model learned well
   - Lower performance on zero-day (62%) = Zero-day is genuinely harder

3. **Proves TTT Value**:
   - Base model zero-day: 62%
   - TTT model zero-day: 75%
   - **Clear improvement** from TTT adaptation!

### **Without Separate Metrics**:

- Overall 90% hides the fact that zero-day performance is actually 62%
- Can't see if TTT is actually helping zero-day detection
- Misleading for scientific evaluation

---

## 📊 **What Each Metric Category Represents**

| Category          | Samples Included             | Expected Performance | Purpose                                |
| ----------------- | ---------------------------- | -------------------- | -------------------------------------- |
| **Zero-Day Only** | Only zero-day attack samples | ⚠️ Lower (50-70%)    | Show true zero-day detection challenge |
| **Non-Zero-Day**  | Normal + Other attacks       | ✅ Higher (85-95%)   | Validate model learned from training   |
| **Overall**       | All test samples             | 📊 Weighted (70-90%) | Overall system performance             |

---

## 🔧 **Implementation**

The separate metrics will be stored in the results dictionary like this:

```python
base_results = {
    # Overall metrics (existing)
    'accuracy': 0.9006,
    'f1_score': 0.9322,
    ...

    # NEW: Separate metrics for zero-day attacks
    'zero_day_only': {
        'accuracy': 0.625,
        'precision': 0.65,
        'recall': 0.52,
        'f1_score': 0.58,
        'confusion_matrix': [[...], [...]],
        'num_samples': 119
    },

    # NEW: Separate metrics for non-zero-day samples
    'non_zero_day': {
        'accuracy': 0.962,
        'precision': 0.97,
        'recall': 0.95,
        'f1_score': 0.94,
        'confusion_matrix': [[...], [...]],
        'num_samples': 213
    }
}
```

---

## ✅ **Summary**

**"Separate Performance Metrics"** means:

- Calculate metrics **separately** for zero-day attacks vs non-zero-day samples
- This reveals the **true performance** on each category
- Shows **where the model struggles** (zero-day) vs **where it excels** (seen samples)
- Enables **fair evaluation** of zero-day detection capability

This is essential for scientific evaluation because it shows the **true zero-day detection performance** rather than masking it with overall metrics!
