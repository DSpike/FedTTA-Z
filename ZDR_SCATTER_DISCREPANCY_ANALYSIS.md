# ZDR vs Scatter Plot Discrepancy Analysis

## 🔍 **Problem Statement**

The scatter plot shows **good separation** between attack and normal samples during TTT adaptation, but **ZDR (Zero-Day Detection Rate) is zero** in the performance visualization.

---

## 📊 **Root Cause Analysis**

### **1. Scatter Plot Shows ALL Attacks (Known + Zero-Day)**

**Location**: `coordinators/centralized_coordinator.py` lines 580-603

The scatter plot uses:
- **`attack_probs`**: Attack probabilities (sum of all non-zero class probabilities for multiclass)
- **`binary_labels`**: Created from **argmax predictions** - `binary_labels = (predictions != 0).long()`

**Key Point**: The scatter plot includes **BOTH known attacks AND zero-day attacks**. The binary_labels are created using:
```python
predictions = torch.argmax(logits, dim=1)  # Multiclass predictions
binary_labels = (predictions != 0).long()  # 0=normal, 1=attack (ANY attack)
```

This means:
- ✅ **Known attacks** are labeled as 1 (attack)
- ✅ **Zero-day attacks** are labeled as 1 (attack)
- ✅ **Normal samples** are labeled as 0 (normal)

**Result**: The scatter plot shows good separation because it includes **all attacks** (known + zero-day), and TTT successfully separates **all attacks** from normal.

---

### **2. ZDR Uses Threshold-Based Predictions (NOT Argmax)**

**Location**: `main.py` lines 6539-6723 (`_evaluate_ttt_model`)

ZDR calculation uses:
- **`ttt_predictions_np`**: Created using a **threshold** (not argmax)
- **Threshold**: Dynamically optimized (PR-based, ZDR-optimized, or ROC-based)
- **Only zero-day samples**: ZDR is calculated ONLY on zero-day samples

**Key Point**: The threshold might be **too high**, causing all zero-day samples to be predicted as Normal (0):

```python
# Threshold-based predictions
optimal_threshold = ...  # Optimized threshold (e.g., 0.6)
ttt_predictions_np = (attack_probabilities >= optimal_threshold).astype(int)
```

**Problem**: If zero-day attack probabilities are below the threshold (e.g., mean=0.4, threshold=0.6), then:
- ❌ **All zero-day samples** are predicted as Normal (0)
- ❌ **ZDR = TP / (TP + FN) = 0 / (0 + N) = 0.0**

---

## 🎯 **Why This Happens**

### **Scenario 1: Threshold Too High**

1. **Scatter plot** shows good separation (mean attack prob for all attacks = 0.7, normal = 0.2)
2. **But zero-day samples specifically** have lower attack probabilities (mean = 0.4)
3. **Threshold** is optimized for overall F1 (might be 0.6)
4. **Result**: All zero-day samples (prob < 0.6) are predicted as Normal → ZDR = 0

### **Scenario 2: TTT Adapts to Known Attacks, Not Zero-Day**

1. **TTT adaptation** uses query set with both known and zero-day attacks
2. **TTT successfully adapts** to separate known attacks from normal
3. **But zero-day attacks** are novel and TTT doesn't adapt well to them
4. **Result**: Known attacks get high attack probabilities, zero-day attacks get low probabilities
5. **Scatter plot** shows good separation (dominated by known attacks)
6. **ZDR** is zero (zero-day attacks have low probabilities, below threshold)

---

## ✅ **Solution: Diagnostic Logging**

Add diagnostic logging to compare:
1. **Scatter plot predictions** (argmax-based) vs **ZDR predictions** (threshold-based)
2. **Zero-day attack probabilities** vs **Known attack probabilities**
3. **Threshold value** used for ZDR calculation

**Location**: `main.py` lines 7242-7259

Already has some logging, but we should add:
- Comparison of zero-day vs known attack probabilities
- Threshold value and how many zero-day samples are above/below threshold
- Scatter plot predictions vs threshold-based predictions for zero-day samples

---

## 🔧 **Recommended Fixes**

### **Fix 1: Add Zero-Day Specific Threshold Optimization**

Currently, threshold is optimized for **overall F1** or **ZDR with FAR constraint**. But if zero-day samples have systematically lower probabilities, we need a **zero-day-specific threshold**.

**Option A**: Use a **lower threshold** for zero-day samples specifically
**Option B**: Optimize threshold to **maximize ZDR** (not overall F1)

### **Fix 2: Add Diagnostic Logging**

Add logging to show:
- Zero-day attack probabilities (mean, std, min, max)
- Known attack probabilities (mean, std, min, max)
- Threshold value
- How many zero-day samples are above/below threshold
- Scatter plot predictions vs threshold-based predictions for zero-day samples

### **Fix 3: Verify Zero-Day Mask Alignment**

Ensure `zero_day_mask` in TTT adaptation data matches the `zero_day_mask` used for ZDR calculation. If they don't align, ZDR will be calculated on wrong samples.

---

## 📋 **Next Steps**

1. ✅ Add diagnostic logging to compare scatter plot vs ZDR predictions
2. ✅ Check if zero-day samples have systematically lower attack probabilities
3. ✅ Verify threshold optimization is considering zero-day samples
4. ✅ Consider zero-day-specific threshold or lower threshold for zero-day detection



