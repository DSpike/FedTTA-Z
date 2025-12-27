# ZDR Zero Problem - Root Cause Analysis Summary

## 🔍 **Problem Identified**

**Issue**: ZDR (Zero-Day Detection Rate) is **zero** while the scatter plot shows **good separation** between attack and normal samples.

## 📊 **Root Cause**

### **1. Scatter Plot vs ZDR Use Different Prediction Methods**

- **Scatter Plot** (during TTT adaptation):
  - Uses **argmax predictions**: `binary_labels = (predictions != 0).long()`
  - Shows **ALL attacks** (known + zero-day) vs normal
  - Good separation because it includes **known attacks** which TTT adapts well to

- **ZDR Calculation** (after TTT):
  - Uses **threshold-based predictions**: `ttt_predictions = (attack_probabilities >= optimal_threshold).long()`
  - Only considers **zero-day samples**
  - If zero-day probabilities are below threshold → all predicted as Normal → ZDR = 0

### **2. Why Zero-Day Probabilities Are Lower**

**Hypothesis**: TTT adapts well to **known attacks** (seen during training) but struggles with **zero-day attacks** (novel patterns).

**Evidence Needed**:
- Zero-day attack probabilities (mean, std, min, max)
- Known attack probabilities (for comparison)
- Threshold value used
- How many zero-day samples are above/below threshold

## ✅ **Diagnostic Code Added**

Added comprehensive diagnostic logging at `main.py` lines 7256-7284 that will show:

1. **Zero-day attack probabilities**:
   - Mean, Std, Min, Max
   - Samples above/below threshold

2. **Known attack probabilities** (for comparison):
   - Mean, Std, Min, Max
   - Samples above/below threshold

3. **Root cause explanation**:
   - Probability difference between zero-day and known attacks
   - Why scatter plot shows good separation but ZDR is zero

## 🔧 **Next Steps**

1. **Run the system** and check the diagnostic output
2. **Verify** if zero-day probabilities are systematically lower than known attack probabilities
3. **Consider fixes**:
   - Lower threshold for zero-day detection
   - Zero-day-specific threshold optimization
   - Improve TTT adaptation for zero-day samples

## 📋 **Expected Diagnostic Output**

When ZDR is zero, you should see:
```
⚠️  CRITICAL: All {N} zero-day samples are predicted as Normal (0)!
   🔍 DIAGNOSTIC: Zero-day attack probabilities:
      Mean: X.XXXX, Std: X.XXXX
      Min: X.XXXX, Max: X.XXXX
      Samples above threshold (X.XXXX): 0/{N}
   🔍 DIAGNOSTIC: Known attack probabilities (for comparison):
      Mean: X.XXXX, Std: X.XXXX
      Min: X.XXXX, Max: X.XXXX
      Samples above threshold (X.XXXX): {M}/{K}
   🔍 ROOT CAUSE: Zero-day probabilities are X.XXXX LOWER than known attack probabilities!
      This explains why scatter plot shows good separation (includes known attacks)
      but ZDR is zero (zero-day probabilities below threshold)
```



