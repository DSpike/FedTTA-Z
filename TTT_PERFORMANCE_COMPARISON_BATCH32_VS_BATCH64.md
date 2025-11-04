# TTT Performance Comparison: Batch Size 32 vs Batch Size 64

## Overview

This document compares the overall TTT performance between batch size 32 and batch size 64 to determine which configuration provides better results.

---

## Latest Run Performance (Batch Size 64)

### **Single Evaluation Results**:

**Base Model:**

- Accuracy: **0.8705** (87.05%)
- F1-Score: **0.9051** (90.51%)
- AUC-PR: **0.8765** (87.65%)
- Zero-day Detection Rate: **0.8649** (86.49%)

**TTT Model:**

- Accuracy: **0.8765** (87.65%)
- F1-Score: **0.9099** (90.99%)
- AUC-PR: **0.9360** (93.60%)
- Zero-day Detection Rate: **0.8919** (89.19%)

**Improvements (TTT vs Base):**

- Accuracy: **+0.0060** (+0.60%)
- F1-Score: **+0.0048** (+0.48%)
- AUC-PR: **+0.0595** (+5.95%) ⭐ **PRIMARY METRIC**
- Zero-day Detection: **+0.0270** (+2.70%)

### **K-Fold Cross-Validation Results**:

**Base Model:**

- Accuracy: **0.8765 ± 0.0261** (87.65% ± 2.61%)

**TTT Model:**

- Accuracy: **0.8734 ± 0.0249** (87.34% ± 2.49%)
- F1-Score: **0.8454 ± 0.0312**
- MCC: **0.7056 ± 0.0581**

**K-Fold Improvement:**

- Accuracy: **-0.0031** (-0.31%) ⚠️ (slight decrease, but within std dev)

---

## Comparison with Previous Run (Batch Size 32)

### **From Previous Analysis (Batch Size 32)**:

Based on earlier runs and documentation (PROJECT_SUMMARY.md, HOW_TO_FRAME_YOUR_CONTRIBUTION.md), batch size 32 showed:

**K-Fold CV (From PROJECT_SUMMARY.md - Previous Run):**

- Base Model: **81.90% ± 4.45%**
- TTT Model: **88.83% ± 4.97%**
- Improvement: **+6.93%** ✅

**Single Evaluation (From Previous Runs):**

- Base Model Accuracy: ~0.798-0.855
- TTT Model Accuracy: ~0.873-0.888
- Improvement: **+0.0181 to +0.020** (+1.81% to +2.0%)

**AUC-PR Improvement:**

- **+0.0670** (+6.70%) (from previous run logs)

**Zero-day Detection:**

- Base: ~0.785-0.855
- TTT: ~0.873-0.900
- Improvement: **+0.0429** (+4.29%)

**Note**: These results are from different runs with potentially different configurations (rounds, clients, attack types). Direct comparison may not be completely fair.

---

## Direct Comparison

### **Performance Metrics Comparison**:

| Metric                         | Batch Size 32             | Batch Size 64    | Better?                              |
| ------------------------------ | ------------------------- | ---------------- | ------------------------------------ |
| **Single Evaluation Accuracy** | ~+0.018-0.020 (+1.8-2.0%) | +0.0060 (+0.60%) | ⚠️ **Batch 32** (larger improvement) |
| **K-Fold CV Accuracy**         | +6.93%                    | -0.31%           | ✅ **Batch 32** (much better)        |
| **AUC-PR Improvement**         | +0.0670 (+6.70%)          | +0.0595 (+5.95%) | ⚠️ **Batch 32** (slightly better)    |
| **Zero-day Detection**         | +0.0429 (+4.29%)          | +0.0270 (+2.70%) | ⚠️ **Batch 32** (better)             |
| **F1-Score Improvement**       | +0.0136 (+1.36%)          | +0.0048 (+0.48%) | ⚠️ **Batch 32** (better)             |

### **Convergence Metrics Comparison**:

| Metric                  | Batch Size 32        | Batch Size 64            | Better?                          |
| ----------------------- | -------------------- | ------------------------ | -------------------------------- |
| **Loss Reduction**      | -7.3%                | **-11.4%**               | ✅ **Batch 64** (much better)    |
| **Gradient Norm Trend** | +5.31% INCREASING ❌ | **-4.09% DECREASING** ✅ | ✅ **Batch 64** (reversed trend) |
| **Loss per Step**       | -0.0005              | **-0.00068**             | ✅ **Batch 64** (faster)         |
| **Gradient Stability**  | Medium               | **High**                 | ✅ **Batch 64** (more stable)    |

---

## Key Findings

### ⚠️ **Performance Metrics: Batch Size 32 is Better**

1. **Larger Accuracy Improvements**:

   - Batch 32: +1.8-2.0% (single) / +6.93% (k-fold)
   - Batch 64: +0.60% (single) / -0.31% (k-fold)
   - **Batch 32 shows significantly larger improvements**

2. **Better AUC-PR Improvement**:

   - Batch 32: +6.70%
   - Batch 64: +5.95%
   - **Batch 32 is slightly better**

3. **Better Zero-day Detection**:
   - Batch 32: +4.29%
   - Batch 64: +2.70%
   - **Batch 32 is better**

### ✅ **Convergence Metrics: Batch Size 64 is Better**

1. **Faster Loss Convergence**:

   - Batch 32: -7.3% loss reduction
   - Batch 64: **-11.4% loss reduction** ✅
   - **Batch 64 converges faster**

2. **Better Gradient Norm Behavior**:

   - Batch 32: +5.31% INCREASING ❌
   - Batch 64: **-4.09% DECREASING** ✅
   - **Batch 64 shows proper convergence trend**

3. **More Stable Optimization**:
   - Batch 64 provides more stable gradients
   - Better loss component balance
   - Smoother optimization path

---

## Analysis: Why the Discrepancy?

### **Possible Explanations**:

1. **Different Runs/Datasets**:

   - The runs may have used different configurations
   - Different number of rounds/clients
   - Different data splits or attack types

2. **Statistical Variance**:

   - K-fold CV shows batch 64 has slightly lower accuracy
   - But within standard deviation (0.0249 vs 0.0261)
   - May not be statistically significant

3. **Trade-off Between Stability and Performance**:

   - Batch 64: More stable optimization (better convergence)
   - Batch 32: More exploration (better final performance)
   - May need to balance both aspects

4. **Different Evaluation Timing**:
   - Previous batch 32 results may have been from different training stage
   - Current batch 64 results are from latest run
   - Direct comparison may not be fair

---

## Conclusion

### **Overall Assessment**:

**Performance Metrics**: ⚠️ **Batch Size 32 appears better**

- Larger accuracy improvements
- Better AUC-PR and zero-day detection improvements
- **However**, this may be due to different run conditions

**Convergence Metrics**: ✅ **Batch Size 64 is clearly better**

- Faster loss convergence (-11.4% vs -7.3%)
- Proper gradient norm trend (decreasing vs increasing)
- More stable optimization

### **Recommendation**:

**For Stability and Convergence**: ✅ **Use Batch Size 64**

- Better gradient norm behavior
- Faster loss convergence
- More stable optimization

**For Final Performance**: ⚠️ **May need more investigation**

- Batch 32 shows better performance improvements
- But may be due to different run conditions
- Need to run batch 32 again with same conditions for fair comparison

### **Suggested Next Steps**:

1. **Run Batch Size 32 again** with same configuration as batch 64:

   - Same number of rounds (15)
   - Same number of clients (5)
   - Same attack type (Analysis)
   - Direct comparison will be more fair

2. **Test Batch Size 128**:

   - May provide even better stability
   - May maintain or improve performance

3. **Consider Adaptive Batch Size**:
   - Start with smaller batch (exploration)
   - Increase to larger batch (exploitation)
   - Best of both worlds

---

## Summary

**TTT Performance Comparison: Batch 32 vs Batch 64**

| Aspect                     | Batch Size 32                        | Batch Size 64                     | Winner          |
| -------------------------- | ------------------------------------ | --------------------------------- | --------------- |
| **Accuracy Improvement**   | +1.8-2.0% (single) / +6.93% (k-fold) | +0.60% (single) / -0.31% (k-fold) | ⚠️ **Batch 32** |
| **AUC-PR Improvement**     | +6.70%                               | +5.95%                            | ⚠️ **Batch 32** |
| **Zero-day Detection**     | +4.29%                               | +2.70%                            | ⚠️ **Batch 32** |
| **Loss Convergence**       | -7.3%                                | **-11.4%**                        | ✅ **Batch 64** |
| **Gradient Norm**          | +5.31% (increasing) ❌               | **-4.09% (decreasing)** ✅        | ✅ **Batch 64** |
| **Optimization Stability** | Medium                               | **High**                          | ✅ **Batch 64** |

**Overall**: Batch size 64 provides **better convergence behavior** but may show **slightly lower performance improvements** compared to batch size 32. However, the performance difference may be due to different run conditions. A fair comparison requires running both with identical configurations.
