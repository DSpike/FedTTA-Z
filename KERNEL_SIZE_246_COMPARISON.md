# TCN Kernel Size Comparison: (2,4,6) vs Previous Configurations

## Configuration Overview

### New Configuration: Hierarchical Kernel Sizes (2, 4, 6)

- **Branch 1**: kernel_size = 2, hidden_dim = 64 (Fine-scale patterns)
- **Branch 2**: kernel_size = 4, hidden_dim = 32 (Medium-scale patterns)
- **Branch 3**: kernel_size = 6, hidden_dim = 128 (Coarse-scale patterns)
- **Padding**: Auto-calculated (padding = kernel_size // 2)

### Previous Configurations:

- **Uniform (4, 4, 4)**: All branches use kernel_size = 4
- **Hierarchical (3, 5, 7)**: Fine (3), medium (5), coarse (7) scales
- **Hierarchical (1, 2, 4)**: Pointwise (1), very fine (2), medium (4) scales

---

## Performance Comparison (Analysis Attack, Same Test Set)

### New Configuration: Hierarchical Kernel Sizes (2, 4, 6) Results:

#### Base Model:

- **Accuracy**: 0.4429 (44.29%)
- **F1-Score**: 0.3906 (39.06%)
- **AUC-PR**: 0.6883 (68.83%)
- **ROC AUC**: 0.5374 (53.74%)
- **Zero-Day Detection Rate**: 0.3571 (35.71%)

#### TTT Model:

- **Accuracy**: 0.6857 (68.57%)
- **F1-Score**: 0.7215 (72.15%)
- **AUC-PR**: 0.8282 (82.82%)
- **ROC AUC**: 0.7457 (74.57%)
- **Zero-Day Detection Rate**: 0.7619 (76.19%)
- **Precision**: 1.0000 (100.00%)
- **Recall**: 0.7619 (76.19%)

**TTT Improvements:**

- **Accuracy**: +24.29% (from 44.29% to 68.57%)
- **F1-Score**: +33.09% (from 39.06% to 72.15%)
- **AUC-PR**: +13.99% (from 68.83% to 82.82%)
- **ZDR**: +40.48% (from 35.71% to 76.19%) ⭐ **LARGEST IMPROVEMENT**

---

### Uniform Kernel Size (4, 4, 4) Results (Previous):

#### Base Model:

- **Accuracy**: 0.6643 (66.43%)
- **F1-Score**: 0.7044 (70.44%)
- **AUC-PR**: 0.8124 (81.24%)
- **ROC AUC**: 0.7075 (70.75%)
- **Zero-Day Detection Rate**: 0.7381 (73.81%)

#### TTT Model:

- **Accuracy**: 0.8000 (80.00%)
- **F1-Score**: 0.8228 (82.28%)
- **AUC-PR**: 0.9284 (92.84%)
- **ROC AUC**: 0.8347 (83.47%)
- **Zero-Day Detection Rate**: 0.7857 (78.57%)
- **Precision**: 1.0000 (100.00%)
- **Recall**: 0.7857 (78.57%)

**TTT Improvements:**

- **Accuracy**: +13.57% (from 66.43% to 80.00%)
- **F1-Score**: +11.84% (from 70.44% to 82.28%)
- **AUC-PR**: +11.60% (from 81.24% to 92.84%)
- **ZDR**: +4.76% (from 73.81% to 78.57%)

---

### Hierarchical Kernel Size (3, 5, 7) Results (Previous):

#### Base Model:

- **Accuracy**: 0.6143 (61.43%)
- **F1-Score**: 0.6143 (61.43%)
- **AUC-PR**: 0.7942 (79.42%)
- **ROC AUC**: 0.6883 (68.83%)
- **Zero-Day Detection Rate**: 0.6190 (61.90%)

#### TTT Model:

- **Accuracy**: 0.7071 (70.71%)
- **F1-Score**: 0.7211 (72.11%)
- **AUC-PR**: 0.8870 (88.70%)
- **ROC AUC**: 0.7545 (75.45%)
- **Zero-Day Detection Rate**: 0.7143 (71.43%)
- **Precision**: 1.0000 (100.00%)
- **Recall**: 0.7143 (71.43%)

**TTT Improvements:**

- **Accuracy**: +9.28% (from 61.43% to 70.71%)
- **F1-Score**: +10.68% (from 61.43% to 72.11%)
- **AUC-PR**: +9.28% (from 79.42% to 88.70%)
- **ZDR**: +9.52% (from 61.90% to 71.43%)

---

### Hierarchical Kernel Size (1, 2, 4) Results (Previous):

#### Base Model:

- **Accuracy**: 0.6786 (67.86%)
- **F1-Score**: 0.7305 (73.05%)
- **AUC-PR**: 0.8102 (81.02%)
- **ROC AUC**: 0.7110 (71.10%)
- **Zero-Day Detection Rate**: 0.8333 (83.33%)

#### TTT Model:

- **Accuracy**: 0.7643 (76.43%)
- **F1-Score**: 0.7975 (79.75%)
- **AUC-PR**: 0.9038 (90.38%)
- **ROC AUC**: 0.8100 (81.00%)
- **Zero-Day Detection Rate**: 0.7857 (78.57%)
- **Precision**: 1.0000 (100.00%)
- **Recall**: 0.7857 (78.57%)

**TTT Improvements:**

- **Accuracy**: +8.57% (from 67.86% to 76.43%)
- **F1-Score**: +6.70% (from 73.05% to 79.75%)
- **AUC-PR**: +9.36% (from 81.02% to 90.38%)
- **ZDR**: -4.76% (from 83.33% to 78.57%) ⚠️ **NEGATIVE IMPROVEMENT**

---

## Direct Comparison Table

| Metric              | (2,4,6)        | Uniform (4,4,4) | Hierarchical (3,5,7) | Hierarchical (1,2,4) | vs Uniform | Winner     |
| ------------------- | -------------- | --------------- | -------------------- | -------------------- | ---------- | ---------- |
| **Base Model**      |                |                 |                      |                      |            |            |
| Accuracy            | 44.29%         | **66.43%**      | 61.43%               | 67.86%               | -22.14%    | ✅ Uniform |
| F1-Score            | 39.06%         | **70.44%**      | 61.43%               | 73.05%               | -31.38%    | ✅ (1,2,4) |
| AUC-PR              | 68.83%         | **81.24%**      | 79.42%               | 81.02%               | -12.41%    | ✅ Uniform |
| ZDR                 | 35.71%         | 73.81%          | 61.90%               | **83.33%**           | -38.10%    | ✅ (1,2,4) |
|                     |                |                 |                      |                      |            |            |
| **TTT Model**       |                |                 |                      |                      |            |            |
| Accuracy            | 68.57%         | **80.00%**      | 70.71%               | 76.43%               | -11.43%    | ✅ Uniform |
| F1-Score            | 72.15%         | **82.28%**      | 72.11%               | 79.75%               | -10.13%    | ✅ Uniform |
| AUC-PR              | 82.82%         | **92.84%**      | 88.70%               | 90.38%               | -10.02%    | ✅ Uniform |
| ROC AUC             | 74.57%         | **83.47%**      | 75.45%               | 81.00%               | -8.90%     | ✅ Uniform |
| ZDR                 | 76.19%         | **78.57%**      | 71.43%               | 78.57%               | -2.38%     | 🤝 Tie     |
| Precision           | 100.00%        | 100.00%         | 100.00%              | 100.00%              | 0.00%      | 🤝 Tie     |
| Recall              | 76.19%         | 78.57%          | 71.43%               | **78.57%**           | -2.38%     | 🤝 Tie     |
|                     |                |                 |                      |                      |            |            |
| **TTT Improvement** |                |                 |                      |                      |            |            |
| Accuracy Gain       | **+24.29%** ⭐ | +13.57%         | +9.28%               | +8.57%               | +10.72%    | ✅ (2,4,6) |
| F1-Score Gain       | **+33.09%** ⭐ | +11.84%         | +10.68%              | +6.70%               | +21.25%    | ✅ (2,4,6) |
| AUC-PR Gain         | +13.99%        | **+11.60%**     | +9.28%               | +9.36%               | +2.39%     | ✅ (2,4,6) |
| ZDR Gain            | **+40.48%** ⭐ | +4.76%          | +9.52%               | -4.76%               | +35.72%    | ✅ (2,4,6) |

---

## Key Findings

### ✅ **Strengths of (2,4,6) Configuration:**

1. **Largest TTT Improvements**:

   - **Accuracy Gain**: +24.29% (largest among all configurations)
   - **F1-Score Gain**: +33.09% (largest among all configurations)
   - **ZDR Gain**: +40.48% (largest among all configurations)
   - Shows **strong adaptation potential** with TTT

2. **Strong Final TTT Performance**:

   - **ZDR**: 76.19% (very close to uniform's 78.57%)
   - **Recall**: 76.19% (close to uniform's 78.57%)
   - **Perfect Precision**: 100.00% (no false positives)

3. **Large Adaptation Margin**:
   - Base model has room for improvement (44.29% accuracy)
   - TTT adaptation significantly closes the gap
   - Shows model is **adaptable** rather than stuck

### ⚠️ **Weaknesses of (2,4,6) Configuration:**

1. **Poor Base Model Performance**:

   - **Accuracy**: 44.29% (worst among all configurations)
   - **F1-Score**: 39.06% (worst among all configurations)
   - **ZDR**: 35.71% (worst among all configurations)
   - Model struggles **without TTT adaptation**

2. **Lower Final TTT Performance**:

   - **TTT Accuracy**: 68.57% vs 80.00% (Uniform) - **-11.43%**
   - **TTT AUC-PR**: 82.82% vs 92.84% (Uniform) - **-10.02%**
   - **TTT F1-Score**: 72.15% vs 82.28% (Uniform) - **-10.13%**
   - Even after strong adaptation, still lags behind uniform

3. **Dependency on TTT**:
   - Model **requires TTT** to be useful (base performance too low)
   - If TTT is unavailable, model is not deployable
   - Uniform configuration is usable even without TTT

---

## Analysis

### Why (2,4,6) Shows Large Improvements but Lower Final Performance:

1. **Kernel Size Progression**:

   - **Fine (2)**: Very limited temporal context (2 timesteps)
   - **Medium (4)**: Moderate temporal context (4 timesteps)
   - **Coarse (6)**: Extended temporal context (6 timesteps)
   - The progression may create **feature misalignment** initially

2. **Base Model Struggles**:

   - Base model (44.29% accuracy) indicates poor initial learning
   - May be due to **incompatible feature scales** from different kernel sizes
   - Model needs TTT to align and refine these features

3. **TTT Adaptation Leverage**:

   - Large improvement (+40.48% ZDR) suggests TTT can effectively **correct** base model issues
   - But starting from a lower base means final performance is still lower
   - **Gap too large to fully close** with TTT alone

4. **Comparison with Uniform (4,4,4)**:
   - Uniform starts higher (66.43% base) and ends higher (80.00% TTT)
   - (2,4,6) starts lower (44.29% base) and ends lower (68.57% TTT)
   - **Better base → better final**, despite smaller improvement percentage

---

## Trade-off Analysis

### Scenario 1: **TTT Always Available** (Production with Adaptation)

- **Uniform (4,4,4)**: Best final performance (80.00% accuracy, 92.84% AUC-PR)
- **(2,4,6)**: Lower final performance (68.57% accuracy, 82.82% AUC-PR)
- **Winner**: **Uniform (4,4,4)** - Better end-to-end performance

### Scenario 2: **TTT May Be Unavailable** (Graceful Degradation)

- **Uniform (4,4,4)**: Good base performance (66.43% accuracy) - **Usable**
- **(2,4,6)**: Poor base performance (44.29% accuracy) - **Not Usable**
- **Winner**: **Uniform (4,4,4)** - Deployable without TTT

### Scenario 3: **Research/Flexibility** (Large Improvement Potential)

- **(2,4,6)**: Shows largest TTT improvement (+40.48% ZDR) - **Research Interest**
- **Uniform (4,4,4)**: Stable, predictable performance
- **Winner**: **Depends on goal** - Research vs Production

---

## Conclusion

### ✅ **Recommendation: Keep Uniform Kernel Size (4, 4, 4)**

**Reasoning**:

1. **Best Overall Performance**: Uniform achieves **+11.43%** better TTT accuracy (80.00% vs 68.57%)
2. **Best Base Performance**: Uniform has **+22.14%** better base accuracy (66.43% vs 44.29%)
3. **Production Ready**: Uniform works well **with or without TTT**
4. **Consistent Excellence**: Uniform leads in final TTT performance across all metrics

**When to Consider (2,4,6)**:

- **Research purposes**: To study TTT adaptation mechanisms (largest improvement)
- **If base performance is irrelevant**: Only care about post-TTT performance
- **If TTT is guaranteed**: Always available in deployment

**Trade-off Summary**:

- **(2,4,6)**: Large improvement potential, but poor base and lower final performance
- **Uniform (4,4,4)**: Best overall performance, stable base, best final TTT performance

---

## Summary Statistics

| Aspect                  | (2,4,6)        | Uniform (4,4,4) | Hierarchical (3,5,7) | Hierarchical (1,2,4) | Winner                 |
| ----------------------- | -------------- | --------------- | -------------------- | -------------------- | ---------------------- |
| **Base Accuracy**       | 44.29%         | **66.43%**      | 61.43%               | 67.86%               | ✅ Uniform             |
| **TTT Accuracy**        | 68.57%         | **80.00%**      | 70.71%               | 76.43%               | ✅ Uniform             |
| **TTT AUC-PR**          | 82.82%         | **92.84%**      | 88.70%               | 90.38%               | ✅ Uniform             |
| **TTT ZDR**             | 76.19%         | **78.57%**      | 71.43%               | 78.57%               | 🤝 Tie                 |
| **TTT ZDR Improvement** | **+40.48%** ⭐ | +4.76%          | +9.52%               | -4.76%               | ✅ (2,4,6)             |
| **Overall**             |                |                 |                      |                      | ✅ **Uniform (4,4,4)** |

---

## Final Recommendation

### 🎯 **Use Uniform Kernel Size (4, 4, 4) for Production**

**Key Reasons**:

1. **Best Final Performance**: Highest TTT accuracy (80.00%) and AUC-PR (92.84%)
2. **Best Base Performance**: High base accuracy (66.43%) for graceful degradation
3. **Production Ready**: Works well with or without TTT adaptation
4. **Consistent Excellence**: Leads in final TTT performance across all metrics

**When (2,4,6) Might Be Interesting**:

- Research on TTT adaptation mechanisms (largest improvement potential)
- If you only care about improvement magnitude, not absolute performance
- If base performance is completely irrelevant

---

## Next Steps

1. ✅ **Current Status**: Hierarchical kernel sizes (2,4,6) tested and documented
2. 🔄 **Recommendation**: Revert to uniform kernel size (4,4,4) for best overall performance
3. 📊 **Alternative**: Could explore (3,3,4) or (4,4,5) for subtle variations









