# TCN Kernel Size Comparison: (1,2,4) vs Previous Configurations

## Configuration Overview

### New Configuration: Hierarchical Kernel Sizes (1, 2, 4)

- **Branch 1**: kernel_size = 1, hidden_dim = 64 (Pointwise - no temporal context)
- **Branch 2**: kernel_size = 2, hidden_dim = 32 (Very fine-scale patterns)
- **Branch 3**: kernel_size = 4, hidden_dim = 128 (Medium-scale patterns)
- **Padding**: Auto-calculated (padding = kernel_size // 2)

### Previous Configurations:

- **Uniform (4, 4, 4)**: All branches use kernel_size = 4
- **Hierarchical (3, 5, 7)**: Fine (3), medium (5), coarse (7) scales

---

## Performance Comparison (Analysis Attack, Same Test Set)

### New Configuration: Hierarchical Kernel Sizes (1, 2, 4) Results:

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

---

## Direct Comparison Table

| Metric              | (1,2,4)    | Uniform (4,4,4) | Hierarchical (3,5,7) | vs Uniform | vs (3,5,7)  | Winner                    |
| ------------------- | ---------- | --------------- | -------------------- | ---------- | ----------- | ------------------------- |
| **Base Model**      |            |                 |                      |            |             |                           |
| Accuracy            | 67.86%     | 66.43%          | 61.43%               | **+1.43%** | **+6.43%**  | ✅ (1,2,4)                |
| F1-Score            | 73.05%     | 70.44%          | 61.43%               | **+2.61%** | **+11.62%** | ✅ (1,2,4)                |
| AUC-PR              | 81.02%     | 81.24%          | 79.42%               | -0.22%     | **+1.60%**  | ✅ Uniform                |
| ZDR                 | **83.33%** | 73.81%          | 61.90%               | **+9.52%** | **+21.43%** | ✅ (1,2,4)                |
|                     |            |                 |                      |            |             |                           |
| **TTT Model**       |            |                 |                      |            |             |                           |
| Accuracy            | 76.43%     | **80.00%**      | 70.71%               | -3.57%     | **+5.72%**  | ✅ Uniform                |
| F1-Score            | 79.75%     | **82.28%**      | 72.11%               | -2.53%     | **+7.64%**  | ✅ Uniform                |
| AUC-PR              | 90.38%     | **92.84%**      | 88.70%               | -2.46%     | **+1.68%**  | ✅ Uniform                |
| ROC AUC             | 81.00%     | **83.47%**      | 75.45%               | -2.47%     | **+5.55%**  | ✅ Uniform                |
| ZDR                 | 78.57%     | 78.57%          | 71.43%               | 0.00%      | **+7.14%**  | 🤝 Tie (1,2,4) & Uniform  |
| Precision           | 100.00%    | 100.00%         | 100.00%              | 0.00%      | 0.00%       | 🤝 Tie                    |
| Recall              | 78.57%     | 78.57%          | 71.43%               | 0.00%      | **+7.14%**  | 🤝 Tie (1,2,4) & Uniform  |
|                     |            |                 |                      |            |             |                           |
| **TTT Improvement** |            |                 |                      |            |             |                           |
| Accuracy Gain       | +8.57%     | **+13.57%**     | +9.28%               | -5.00%     | -0.71%      | ✅ Uniform                |
| AUC-PR Gain         | +9.36%     | **+11.60%**     | +9.28%               | -2.24%     | +0.08%      | ✅ Uniform                |
| ZDR Gain            | -4.76%     | +4.76%          | **+9.52%**           | -9.52%     | -14.28%     | ⚠️ (1,2,4) shows decrease |

---

## Key Findings

### ✅ **Advantages of (1,2,4) Configuration:**

1. **Best Base Model ZDR**:

   - **83.33%** vs 73.81% (Uniform) and 61.90% (3,5,7)
   - **+9.52%** improvement over uniform
   - **+21.43%** improvement over (3,5,7)

2. **Best Base Model Performance**:

   - **Accuracy**: 67.86% (best among three)
   - **F1-Score**: 73.05% (best among three)
   - Better initial model performance before TTT adaptation

3. **Tied TTT ZDR**:
   - **78.57%** (same as uniform, better than 3,5,7)
   - Perfect precision (100%) maintained across all configurations

### ⚠️ **Disadvantages of (1,2,4) Configuration:**

1. **Lower TTT Accuracy & AUC-PR**:

   - **TTT Accuracy**: 76.43% vs 80.00% (Uniform) - **-3.57%**
   - **TTT AUC-PR**: 90.38% vs 92.84% (Uniform) - **-2.46%**
   - **TTT F1-Score**: 79.75% vs 82.28% (Uniform) - **-2.53%**

2. **Negative TTT ZDR Improvement**:

   - Base ZDR: **83.33%** (best)
   - TTT ZDR: **78.57%** (same as uniform)
   - **ZDR actually decreased** after TTT adaptation (-4.76%)
   - This suggests TTT adaptation may have **overfitted** or **overcorrected** the model

3. **Smaller TTT Improvements**:
   - Accuracy Gain: +8.57% vs +13.57% (Uniform)
   - AUC-PR Gain: +9.36% vs +11.60% (Uniform)

---

## Analysis

### Why (1,2,4) Has Better Base Model but Worse TTT Performance:

1. **Pointwise Branch (kernel=1)**:

   - Captures **instant features** (no temporal context)
   - May be good for initial detection (base model)
   - But lacks temporal patterns that TTT could leverage

2. **Temporal Context Mismatch**:

   - Branch 1: No temporal context (kernel=1)
   - Branch 2: Very limited context (kernel=2)
   - Branch 3: Medium context (kernel=4)
   - This **imbalanced temporal representation** may cause TTT to struggle with adaptation

3. **Overfitting to Base Distribution**:
   - High base ZDR (83.33%) suggests the model is already well-calibrated
   - TTT adaptation may have **pushed the model away** from optimal base performance
   - The pointwise branch may have captured **dataset-specific patterns** that don't generalize

### Comparison with Uniform (4,4,4):

- **Uniform is better for TTT**: More consistent temporal patterns allow TTT to adapt better
- **(1,2,4) is better for base**: Pointwise branch provides instant feature detection
- **Trade-off**: Better base performance vs better TTT improvement

### Comparison with Hierarchical (3,5,7):

- **(1,2,4) outperforms (3,5,7)** across all metrics
- Shows that **smaller kernel sizes** (1,2,4) are better than larger (3,5,7) for this task
- However, still falls short of uniform (4,4,4) for TTT adaptation

---

## Conclusion

### ✅ **Recommendation: Keep Uniform Kernel Size (4, 4, 4)**

**Reasoning**:

1. **Better TTT Performance**: Uniform (4,4,4) achieves **+3.57%** better TTT accuracy and **+2.46%** better AUC-PR
2. **Consistent TTT Improvement**: Uniform shows positive ZDR improvement (+4.76%) while (1,2,4) shows negative (-4.76%)
3. **Better Overall**: Uniform has better balance between base and TTT performance
4. **More Stable**: Uniform configuration is more predictable and stable

**Trade-off Analysis**:

- **(1,2,4)** has **better base model** (especially ZDR: 83.33% vs 73.81%)
- But **worse TTT adaptation** (lower accuracy, lower AUC-PR, negative ZDR improvement)
- The **negative ZDR improvement** (-4.76%) is a **red flag** suggesting TTT is hurting performance
- **Uniform (4,4,4)** has better **overall end-to-end performance** (base → TTT)

---

## Summary Statistics

| Aspect                  | (1,2,4)       | Uniform (4,4,4) | Hierarchical (3,5,7) | Winner                 |
| ----------------------- | ------------- | --------------- | -------------------- | ---------------------- |
| **Base ZDR**            | **83.33%**    | 73.81%          | 61.90%               | ✅ (1,2,4)             |
| **TTT Accuracy**        | 76.43%        | **80.00%**      | 70.71%               | ✅ Uniform             |
| **TTT AUC-PR**          | 90.38%        | **92.84%**      | 88.70%               | ✅ Uniform             |
| **TTT ZDR**             | 78.57%        | **78.57%**      | 71.43%               | 🤝 Tie                 |
| **TTT ZDR Improvement** | **-4.76%** ⚠️ | **+4.76%** ✅   | +9.52%               | ✅ Uniform             |
| **Overall**             |               |                 |                      | ✅ **Uniform (4,4,4)** |

---

## Final Recommendation

### 🎯 **Use Uniform Kernel Size (4, 4, 4)**

**Key Reasons**:

1. **Best TTT Performance**: Highest accuracy (80.00%) and AUC-PR (92.84%)
2. **Consistent Improvement**: Positive ZDR improvement (+4.76%)
3. **Best Balance**: Good base performance with excellent TTT adaptation
4. **Most Stable**: Predictable and reliable performance

**When to Consider (1,2,4)**:

- If **base model performance** is more important than TTT adaptation
- If you need **instant feature detection** (pointwise branch advantage)
- If you plan to use the model **without TTT adaptation**

---

## Next Steps

1. ✅ **Current Status**: Hierarchical kernel sizes (1,2,4) tested and documented
2. 🔄 **Recommendation**: Revert to uniform kernel size (4,4,4) for best TTT performance
3. 📊 **Alternative**: Could explore (2,2,4) or (3,3,4) to balance temporal patterns









