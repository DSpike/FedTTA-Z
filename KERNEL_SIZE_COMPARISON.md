# TCN Kernel Size Comparison: Uniform (4,4,4) vs Hierarchical (3,5,7)

## Configuration Comparison

### Uniform Kernel Size (4, 4, 4):

- **Branch 1**: kernel_size = 4, hidden_dim = 64
- **Branch 2**: kernel_size = 4, hidden_dim = 32
- **Branch 3**: kernel_size = 4, hidden_dim = 128
- **Padding**: All use padding = 2

### Hierarchical Kernel Size (3, 5, 7):

- **Branch 1**: kernel_size = 3, hidden_dim = 64 (fine-scale patterns)
- **Branch 2**: kernel_size = 5, hidden_dim = 32 (medium-scale patterns)
- **Branch 3**: kernel_size = 7, hidden_dim = 128 (coarse-scale patterns)
- **Padding**: Auto-calculated (padding = kernel_size // 2)

---

## Performance Comparison (Same Analysis Attack)

### Uniform Kernel Size (4, 4, 4) Results:

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

### Hierarchical Kernel Size (3, 5, 7) Results:

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

| Metric              | Uniform (4,4,4) | Hierarchical (3,5,7) | Difference  | Winner          |
| ------------------- | --------------- | -------------------- | ----------- | --------------- |
| **Base Model**      |                 |                      |             |                 |
| Accuracy            | 66.43%          | 61.43%               | **-5.00%**  | ✅ Uniform      |
| F1-Score            | 70.44%          | 61.43%               | **-9.01%**  | ✅ Uniform      |
| AUC-PR              | 81.24%          | 79.42%               | **-1.82%**  | ✅ Uniform      |
| ZDR                 | 73.81%          | 61.90%               | **-11.91%** | ✅ Uniform      |
|                     |                 |                      |             |                 |
| **TTT Model**       |                 |                      |             |                 |
| Accuracy            | **80.00%**      | 70.71%               | **-9.29%**  | ✅ Uniform      |
| F1-Score            | **82.28%**      | 72.11%               | **-10.17%** | ✅ Uniform      |
| AUC-PR              | **92.84%**      | 88.70%               | **-4.14%**  | ✅ Uniform      |
| ROC AUC             | **83.47%**      | 75.45%               | **-8.02%**  | ✅ Uniform      |
| ZDR                 | **78.57%**      | 71.43%               | **-7.14%**  | ✅ Uniform      |
| Precision           | 100.00%         | 100.00%              | 0.00%       | 🤝 Tie          |
| Recall              | 78.57%          | 71.43%               | **-7.14%**  | ✅ Uniform      |
|                     |                 |                      |             |                 |
| **TTT Improvement** |                 |                      |             |                 |
| Accuracy Gain       | +13.57%         | +9.28%               | **-4.29%**  | ✅ Uniform      |
| AUC-PR Gain         | +11.60%         | +9.28%               | **-2.32%**  | ✅ Uniform      |
| ZDR Gain            | +4.76%          | +9.52%               | **+4.76%**  | ✅ Hierarchical |

---

## Key Findings

### ⚠️ **Uniform Kernel Size (4,4,4) Performs Better**

1. **Better Overall Performance**:

   - **TTT Accuracy**: Uniform leads by **+9.29%** (80.00% vs 70.71%)
   - **TTT AUC-PR**: Uniform leads by **+4.14%** (92.84% vs 88.70%)
   - **TTT ZDR**: Uniform leads by **+7.14%** (78.57% vs 71.43%)
   - **TTT F1-Score**: Uniform leads by **+10.17%** (82.28% vs 72.11%)

2. **Better Base Model Performance**:

   - **Accuracy**: Uniform leads by **+5.00%**
   - **F1-Score**: Uniform leads by **+9.01%**
   - **ZDR**: Uniform leads by **+11.91%**

3. **Larger TTT Improvements**:
   - Uniform shows larger absolute improvements from TTT adaptation
   - **Accuracy Gain**: +13.57% vs +9.28%
   - **AUC-PR Gain**: +11.60% vs +9.28%

### ✅ **One Advantage of Hierarchical (3,5,7)**:

- **Larger ZDR Improvement**: Hierarchical shows +9.52% ZDR improvement vs +4.76% for uniform
- However, the absolute ZDR value is still lower (71.43% vs 78.57%)

---

## Analysis

### Why Uniform Kernel Size Performs Better:

1. **Consistent Temporal Receptive Field**:

   - All branches capture similar temporal patterns
   - Better feature alignment when concatenating branches
   - More consistent representation across scales

2. **Better Feature Fusion**:

   - Uniform kernel sizes may create more compatible feature representations
   - Hierarchical sizes may create incompatible feature scales that are harder to combine

3. **Optimal Kernel Size for This Task**:

   - Kernel size 4 may be the "sweet spot" for this sequence length (30)
   - Smaller (3) and larger (7) kernels may not capture optimal patterns

4. **Training Stability**:
   - Uniform configuration may be easier to train and optimize
   - Less variance in feature scales reduces optimization challenges

---

## Conclusion

### ✅ **Recommendation: Use Uniform Kernel Size (4, 4, 4)**

**Reasoning**:

1. **Better Performance**: Uniform kernel size outperforms hierarchical by ~4-10% across all metrics
2. **Better Base Model**: Base model performance is significantly better with uniform kernel size
3. **Better TTT Benefits**: Shows larger improvements from TTT adaptation
4. **More Stable**: Uniform configuration may be more stable and easier to optimize

**Trade-off Analysis**:

- Hierarchical kernel sizes theoretically should capture multi-scale patterns better
- However, in practice, uniform kernel size (4) performs better for this task
- The consistent receptive field appears to be more important than multi-scale diversity

---

## Summary Statistics

| Aspect           | Uniform (4,4,4) | Hierarchical (3,5,7) | Winner                 |
| ---------------- | --------------- | -------------------- | ---------------------- |
| **TTT Accuracy** | 80.00%          | 70.71%               | ✅ Uniform (+9.29%)    |
| **TTT AUC-PR**   | 92.84%          | 88.70%               | ✅ Uniform (+4.14%)    |
| **TTT ZDR**      | 78.57%          | 71.43%               | ✅ Uniform (+7.14%)    |
| **TTT F1-Score** | 82.28%          | 72.11%               | ✅ Uniform (+10.17%)   |
| **Base ZDR**     | 73.81%          | 61.90%               | ✅ Uniform (+11.91%)   |
| **Overall**      |                 |                      | ✅ **Uniform (4,4,4)** |

---

## Next Steps

1. ✅ **Current Status**: Hierarchical kernel sizes (3,5,7) tested and compared
2. 🔄 **Recommendation**: Revert to uniform kernel size (4,4,4) for better performance
3. 📊 **Alternative**: Could try other uniform sizes (3,3,3 or 5,5,5) to find optimal single kernel size









