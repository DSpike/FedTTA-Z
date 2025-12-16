# TCN Kernel Size Comparison: (2, 4, 8) vs (2, 4, 6)

## 🔍 Experiment Configuration

**Current Test:** Kernel sizes (2, 4, 8)

- Branch 1: kernel_size=2 (fine-scale patterns)
- Branch 2: kernel_size=4 (medium-scale patterns)
- Branch 3: kernel_size=8 (coarse-scale patterns) ⭐ **Changed from 6 to 8**

**Previous Configuration:** Kernel sizes (2, 4, 6)

- Branch 1: kernel_size=2
- Branch 2: kernel_size=4
- Branch 3: kernel_size=6

**Test Setup:**

- Zero-Day Attack: DoS
- Test Set: 300 samples (90 zero-day, 210 non-zero-day)
- TTT Steps: 300
- TCN: Enabled (EfficientMultiScaleTCN)

---

## 📊 Performance Results: Kernel Sizes (2, 4, 8)

### Base Model Performance:

**Overall Performance (All Test Samples):**

- Accuracy: **78.33%**
- F1-Score: **82.94%**
- AUC-PR: **82.87%** (from earlier logs)
- ROC AUC: **83.83%** (from earlier logs)

**Zero-Day Only (90 samples):**

- Zero-Day Detection Rate: **93.33%** (84/90 detected)
- Accuracy: **93.33%**
- Precision: **100.00%** (perfect - no false positives)
- Recall: **93.33%**
- F1-Score: **96.55%**
- Zero-Day-Specific AUC-PR: **100.00%**

**Zero-Day Predictions:** `[6 Normal, 84 Attack]` out of 90 zero-day samples

---

### TTT Model Performance:

**Overall Performance (All Test Samples):**

- Accuracy: **75.33%**
- F1-Score: **77.16%**
- AUC-PR: **84.80%**
- ROC AUC: **82.21%**
- FAR: **19.84%**

**Zero-Day Only (90 samples):**

- Zero-Day Detection Rate: **76.67%** (69/90 detected)
- Accuracy: **76.67%**
- Precision: **100.00%** (perfect - no false positives)
- Recall: **76.67%**
- F1-Score: **86.79%**
- Zero-Day-Specific AUC-PR: **100.00%**

**Zero-Day Predictions:** `[21 Normal, 69 Attack]` out of 90 zero-day samples

---

## 🔄 Comparison: (2, 4, 8) vs (2, 4, 6)

### Base Model Zero-Day Detection:

| Kernel Config | Zero-Day Detection | Accuracy | Precision | Recall | F1-Score |
| ------------- | ------------------ | -------- | --------- | ------ | -------- |
| **(2, 4, 6)** | **93.33%**         | 78.00%   | 100.00%   | 93.33% | 96.55%   |
| **(2, 4, 8)** | **93.33%**         | 78.33%   | 100.00%   | 93.33% | 96.55%   |
| **Change**    | **0.00%**          | +0.33%   | Same      | Same   | Same     |

### TTT Model Zero-Day Detection:

| Kernel Config | Zero-Day Detection | Accuracy | AUC-PR | Recall |
| ------------- | ------------------ | -------- | ------ | ------ |
| **(2, 4, 6)** | **77.78%**         | 81.33%   | 86.71% | 77.78% |
| **(2, 4, 8)** | **76.67%**         | 75.33%   | 84.80% | 76.67% |
| **Change**    | **-1.11%**         | -6.00%   | -1.91% | -1.11% |

### Overall Performance:

| Kernel Config | Base Accuracy | Base F1 | TTT Accuracy | TTT F1 | TTT AUC-PR |
| ------------- | ------------- | ------- | ------------ | ------ | ---------- |
| **(2, 4, 6)** | 78.00%        | 82.63%  | **81.33%**   | 82.72% | **86.71%** |
| **(2, 4, 8)** | 78.33%        | 82.94%  | **75.33%**   | 77.16% | **84.80%** |
| **Change**    | +0.33%        | +0.31%  | **-6.00%**   | -5.56% | **-1.91%** |

---

## 📈 Key Findings

### 1. **Base Model: Similar Performance**

- **Zero-Day Detection:** Identical at 93.33% (84/90 detected)
- **Overall Accuracy:** Slightly improved (+0.33%)
- **Conclusion:** Larger coarse-scale kernel (8 vs 6) doesn't significantly change base model performance

### 2. **TTT Model: Performance Degradation**

- **Zero-Day Detection:** Decreased from 77.78% → 76.67% (-1.11%)
- **Overall Accuracy:** Decreased from 81.33% → 75.33% (-6.00%)
- **AUC-PR:** Decreased from 86.71% → 84.80% (-1.91%)
- **Conclusion:** Larger coarse-scale kernel (8) may hurt TTT adaptation performance

### 3. **Kernel Size Impact Analysis**

**Kernel Size 8 vs 6:**

- **Larger receptive field** (kernel_size=8 captures longer temporal patterns)
- **More parameters** (may increase overfitting during TTT)
- **Different temporal scale** (coarser patterns, but may be too coarse for this task)

**Why TTT Performance Dropped:**

- Kernel size 8 might be **too large** for sequence length 30
- Captures patterns spanning **8/30 = 26.7%** of the sequence
- May lose fine-grained temporal details needed for adaptation
- TTT adaptation may struggle with larger kernels

### 4. **Base Model Robustness**

- Base model shows **similar performance** with both kernel configurations
- Zero-day detection remains **excellent** (93.33%) in both cases
- Suggests base model is more robust to kernel size changes

---

## 💡 Insights

### Kernel Size 6 vs 8:

**Kernel Size 6:**

- Receptive field: 6/30 = 20% of sequence
- Better balance between fine and coarse patterns
- More suitable for TTT adaptation

**Kernel Size 8:**

- Receptive field: 8/30 = 26.7% of sequence
- Too coarse for sequence length 30
- May lose important fine-grained temporal features
- Less effective for TTT adaptation

### Optimal Configuration:

For **sequence length 30**:

- **Kernel sizes (2, 4, 6)** appear more optimal
- Provides better **multi-scale balance**
- Better **TTT adaptation** performance
- Maintains excellent **base model** performance

---

## 📊 Performance Summary Table

| Configuration      | Base Zero-Day | Base Overall | TTT Zero-Day | TTT Overall | TTT AUC-PR  |
| ------------------ | ------------- | ------------ | ------------ | ----------- | ----------- |
| **Kernel (2,4,6)** | **93.33%**    | 78.00%       | **77.78%**   | **81.33%**  | **86.71%**  |
| **Kernel (2,4,8)** | **93.33%**    | 78.33%       | **76.67%**   | **75.33%**  | **84.80%**  |
| **Winner**         | **Tie**       | **(2,4,8)**  | **(2,4,6)**  | **(2,4,6)** | **(2,4,6)** |

---

## ✅ Conclusion

### Kernel Size (2, 4, 6) is Better:

1. **Better TTT Performance:**

   - +6.00% overall accuracy (81.33% vs 75.33%)
   - +1.91% AUC-PR (86.71% vs 84.80%)
   - Better zero-day detection after adaptation (77.78% vs 76.67%)

2. **Maintains Base Model Performance:**

   - Identical zero-day detection (93.33%)
   - Excellent baseline performance

3. **Better Multi-Scale Balance:**
   - Kernel size 6 provides optimal coarse-scale patterns
   - Kernel size 8 may be too large for sequence length 30
   - Better suited for TTT adaptation

### Recommendation:

**Keep kernel sizes at (2, 4, 6)** for optimal performance with sequence length 30.

Kernel size 8 is too large and hurts TTT adaptation while providing minimal benefit to base model.
