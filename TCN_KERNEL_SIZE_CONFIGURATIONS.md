# TCN Kernel Size Configurations Used

## Current Configuration (Active)

### In `config.py`:

```python
tcn_kernel_sizes: tuple = (2, 3, 3)  # Optimized from Optuna Trial 1
```

**Status**: Currently active in the codebase

- **Layer 1**: kernel_size = 2 (fine-scale patterns)
- **Layer 2**: kernel_size = 3 (medium-scale patterns)
- **Layer 3**: kernel_size = 3 (medium-scale patterns)

---

## Optuna Optimized Configuration

### From `best_hyperparameters.json` (Trial 7):

```json
"tcn_kernel_size_1": 4,
"tcn_kernel_size_2": 3,
"tcn_kernel_size_3": 3
```

**Status**: Best trial from Optuna optimization

- **Layer 1**: kernel_size = 4
- **Layer 2**: kernel_size = 3
- **Layer 3**: kernel_size = 3
- **Note**: This differs from config.py - there may be a mismatch

**Best Trial Metrics**:

- Balanced Base+TTT Score: 0.7494
- Base F1: 0.8211
- TTT F1: 0.8357
- Base ZDR: 0.9071
- TTT ZDR: 0.9007

---

## Optuna Search Space

### In `optimize_hyperparameters.py`:

```python
tcn_kernel_size_1 = trial.suggest_int("tcn_kernel_size_1", 2, 6)
tcn_kernel_size_2 = trial.suggest_int("tcn_kernel_size_2", 2, 6)
tcn_kernel_size_3 = trial.suggest_int("tcn_kernel_size_3", 2, 6)
```

**Search Range**: Each kernel size can be between **2 and 6** (inclusive)

---

## Previously Tested Configurations

### 1. **Uniform Configuration: (4, 4, 4)**

- **Layer 1**: kernel_size = 4
- **Layer 2**: kernel_size = 4
- **Layer 3**: kernel_size = 4
- **Status**: Tested, not optimal

**Performance** (Analysis Attack):

- Base Model: Acc=66.43%, F1=70.44%, ZDR=73.81%
- TTT Model: Acc=80.00%, F1=82.28%, ZDR=78.57%

---

### 2. **Hierarchical Configuration: (3, 5, 7)**

- **Layer 1**: kernel_size = 3 (fine-scale)
- **Layer 2**: kernel_size = 5 (medium-scale)
- **Layer 3**: kernel_size = 7 (coarse-scale)
- **Status**: Tested, not optimal

**Performance** (Analysis Attack):

- Base Model: Acc=61.43%, F1=61.43%, ZDR=61.90%
- TTT Model: Acc=70.71%, F1=72.11%, ZDR=71.43%

**Result**: Worse than uniform (4,4,4)

---

### 3. **Hierarchical Configuration: (1, 2, 4)**

- **Layer 1**: kernel_size = 1 (pointwise - no temporal context)
- **Layer 2**: kernel_size = 2 (very fine-scale)
- **Layer 3**: kernel_size = 4 (medium-scale)
- **Status**: Tested, not optimal

**Performance** (Analysis Attack):

- Base Model: Acc=67.86%, F1=73.05%, ZDR=83.33%
- TTT Model: Acc=76.43%, F1=79.75%, ZDR=78.57%

**Result**: Better base model ZDR but worse TTT performance

---

### 4. **Hierarchical Configuration: (2, 4, 6)**

- **Layer 1**: kernel_size = 2 (fine-scale)
- **Layer 2**: kernel_size = 4 (medium-scale)
- **Layer 3**: kernel_size = 6 (coarse-scale)
- **Status**: Tested, found to be better than (2,4,8)

**Performance** (DoS Attack, sequence_length=30):

- Base Model: ZDR=93.33%, Overall Acc=78.00%
- TTT Model: ZDR=77.78%, Overall Acc=81.33%, AUC-PR=86.71%

**Result**: Better TTT performance than (2,4,8)

---

### 5. **Hierarchical Configuration: (2, 4, 8)**

- **Layer 1**: kernel_size = 2 (fine-scale)
- **Layer 2**: kernel_size = 4 (medium-scale)
- **Layer 3**: kernel_size = 8 (coarse-scale)
- **Status**: Tested, found to be worse than (2,4,6)

**Performance** (DoS Attack, sequence_length=30):

- Base Model: ZDR=93.33%, Overall Acc=78.33%
- TTT Model: ZDR=76.67%, Overall Acc=75.33%, AUC-PR=84.80%

**Result**: Kernel size 8 too large for sequence_length=30 (26.7% of sequence)

**Insight**: Kernel size 8 captures too coarse patterns, losing fine-grained temporal details needed for TTT adaptation.

---

## Summary Table

| Configuration             | Layer 1 | Layer 2 | Layer 3 | Status         | Best For             |
| ------------------------- | ------- | ------- | ------- | -------------- | -------------------- |
| **Current (config.py)**   | 2       | 3       | 3       | ✅ Active      | General use          |
| **Optuna Best (Trial 7)** | 4       | 3       | 3       | 📊 Optimized   | Balanced performance |
| **Uniform (4,4,4)**       | 4       | 4       | 4       | ⚠️ Tested      | Baseline             |
| **Hierarchical (3,5,7)**  | 3       | 5       | 7       | ⚠️ Tested      | Not optimal          |
| **Hierarchical (1,2,4)**  | 1       | 2       | 4       | ⚠️ Tested      | Base model ZDR       |
| **Hierarchical (2,4,6)**  | 2       | 4       | 6       | ✅ Recommended | TTT performance      |
| **Hierarchical (2,4,8)**  | 2       | 4       | 8       | ❌ Too large   | Not recommended      |

---

## Recommendations

### For Sequence Length 30:

- **Best**: (2, 4, 6) - Better TTT adaptation performance
- **Alternative**: (2, 3, 3) - Current config, more conservative

### For Sequence Length 20-50 (Optuna optimized):

- **Best**: (4, 3, 3) - From Optuna Trial 7
- **Alternative**: (2, 3, 3) - Current config

### General Guidelines:

1. **Layer 1 (Fine-scale)**: 2-4 (captures immediate temporal patterns)
2. **Layer 2 (Medium-scale)**: 3-5 (captures short-term patterns)
3. **Layer 3 (Coarse-scale)**: 3-6 (captures longer-term patterns, but not too large)

**Avoid**:

- Kernel size > 6 for sequence_length ≤ 30 (too coarse)
- Kernel size = 1 (no temporal context)
- All layers same size (loses multi-scale benefit)

---

## Configuration Mismatch Note

⚠️ **Warning**: There's a discrepancy between:

- `config.py`: `(2, 3, 3)` - says "Optimized from Optuna Trial 1"
- `best_hyperparameters.json`: `(4, 3, 3)` - from Trial 7

**Recommendation**: Update `config.py` to match the best trial or verify which configuration is actually being used.



