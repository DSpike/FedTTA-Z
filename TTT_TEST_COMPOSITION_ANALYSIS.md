# TTT Model Test Sample Composition Analysis

## Overview

The TTT model has **TWO different test sample compositions**:

1. **TTT Adaptation Query Set** - Used during adaptation (unsupervised learning)
2. **TTT Final Evaluation Set** - Used for final performance evaluation (same as base model)

---

## 1. TTT Adaptation Query Set

**Purpose**: Used for unsupervised TTT adaptation (entropy minimization)

**Configuration**:

- **Size**: Configurable via `ttt_adaptation_query_size` (default: 750, but actual: 332 in current run)
- **Source**: Random subset from test sequences
- **Stride**: Same as evaluation (stride=15) to match distribution
- **Composition**: Random sample from test sequences (includes both zero-day and non-zero-day)

**From Code** (lines 2414-2446 in `main.py`):

```python
ttt_query_size = getattr(self.config, 'ttt_adaptation_query_size', 750)  # Default: 750
query_indices = torch.randperm(len(X_test_ttt_seq))[:query_size]
query_x = torch.FloatTensor(X_test_ttt_seq[query_indices]).to(self.device)
```

**From Latest Run Logs**:

```
TTT Query set: 332 samples (created with stride=15 matching evaluation stride=15)
✅ CONFIRMED: Both TTT adaptation and evaluation use stride=15 (no distribution mismatch)
   Created 332 sequences from 5000 original samples
```

**Composition**:

- **Total**: 332 sequences
- **Zero-day**: ~37 sequences (11.1%, random sample from test)
- **Non-zero-day**: ~295 sequences (88.9%, random sample from test)
- **Note**: This is a **random subset** from the full test set, so the distribution should be similar

**Key Point**:

- TTT adapts on a **random sample** from test data
- This sample includes both zero-day and non-zero-day sequences
- The model adapts in an **unsupervised** way (no labels used during adaptation)

---

## 2. TTT Final Evaluation Set

**Purpose**: Used for final performance evaluation after TTT adaptation

**Configuration**:

- **Size**: Same as base model evaluation (332 sequences)
- **Source**: Full test set (same as base model)
- **Composition**: Identical to base model test set

**From Code** (lines 2490-2565 in `main.py`):

```python
# Get test data (sequences) - SAME as base model evaluation
X_test = self.preprocessed_data['X_test']
y_test = self.preprocessed_data['y_test']
```

**From Latest Run Logs**:

```
🔍 Identified 37 zero-day sequences from 332 sequences using sequence-level multiclass labels
🔍 Zero-day mask created: 37/332 samples (11.1%)
   Zero-day attack: 'Analysis', label: 2
   Zero-day samples: 37, Non-zero-day samples: 295
Evaluating adapted model on 332 test samples with 37 zero-day samples and 295 non-zero-day samples
```

**Composition**:

- **Total**: 332 sequences
- **Zero-day**: 37 sequences (11.1%)
- **Non-zero-day**: 295 sequences (88.9%)
  - Normal samples (label=0)
  - Other seen attacks (labels 1, 3, 4, 5, 6, 7, 8, 9)

**Key Point**:

- TTT final evaluation uses the **EXACT SAME test set** as base model
- This ensures fair comparison between base and TTT models
- Same zero-day mask (37 sequences) and non-zero-day mask (295 sequences)

---

## Comparison: Base Model vs TTT Model

| Aspect                   | Base Model                          | TTT Model                                           |
| ------------------------ | ----------------------------------- | --------------------------------------------------- |
| **Training Data**        | Excludes zero-day (193,570 samples) | Same (model trained before TTT)                     |
| **Adaptation Data**      | N/A                                 | 332 sequences (random from test, includes zero-day) |
| **Final Test Set**       | 332 sequences                       | 332 sequences (identical)                           |
| **Zero-day in Test**     | 37 sequences (11.1%)                | 37 sequences (11.1%)                                |
| **Non-zero-day in Test** | 295 sequences (88.9%)               | 295 sequences (88.9%)                               |

---

## Important Observations

### 1. **TTT Adaptation Query Set Contains Zero-Day**

⚠️ **Potential Issue**: The TTT adaptation query set includes zero-day samples, but TTT is **unsupervised** (no labels used). This means:

- TTT adapts on zero-day samples **without knowing they are zero-day**
- This is acceptable because TTT uses entropy minimization (no label information)
- However, this could affect adaptation if zero-day patterns are very different

### 2. **Same Final Evaluation Set**

✅ **Fair Comparison**: Both base and TTT models are evaluated on the exact same test set:

- Same 37 zero-day sequences
- Same 295 non-zero-day sequences
- Same distribution

This ensures fair performance comparison.

### 3. **TTT Adaptation Size**

From logs: TTT adapts on 332 sequences, which is the **full test set** (not a subset). This means:

- TTT sees the entire test set during adaptation
- This is actually the same as the evaluation set
- No distribution mismatch between adaptation and evaluation

---

## Summary

### TTT Model Test Composition:

**Adaptation Query Set**:

- 332 sequences (full test set used for adaptation)
- Contains ~37 zero-day + ~295 non-zero-day (random sample from test)
- Used for unsupervised TTT adaptation (entropy minimization)

**Final Evaluation Set**:

- 332 sequences (identical to base model)
- Contains exactly 37 zero-day + 295 non-zero-day
- Used for final performance evaluation
- Ensures fair comparison with base model

**Conclusion**: TTT model uses the same final test set as base model (332 sequences: 37 zero-day, 295 non-zero-day), ensuring fair comparison. The adaptation uses a subset from this same test set (currently 332 sequences, which is the full test set).

