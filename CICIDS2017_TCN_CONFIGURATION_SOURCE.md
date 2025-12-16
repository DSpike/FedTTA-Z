# CICIDS2017 TCN Configuration Source Analysis

## 🔍 **Current Configuration in `config_loader.py`**

```python
'CICIDS2017': {
    'sequence_length': 25,
    'sequence_stride': 15,
    'tcn_kernel_sizes': (3, 5, 7),
    # ...
}
```

## 📊 **Optimized Configuration from Optuna**

From `best_hyperparameters_cicids.json` (Optuna optimization results):

```json
{
  "sequence_length": 25, // ✅ MATCHES
  "sequence_stride": 12, // ❌ DIFFERENT (config has 15)
  "tcn_kernel_size_1": 3, // ✅ MATCHES Layer 1
  "tcn_kernel_size_2": 4, // ❌ DIFFERENT (config has 5)
  "tcn_kernel_size_3": 4 // ❌ DIFFERENT (config has 7)
}
```

**Optimized TCN Kernel Sizes**: `(3, 4, 4)`  
**Current Config TCN Kernel Sizes**: `(3, 5, 7)`

---

## ❓ **How Was It Decided?**

### **Honest Answer: I Used Estimated/Heuristic Values**

The current configuration in `config_loader.py` was **NOT** based on the Optuna optimization results. Instead, I used:

1. **Heuristic reasoning**:

   - **Sequence Length (25)**: Matched optimized value ✅
   - **Sequence Stride (15)**: Estimated based on sequence_length (typically 60% of sequence_length)
   - **TCN Kernel Sizes (3, 5, 7)**: Used a "hierarchical" pattern:
     - Layer 1: 3 (fine-scale patterns)
     - Layer 2: 5 (medium-scale patterns)
     - Layer 3: 7 (coarse-scale patterns)
     - This is a common pattern for multi-scale temporal feature extraction

2. **Comparison with other datasets**:

   - **KDD**: `(2, 3, 3)` - smaller kernels
   - **UNSW**: `(3, 3, 6)` - medium kernels
   - **CICIDS2017**: `(3, 5, 7)` - larger kernels (estimated for larger dataset)
   - **CICIDS2023**: `(3, 4, 5)` - medium-large kernels

3. **Reasoning for (3, 5, 7)**:
   - CICIDS2017 has **78 features** (larger than KDD's 41, UNSW's 43)
   - Larger dataset might benefit from larger receptive fields
   - Hierarchical pattern (3→5→7) captures multi-scale temporal patterns
   - **BUT**: This was **NOT validated** with actual optimization results

---

## ⚠️ **Discrepancy Found**

### **Optimized Values (from Optuna)**:

- `tcn_kernel_sizes: (3, 4, 4)`
- `sequence_stride: 12`

### **Current Config Values**:

- `tcn_kernel_sizes: (3, 5, 7)`
- `sequence_stride: 15`

**Difference**: The optimized values suggest:

- **Smaller kernels** in layers 2 and 3 (4, 4 instead of 5, 7)
- **Smaller stride** (12 instead of 15)

---

## 🎯 **Recommendation**

### **Option 1: Use Optimized Values (Recommended)**

```python
'CICIDS2017': {
    'sequence_length': 25,      # ✅ Already correct
    'sequence_stride': 12,      # ⚠️ Should be 12 (not 15)
    'tcn_kernel_sizes': (3, 4, 4),  # ⚠️ Should be (3, 4, 4) (not 3, 5, 7)
    # ...
}
```

**Rationale**: These values were **actually optimized** using Optuna, so they should perform better.

### **Option 2: Keep Current Values**

If you want to test the hierarchical (3, 5, 7) pattern, you can keep it, but it's **not based on optimization**.

---

## 📋 **Summary**

| Parameter          | Current Config | Optimized Value | Source                              |
| ------------------ | -------------- | --------------- | ----------------------------------- |
| `sequence_length`  | 25             | 25              | ✅ From Optuna                      |
| `sequence_stride`  | 15             | 12              | ❌ Estimated (should use 12)        |
| `tcn_kernel_sizes` | (3, 5, 7)      | (3, 4, 4)       | ❌ Heuristic (should use optimized) |

**Decision Method**:

- ❌ **NOT from Optuna optimization** (despite optimization results existing)
- ✅ **Heuristic/estimated** based on:
  - Dataset size (78 features)
  - Comparison with other datasets
  - Common multi-scale patterns

**Recommendation**: **Update to use optimized values** `(3, 4, 4)` and `stride: 12` for better performance.



