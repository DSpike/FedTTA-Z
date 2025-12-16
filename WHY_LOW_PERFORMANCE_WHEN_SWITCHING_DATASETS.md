# Why Performance is Low When Switching Datasets

## 🔍 **Root Causes Identified**

Based on your recent run with UNSW-NB15 dataset using KDD-optimized hyperparameters:

---

## ❌ **Critical Issues**

### **1. Confidence Rejection Threshold Too High** ⚠️ **PRIMARY ISSUE**

**Current Setting:**

```python
confidence_rejection_threshold: float = 0.90  # 90% confidence required
```

**Impact from Your Run:**

- **741/756 samples rejected** (98.1% rejection rate!)
- Only **13-15 samples** used for evaluation
- **Zero-day detection rate: 0.0000** (0% - no samples to evaluate!)

**Why This Happens:**

- Threshold optimized for KDD dataset (where model is more confident)
- UNSW dataset has different feature distribution → model less confident
- Model trained on KDD patterns → uncertain on UNSW patterns

**Fix:**

```python
confidence_rejection_threshold: float = 0.70  # Lower to 70% for UNSW
# Or even 0.60 for more coverage
```

---

### **2. Hyperparameters Optimized for KDD, Not UNSW** ⚠️ **MAJOR ISSUE**

**Current Settings (KDD-Optimized):**

```python
sequence_length: int = 22      # KDD-optimized
sequence_stride: int = 12      # KDD-optimized
tcn_kernel_sizes: tuple = (2, 3, 3)  # KDD-optimized
hidden_dim: int = 128          # KDD-optimized
embedding_dim: int = 256       # KDD-optimized
meta_epochs: int = 21          # KDD-optimized
k_shot: int = 152              # KDD-optimized
n_query: int = 16              # KDD-optimized
learning_rate: float = 0.0016  # KDD-optimized
```

**UNSW-Optimized Settings (from unsw-nb15-version branch):**

```python
sequence_length: int = 21      # UNSW-optimized
sequence_stride: int = 13      # UNSW-optimized
tcn_kernel_sizes: tuple = (3, 3, 6)  # UNSW-optimized
hidden_dim: int = 256          # UNSW-optimized (2x larger!)
embedding_dim: int = 128       # UNSW-optimized (2x smaller!)
meta_epochs: int = 18          # UNSW-optimized
k_shot: int = 118              # UNSW-optimized
n_query: int = 20              # UNSW-optimized
learning_rate: float = 0.0011  # UNSW-optimized
```

**Impact:**

- **Hidden dimension mismatch**: 128 (KDD) vs 256 (UNSW) - **2x difference!**
- **Embedding dimension mismatch**: 256 (KDD) vs 128 (UNSW) - **2x difference!**
- **TCN kernel sizes**: (2,3,3) vs (3,3,6) - different receptive fields
- **Sequence length**: 22 vs 21 - minor but affects temporal patterns

**Result:**

- Model architecture not optimal for UNSW data
- Feature extraction may miss important patterns
- Lower learning capacity (smaller hidden_dim)

---

### **3. Input Dimension Verification Needed**

**Current:**

```python
input_dim: int = 43  # UNSW-NB15
```

**Should Verify:**

- UNSW-NB15 after feature engineering: 45-48 features
- After feature selection: 43 features (if enabled)
- **Need to check actual feature count** after preprocessing

**Risk:**

- If actual features ≠ 43, model will fail or use wrong architecture
- Feature mismatch causes poor performance

---

### **4. Attack Types Dictionary Mismatch**

**Current (UNSW):**

```python
attack_types = {
    'Normal': 0,
    'DoS': 4,  # Label 4
    ...
}
```

**Issue:**

- Zero-day attack "DoS" is label 4 in UNSW
- But model was trained/optimized for KDD attack patterns
- Different attack representations → poor generalization

---

## 📊 **Performance Impact Analysis**

### **From Your Recent Run:**

| Metric                      | Value  | Issue                   |
| --------------------------- | ------ | ----------------------- |
| **Base Accuracy**           | 60.00% | Low (should be 75-85%+) |
| **TTT Accuracy**            | 53.85% | Lower than base!        |
| **Zero-Day Detection Rate** | 0.00%  | **CRITICAL: 0%!**       |
| **F1-Score**                | 0.00%  | **CRITICAL: 0%!**       |
| **Samples Used**            | 13/756 | **98% rejected!**       |

### **Why Zero-Day Detection is 0%:**

1. **Confidence threshold too high** (0.90)
2. **All zero-day samples rejected** (confidence < 0.90)
3. **No samples to evaluate** → ZDR = 0%

---

## ✅ **Solutions**

### **Solution 1: Lower Confidence Threshold (IMMEDIATE FIX)**

```python
# config.py
confidence_rejection_threshold: float = 0.70  # Lower from 0.90 to 0.70
```

**Expected Impact:**

- ✅ More samples available for evaluation
- ✅ Zero-day samples included
- ✅ Better performance metrics

---

### **Solution 2: Use UNSW-Optimized Hyperparameters**

```python
# config.py - Update to UNSW-optimized values
hidden_dim: int = 256          # Increase from 128
embedding_dim: int = 128       # Decrease from 256
sequence_length: int = 21      # Decrease from 22
sequence_stride: int = 13       # Increase from 12
tcn_kernel_sizes: tuple = (3, 3, 6)  # Change from (2, 3, 3)
meta_epochs: int = 18          # Decrease from 21
k_shot: int = 118              # Decrease from 152
n_query: int = 20              # Increase from 16
learning_rate: float = 0.0011  # Decrease from 0.0016
```

**Expected Impact:**

- ✅ Better feature extraction for UNSW patterns
- ✅ More appropriate model capacity
- ✅ Better temporal pattern recognition
- ✅ +10-20% performance improvement

---

### **Solution 3: Verify Input Dimension**

```python
# After preprocessing, check actual feature count
actual_input_dim = preprocessed_data['X_train'].shape[1]
if actual_input_dim != config.input_dim:
    logger.warning(f"Input dimension mismatch: config={config.input_dim}, actual={actual_input_dim}")
    # Update model architecture
```

---

### **Solution 4: Re-optimize for UNSW (BEST LONG-TERM)**

Run Optuna optimization specifically for UNSW dataset:

```bash
python optimize_hyperparameters.py --dataset UNSW
```

**Benefits:**

- ✅ Hyperparameters optimized for UNSW
- ✅ Best possible performance
- ✅ Dataset-specific tuning

---

## 🎯 **Quick Fix (Apply Now)**

**Priority 1: Lower Confidence Threshold**

```python
confidence_rejection_threshold: float = 0.70  # Change from 0.90
```

**Priority 2: Update Key Hyperparameters**

```python
hidden_dim: int = 256          # UNSW-optimized
embedding_dim: int = 128       # UNSW-optimized
tcn_kernel_sizes: tuple = (3, 3, 6)  # UNSW-optimized
```

**Priority 3: Verify Input Dimension**

- Check logs after preprocessing
- Ensure `input_dim` matches actual feature count

---

## 📊 **Expected Performance After Fixes**

### **Before (Current):**

- Accuracy: 60%
- ZDR: 0%
- Samples: 13/756 (98% rejected)

### **After Fixes:**

- Accuracy: **75-85%** (expected)
- ZDR: **60-80%** (expected)
- Samples: **500-700/756** (reasonable rejection)

---

## 💡 **Key Takeaway**

**The main issue is NOT the single config file approach** - it's that:

1. **Confidence threshold is too high** (rejecting 98% of samples!)
2. **Hyperparameters are optimized for KDD, not UNSW**
3. **Model architecture doesn't match UNSW data characteristics**

**Fix these three issues and performance should improve significantly!** ✅



