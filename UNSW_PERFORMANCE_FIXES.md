# UNSW-NB15 Performance Fixes - Action Plan

## 🔍 **Current Performance Analysis**

### **From Latest Results:**

- **Base Model**: 70.49% accuracy, 74.63% F1-score
- **TTT Model**: 85.96% accuracy, 92.45% F1-score
- **Zero-Day Detection Rate**: 85.96% (good!)

### **Issues:**

1. ⚠️ **Base model performance is low** (70% vs expected 75-85%)
2. ✅ **TTT model is acceptable** (86% accuracy)
3. ⚠️ **Non-zero-day performance is low** (65.19% accuracy)

---

## ❌ **Root Causes Identified**

### **1. Training Configuration Too Conservative** ⚠️ **PRIMARY ISSUE**

**Current Settings:**

```python
meta_epochs: int = 18  # Too low for UNSW complexity
k_shot: int = 118  # Too low for sufficient support
learning_rate: float = 0.001096821720752952  # Too conservative
```

**Problem:**

- **18 epochs** may not be enough for UNSW's complex feature interactions
- **118 k-shot** may provide insufficient support samples for meta-learning
- **0.0011 learning rate** may cause slow convergence

**Impact:**

- Model doesn't fully learn UNSW patterns
- Underfitting due to insufficient training
- Base model performance suffers

---

### **2. Zero-Day Attack Label Comment Error** ✅ **FIXED**

**Issue:**

- Comment said "label 4" but Backdoor is actually label 3
- **Fixed**: Updated comment to correct label

---

### **3. Model Architecture May Need Adjustment** ⚠️ **MEDIUM PRIORITY**

**Current:**

```python
hidden_dim: int = 256  # UNSW-optimized
embedding_dim: int = 128  # UNSW-optimized
sequence_length: int = 21  # UNSW-optimized
tcn_kernel_sizes: tuple = (3, 3, 6)  # UNSW-optimized
```

**Potential Issues:**

- TCN kernel sizes (3, 3, 6) might not capture optimal patterns
- Sequence length 21 might be suboptimal
- Hidden dimension 256 might need adjustment

---

## ✅ **Recommended Fixes**

### **Fix 1: Increase Training Intensity** ⚠️ **HIGH PRIORITY**

```python
# config.py
meta_epochs: int = 25  # Increase from 18 (39% increase)
k_shot: int = 150  # Increase from 118 (27% increase)
learning_rate: float = 0.0015  # Increase from 0.0011 (36% increase)
```

**Expected Impact:**

- Better convergence → +5-10% base model accuracy
- More support samples → better meta-learning
- Faster learning → better feature extraction

---

### **Fix 2: Verify Input Dimension** ⚠️ **HIGH PRIORITY**

The system already checks input dimension (main.py lines 923-927), but we should verify it's working:

```python
# Add explicit logging
actual_input_dim = self.preprocessed_data['X_train'].shape[1]
logger.info(f"🔍 Input dimension check: Config={self.config.input_dim}, Actual={actual_input_dim}")
if actual_input_dim != self.config.input_dim:
    logger.warning(f"⚠️  Input dimension mismatch! Updating model architecture...")
```

---

### **Fix 3: Adjust TCN Configuration** ⚠️ **MEDIUM PRIORITY**

Try different TCN kernel sizes:

```python
# Option 1: Larger kernels for better temporal patterns
tcn_kernel_sizes: tuple = (3, 5, 7)  # Instead of (3, 3, 6)

# Option 2: Longer sequences
sequence_length: int = 25  # Instead of 21
```

---

### **Fix 4: Increase Meta-Learning Query Size** ⚠️ **LOW PRIORITY**

```python
n_query: int = 25  # Increase from 20 (25% increase)
```

---

## 🎯 **Implementation Priority**

### **Priority 1 (Do Now):**

1. ✅ Fix zero-day attack label comment
2. ⚠️ Increase `meta_epochs` to 25
3. ⚠️ Increase `k_shot` to 150
4. ⚠️ Increase `learning_rate` to 0.0015

### **Priority 2 (If Priority 1 doesn't help):**

1. Adjust TCN kernel sizes
2. Increase sequence length
3. Verify input dimension logging

### **Priority 3 (Long-term):**

1. Run Optuna optimization for UNSW
2. Experiment with different architectures
3. Fine-tune hyperparameters

---

## 📊 **Expected Performance After Fixes**

### **Current:**

- Base: 70% accuracy
- TTT: 86% accuracy
- Non-zero-day: 65% accuracy

### **Expected After Priority 1 Fixes:**

- Base: **75-80%** accuracy (+5-10%)
- TTT: **88-92%** accuracy (+2-6%)
- Non-zero-day: **70-75%** accuracy (+5-10%)

---

## 🔧 **Quick Fix Commands**

```python
# In config.py, update these lines:
meta_epochs: int = 25  # Line 521
k_shot: int = 150  # Line 657
learning_rate: float = 0.0015  # Line 21
```

---

## 📝 **Testing Plan**

1. **Apply Priority 1 fixes**
2. **Run training** and monitor:
   - Training loss convergence
   - Validation accuracy
   - Base model performance
3. **Compare results** with previous run
4. **If improvement < 5%**, apply Priority 2 fixes
5. **If still low**, consider Priority 3 (Optuna optimization)

---

## 💡 **Key Insight**

The main issue is **insufficient training intensity** for UNSW's complexity. The model needs:

- More epochs to learn complex patterns
- More support samples for better meta-learning
- Higher learning rate for faster convergence

These fixes should improve base model performance from 70% to 75-80%.



