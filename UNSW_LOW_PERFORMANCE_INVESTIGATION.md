# UNSW-NB15 Low Performance Investigation

## 🔍 **Current Performance Status**

Based on `performance_metrics_.json`:

### **Base Model:**
- **Accuracy**: 70.49% (0.7049)
- **F1-Score**: 74.63% (0.7463)
- **Precision**: 73.01%
- **Recall**: 76.32%
- **ROC-AUC**: 75.36%
- **AUC-PR**: 75.63%

### **TTT Model:**
- **Accuracy**: 85.96% (0.8596) ✅
- **F1-Score**: 92.45% (0.9245) ✅
- **Precision**: ~90%
- **Recall**: ~95%

### **Assessment:**
- **Base Model**: ⚠️ **MODERATE** (70% accuracy is below expected 75-85%)
- **TTT Model**: ✅ **GOOD** (86% accuracy, 92% F1 is acceptable)

---

## ❌ **Potential Issues Identified**

### **1. Zero-Day Attack Configuration Issue** ⚠️ **CRITICAL**

**Current Setting:**
```python
zero_day_attack: str = "Backdoor"  # UNSW-NB15 attack type (label 4)
```

**Problem:**
- Comment says "label 4" but **Backdoor is actually label 3** in UNSW-NB15!
- UNSW attack types:
  - Normal: 0
  - Fuzzers: 1
  - Analysis: 2
  - **Backdoor: 3** ← Current setting
  - DoS: 4
  - Exploits: 5
  - Generic: 6
  - Reconnaissance: 7
  - Shellcode: 8
  - Worms: 9

**Impact:**
- If the comment is wrong, the zero-day filtering might be incorrect
- Could cause zero-day samples to be included in training (data leakage)
- Or zero-day samples might not be found in test set

**Fix:**
```python
zero_day_attack: str = "Backdoor"  # UNSW-NB15 attack type (label 3) ← CORRECT LABEL
```

---

### **2. Input Dimension Verification** ⚠️ **HIGH PRIORITY**

**Current Setting:**
```python
input_dim: int = 43  # UNSW-NB15 has 43 features
```

**Need to Verify:**
- UNSW-NB15 raw: 49 features
- After feature engineering: 45-48 features
- After feature selection: 43 features (if enabled)

**Risk:**
- If actual features ≠ 43, model architecture mismatch
- Causes poor feature extraction
- Leads to low performance

**Check:**
```python
# After preprocessing, verify:
actual_input_dim = preprocessed_data['X_train'].shape[1]
if actual_input_dim != config.input_dim:
    logger.error(f"CRITICAL: Input dimension mismatch! Config={config.input_dim}, Actual={actual_input_dim}")
```

---

### **3. Confidence Rejection Threshold** ⚠️ **MEDIUM PRIORITY**

**Current Setting:**
```python
confidence_rejection_threshold: float = 0.70  # UNSW-optimized
```

**Status:**
- ✅ Already set to 0.70 (was 0.90, fixed earlier)
- Should be appropriate for UNSW

**Check:**
- Verify how many samples are being rejected
- If >50% rejected, consider lowering to 0.60-0.65

---

### **4. Training Configuration Issues** ⚠️ **MEDIUM PRIORITY**

**Current Settings:**
```python
meta_epochs: int = 18  # UNSW-optimized
k_shot: int = 118  # UNSW-optimized
n_query: int = 20  # UNSW-optimized
learning_rate: float = 0.001096821720752952  # UNSW-optimized
```

**Potential Issues:**
- **Meta epochs might be too low**: 18 epochs may not be enough for convergence
- **K-shot might be too low**: 118 support samples may be insufficient
- **Learning rate might be too low**: 0.0011 may cause slow convergence

**Recommendations:**
- Try increasing `meta_epochs` to 25-30
- Try increasing `k_shot` to 150-200
- Try increasing `learning_rate` to 0.0015-0.002

---

### **5. Data Preprocessing Issues** ⚠️ **HIGH PRIORITY**

**Potential Problems:**
1. **Feature scaling**: UNSW features might not be properly scaled
2. **Categorical encoding**: Protocol, service, state encoding might be incorrect
3. **Feature selection**: If enabled, might be removing important features
4. **Data imbalance**: UNSW has severe class imbalance (Normal vs Attacks)

**Check:**
- Verify feature scaling is applied (StandardScaler)
- Check categorical encoding (one-hot vs label encoding)
- Verify feature selection ratio (0.8 = keep 80% of features)
- Check class distribution in training/test sets

---

### **6. Model Architecture Issues** ⚠️ **MEDIUM PRIORITY**

**Current Settings:**
```python
hidden_dim: int = 256  # UNSW-optimized
embedding_dim: int = 128  # UNSW-optimized
sequence_length: int = 21  # UNSW-optimized
tcn_kernel_sizes: tuple = (3, 3, 6)  # UNSW-optimized
```

**Potential Issues:**
- **TCN kernel sizes**: (3, 3, 6) might not capture optimal temporal patterns
- **Sequence length**: 21 might be too short for UNSW patterns
- **Hidden dimension**: 256 might be too large (overfitting) or too small (underfitting)

---

## 🔧 **Investigation Steps**

### **Step 1: Verify Zero-Day Attack Configuration**

```python
# Check if Backdoor samples exist in test set
# Check if Backdoor is correctly filtered from training
# Verify label mapping
```

### **Step 2: Verify Input Dimension**

```python
# After preprocessing, log actual feature count
logger.info(f"Actual input dimension: {X_train.shape[1]}")
logger.info(f"Config input dimension: {config.input_dim}")
```

### **Step 3: Check Data Distribution**

```python
# Check class distribution
logger.info(f"Training class distribution: {np.bincount(y_train)}")
logger.info(f"Test class distribution: {np.bincount(y_test)}")
logger.info(f"Zero-day samples in test: {zero_day_mask.sum()}")
```

### **Step 4: Check Training Convergence**

```python
# Monitor training loss
# Check if model is converging
# Verify gradient updates
```

### **Step 5: Check Feature Quality**

```python
# Verify feature scaling
# Check for NaN/Inf values
# Verify categorical encoding
```

---

## ✅ **Recommended Fixes**

### **Priority 1: Fix Zero-Day Attack Label**

```python
# config.py
zero_day_attack: str = "Backdoor"  # UNSW-NB15 attack type (label 3) ← CORRECT
```

### **Priority 2: Verify Input Dimension**

Add logging in preprocessing to verify actual feature count matches config.

### **Priority 3: Increase Training Intensity**

```python
meta_epochs: int = 25  # Increase from 18
k_shot: int = 150  # Increase from 118
learning_rate: float = 0.0015  # Increase from 0.0011
```

### **Priority 4: Check Data Preprocessing**

Verify:
- Feature scaling applied correctly
- Categorical encoding correct
- No data leakage (zero-day in training)
- Class distribution reasonable

---

## 📊 **Expected Performance After Fixes**

### **Current:**
- Base: 70% accuracy
- TTT: 86% accuracy

### **Expected After Fixes:**
- Base: **75-85%** accuracy
- TTT: **88-92%** accuracy
- ZDR: **80-90%** (if zero-day configuration is correct)

---

## 🎯 **Next Steps**

1. ✅ Verify zero-day attack label (Backdoor = 3, not 4)
2. ✅ Add input dimension verification logging
3. ✅ Check data preprocessing pipeline
4. ✅ Monitor training convergence
5. ✅ Verify feature quality




