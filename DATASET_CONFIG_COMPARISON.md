# Dataset Configuration Comparison: KDD vs UNSW-NB15

## 📊 **Quick Reference Table**

| Parameter                            | KDD-Optimized | UNSW-Optimized | Difference    | Impact                        |
| ------------------------------------ | ------------- | -------------- | ------------- | ----------------------------- |
| **`input_dim`**                      | 41            | 43             | +2            | Minor - feature count         |
| **`hidden_dim`**                     | 128           | 256            | **+128 (2x)** | **HIGH** - Model capacity     |
| **`embedding_dim`**                  | 256           | 128            | **-128 (2x)** | **HIGH** - Embedding space    |
| **`sequence_length`**                | 22            | 21             | -1            | Low - Temporal window         |
| **`sequence_stride`**                | 12            | 13             | +1            | Low - Sequence overlap        |
| **`tcn_kernel_sizes`**               | (2, 3, 3)     | (3, 3, 6)      | Different RF  | **MEDIUM** - Receptive fields |
| **`meta_epochs`**                    | 21            | 18             | -3            | Low - Training epochs         |
| **`k_shot`**                         | 152           | 118            | -34           | Medium - Support size         |
| **`n_query`**                        | 16            | 20             | +4            | Low - Query size              |
| **`learning_rate`**                  | 0.0016        | 0.0011         | -0.0005       | Medium - Learning speed       |
| **`confidence_rejection_threshold`** | 0.90          | 0.70           | **-0.20**     | **HIGH** - Sample rejection   |

---

## 🔍 **Detailed Comparison**

### **1. Model Architecture**

#### **KDD Configuration:**

```python
input_dim: int = 41
hidden_dim: int = 128
embedding_dim: int = 256
```

#### **UNSW Configuration:**

```python
input_dim: int = 43
hidden_dim: int = 256
embedding_dim: int = 128
```

**Analysis:**

- **Hidden dimension**: UNSW uses **2x larger** hidden dimension (256 vs 128)
  - **Reason**: UNSW has more complex feature interactions
  - **Impact**: Better feature extraction, but slower training
- **Embedding dimension**: UNSW uses **2x smaller** embedding dimension (128 vs 256)
  - **Reason**: UNSW features are more compact, less redundancy
  - **Impact**: Faster inference, but potentially less expressive

---

### **2. TCN Configuration**

#### **KDD Configuration:**

```python
sequence_length: int = 22
sequence_stride: int = 12
tcn_kernel_sizes: tuple = (2, 3, 3)
```

#### **UNSW Configuration:**

```python
sequence_length: int = 21
sequence_stride: int = 13
tcn_kernel_sizes: tuple = (3, 3, 6)
```

**Analysis:**

- **Sequence length**: Similar (22 vs 21) - minor difference
- **Sequence stride**: Similar (12 vs 13) - minor difference
- **TCN kernel sizes**: **Different receptive fields**
  - **KDD**: RF = 3, 7, 15 (from dilations 1, 2, 4)
  - **UNSW**: RF = 5, 11, 23 (from dilations 1, 2, 4 with larger kernels)
  - **Impact**: UNSW captures longer temporal patterns

---

### **3. Meta-Learning Configuration**

#### **KDD Configuration:**

```python
meta_epochs: int = 21
k_shot: int = 152
n_query: int = 16
learning_rate: float = 0.0016387494099028342
```

#### **UNSW Configuration:**

```python
meta_epochs: int = 18
k_shot: int = 118
n_query: int = 20
learning_rate: float = 0.001096821720752952
```

**Analysis:**

- **Meta epochs**: UNSW needs fewer epochs (18 vs 21)
  - **Reason**: UNSW data may converge faster
  - **Impact**: Faster training
- **K-shot**: UNSW uses fewer support samples (118 vs 152)
  - **Reason**: UNSW may have better class separation
  - **Impact**: Faster meta-training
- **N-query**: UNSW uses more query samples (20 vs 16)
  - **Reason**: Better evaluation during meta-training
  - **Impact**: More stable meta-learning
- **Learning rate**: UNSW uses lower learning rate (0.0011 vs 0.0016)
  - **Reason**: Larger hidden dimension requires smaller steps
  - **Impact**: More stable training, slower convergence

---

### **4. TTT Configuration**

#### **KDD Configuration:**

```python
ttt_base_steps: int = 70
ttt_adaptation_query_size: int = 1198
ttt_batch_size: int = 64
ttt_lr: float = 0.002
ttt_l2_reg_weight: float = 0.01
confidence_rejection_threshold: float = 0.90  # STRICT
```

#### **UNSW Configuration:**

```python
ttt_base_steps: int = 70  # Same
ttt_adaptation_query_size: int = 1198  # Same
ttt_batch_size: int = 64  # Same
ttt_lr: float = 0.002  # Same
ttt_l2_reg_weight: float = 0.01  # Same
confidence_rejection_threshold: float = 0.70  # RELAXED
```

**Analysis:**

- **TTT parameters**: Mostly the same (both optimized separately)
- **Confidence threshold**: **CRITICAL DIFFERENCE**
  - **KDD**: 0.90 (strict - model is confident on KDD)
  - **UNSW**: 0.70 (relaxed - model less confident on UNSW)
  - **Impact**: Using KDD threshold (0.90) on UNSW rejects **98% of samples!**

---

## 🔄 **How to Switch Between Datasets**

### **Option 1: Manual Edit (Current Approach)**

Edit `config.py` directly:

**For KDD:**

```python
input_dim: int = 41
hidden_dim: int = 128
embedding_dim: int = 256
sequence_length: int = 22
sequence_stride: int = 12
tcn_kernel_sizes: tuple = (2, 3, 3)
meta_epochs: int = 21
k_shot: int = 152
n_query: int = 16
learning_rate: float = 0.0016387494099028342
confidence_rejection_threshold: float = 0.90
data_path: str = "KDDTest+.txt"
test_path: str = "KDDTest+.txt"
zero_day_attack: str = "DoS"
use_category_grouping: bool = True
```

**For UNSW:**

```python
input_dim: int = 43
hidden_dim: int = 256
embedding_dim: int = 128
sequence_length: int = 21
sequence_stride: int = 13
tcn_kernel_sizes: tuple = (3, 3, 6)
meta_epochs: int = 18
k_shot: int = 118
n_query: int = 20
learning_rate: float = 0.001096821720752952
confidence_rejection_threshold: float = 0.70
data_path: str = "UNSW_NB15_training-set.csv"
test_path: str = "UNSW_NB15_testing-set.csv"
zero_day_attack: str = "DoS"
use_category_grouping: bool = False
```

---

### **Option 2: Use Backup File (Recommended)**

1. **Restore KDD settings:**

   ```python
   from config_kdd_backup import get_kdd_config
   kdd_config = get_kdd_config()
   # Copy values to config.py
   ```

2. **Or create a switch script** (see `switch_dataset_config.py`)

---

## ⚠️ **Critical Settings to Change When Switching**

### **MUST CHANGE (Performance Critical):**

1. **`hidden_dim`**: 128 (KDD) ↔ 256 (UNSW) - **2x difference!**
2. **`embedding_dim`**: 256 (KDD) ↔ 128 (UNSW) - **2x difference!**
3. **`confidence_rejection_threshold`**: 0.90 (KDD) ↔ 0.70 (UNSW) - **Rejects 98% if wrong!**
4. **`tcn_kernel_sizes`**: (2,3,3) (KDD) ↔ (3,3,6) (UNSW) - Different RF
5. **`input_dim`**: 41 (KDD) ↔ 43 (UNSW) - Model architecture
6. **`data_path`** and **`test_path`**: Dataset files
7. **`use_category_grouping`**: True (KDD) ↔ False (UNSW)

### **SHOULD CHANGE (Optimization):**

1. **`sequence_length`**: 22 (KDD) ↔ 21 (UNSW)
2. **`sequence_stride`**: 12 (KDD) ↔ 13 (UNSW)
3. **`meta_epochs`**: 21 (KDD) ↔ 18 (UNSW)
4. **`k_shot`**: 152 (KDD) ↔ 118 (UNSW)
5. **`n_query`**: 16 (KDD) ↔ 20 (UNSW)
6. **`learning_rate`**: 0.0016 (KDD) ↔ 0.0011 (UNSW)

---

## 📝 **Restoration Checklist**

When switching back to KDD, verify:

- [ ] `input_dim = 41`
- [ ] `hidden_dim = 128`
- [ ] `embedding_dim = 256`
- [ ] `sequence_length = 22`
- [ ] `sequence_stride = 12`
- [ ] `tcn_kernel_sizes = (2, 3, 3)`
- [ ] `meta_epochs = 21`
- [ ] `k_shot = 152`
- [ ] `n_query = 16`
- [ ] `learning_rate = 0.0016387494099028342`
- [ ] `confidence_rejection_threshold = 0.90`
- [ ] `data_path = "KDDTest+.txt"`
- [ ] `test_path = "KDDTest+.txt"`
- [ ] `zero_day_attack = "DoS"`
- [ ] `use_category_grouping = True`

---

## 💡 **Key Takeaways**

1. **Hidden/Embedding dimensions are CRITICAL** - 2x difference between datasets
2. **Confidence threshold is CRITICAL** - Wrong threshold rejects 98% of samples
3. **TCN kernel sizes affect receptive fields** - Different temporal patterns
4. **Most TTT parameters are the same** - Only confidence threshold differs
5. **Always verify `input_dim` matches actual feature count** after preprocessing

---

## 📚 **Related Files**

- `config.py` - Main configuration file (currently set for UNSW)
- `config_kdd_backup.py` - KDD-optimized settings backup
- `switch_dataset_config.py` - Helper script to switch between datasets (if created)



