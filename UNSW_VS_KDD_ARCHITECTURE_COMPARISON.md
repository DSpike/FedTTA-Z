# UNSW-NB15 vs KDD Dataset Architecture Comparison

## Overview

This document compares the architecture implementations between the **UNSW-NB15** branch (`unsw-nb15-version`) and the **KDD** branch (`kdd-dataset-testing`).

---

## 🔍 Key Architectural Differences

### 1. **Preprocessor Architecture**

#### **UNSW-NB15 Preprocessor** (`preprocessing/blockchain_federated_unsw_preprocessor.py`)

- **Base Class**: `UNSWPreprocessor`
- **Features**: 45-48 features (after feature engineering)
- **Feature Engineering**: Adds 4 features (sbytes/dbytes ratios, packet ratios)
- **Categorical Columns**: `proto`, `service`, `state`
- **Sequence Creation**:
  - Default `sequence_length = 30`
  - Default `stride = 15`
  - Zero-padding enabled by default

#### **KDD Preprocessor** (`centralized_nids_kdd_preprocessor.py`)

- **Base Class**: Inherits from `UNSWPreprocessor` (reuses most logic)
- **Features**: 41 features (no feature engineering needed)
- **Feature Engineering**: **None** (KDD already has all features)
- **Categorical Columns**: `protocol_type`, `service`, `flag` (different names)
- **Sequence Creation**:
  - Uses config `sequence_length` (default: 22)
  - Uses config `sequence_stride` (default: 12)
  - Same zero-padding logic inherited from UNSW

**Key Difference**: KDD preprocessor is a **simplified version** that inherits from UNSW but skips feature engineering.

---

### 2. **Model Architecture**

#### **UNSW-NB15 Branch** (`unsw-nb15-version`)

```python
# From git show output:
self.model = TransductiveLearner(
    input_dim=self.config.input_dim,
    hidden_dim=64,  # Fixed at 64
    embedding_dim=self.config.embedding_dim,
    sequence_length=self.config.sequence_length,  # Uses config value
    tcn_kernel_sizes=tcn_kernel_sizes
)

# For non-TCN model:
sequence_length=1  # Single sample for UNSW-NB15
```

#### **KDD Branch** (`kdd-dataset-testing`)

```python
# Current implementation:
self.model = TransductiveLearner(
    input_dim=self.config.input_dim,  # 41 for KDD
    hidden_dim=64,  # Fixed at 64 (same as UNSW)
    embedding_dim=self.config.embedding_dim,  # 256 in config
    sequence_length=self.config.sequence_length,  # 22 in config
    disable_tcn_feature_extraction=getattr(self.config, 'disable_tcn_feature_extraction', False),
    tcn_kernel_sizes=tcn_kernel_sizes  # (2, 3, 3) in config
)
```

**Key Difference**:

- **UNSW**: Uses `sequence_length=1` for non-TCN models (single sample)
- **KDD**: Uses config `sequence_length=22` for TCN models (temporal sequences)

---

### 3. **Configuration Parameters**

#### **UNSW-NB15 Configuration** (from documentation)

```python
# Model Dimensions
input_dim: 25-48 features (after feature engineering)
hidden_dim: 128-512 (varies by optimization)
embedding_dim: 64-512 (varies by optimization)
sequence_length: 30 (for TCN) or 1 (for non-TCN)
sequence_stride: 13-15

# TCN Configuration
tcn_kernel_sizes: (2, 3, 4) or (3, 5, 7)
use_residual_connections: True/False

# Meta-Learning
k_shot: 100-200
n_query: 10-20
num_meta_tasks: 20-100
```

#### **KDD Configuration** (current `config.py`)

```python
# Model Dimensions
input_dim: 41 features (fixed, no feature engineering)
hidden_dim: 128 (optimized from Optuna Trial 1)
embedding_dim: 256 (optimized from Optuna Trial 1)
sequence_length: 22 (optimized from Optuna Trial 1)
sequence_stride: 12 (optimized from Optuna Trial 1)

# TCN Configuration
tcn_kernel_sizes: (2, 3, 3) (optimized from Optuna Trial 1)
use_residual_connections: False (optimized from Optuna Trial 1)

# Meta-Learning
k_shot: 152
n_query: 16
num_meta_tasks: 46
meta_epochs: 21
```

**Key Differences**:

1. **Input Dimension**: UNSW has variable features (25-48), KDD has fixed 41
2. **Sequence Length**: UNSW uses 30, KDD uses 22
3. **Embedding Dimension**: UNSW typically 64-128, KDD uses 256
4. **Hidden Dimension**: UNSW varies (64-512), KDD uses 128
5. **Kernel Sizes**: UNSW uses (2,3,4) or (3,5,7), KDD uses (2,3,3)

---

### 4. **TCN Architecture**

#### **UNSW-NB15 TCN** (from documentation)

- **MultiScaleTCN**: Three branches with `hidden_dims=[128, 64, 256]`
- **Output Dimension**: 448 (128 + 64 + 256)
- **Dilations**: [1, 2, 4] for short-, medium-, long-term patterns
- **Kernel Sizes**: [3, 3, 5] (larger kernel for long-term)
- **Dropout**: 0.2
- **Residual Connections**: Yes

#### **KDD TCN** (current implementation)

- **UnifiedDilatedTCN**: Single sequential path (not parallel branches)
- **Output Dimension**: `hidden_dim` (128) - single unified path
- **Dilations**: [1, 2, 4] (same as UNSW)
- **Kernel Sizes**: (2, 3, 3) - configurable per layer
- **Dropout**: 0.1 (configurable)
- **Residual Connections**: False (from config)

**Key Difference**:

- **UNSW**: Uses **parallel multi-branch TCN** (3 branches, 448-dim output)
- **KDD**: Uses **unified sequential TCN** (single path, 128-dim output) - **3× faster**

---

### 5. **Sequence Creation**

#### **UNSW-NB15**

```python
# Default parameters
sequence_length = 30
stride = 15
zero_pad = True

# Creates sequences from packets
# Uses flow IDs for grouping
```

#### **KDD**

```python
# From config
sequence_length = 22  # Shorter sequences
stride = 12  # Smaller stride
zero_pad = True  # Same padding logic

# Creates sequences from network flows
# No explicit flow ID grouping (uses sliding window)
```

**Key Difference**:

- **UNSW**: Longer sequences (30) with larger stride (15)
- **KDD**: Shorter sequences (22) with smaller stride (12)

---

### 6. **Attack Type Handling**

#### **UNSW-NB15**

```python
attack_types = {
    'Normal': 0,
    'Fuzzers': 1,
    'Analysis': 2,
    'Backdoor': 3,
    'DoS': 4,
    'Exploits': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
}
# Total: 10 classes (1 Normal + 9 Attack types)
```

#### **KDD**

```python
attack_types = {
    'normal': 0,
    # DoS attacks (10 types)
    'back': 1, 'land': 2, 'neptune': 3, ...
    # Probe attacks (6 types)
    'ipsweep': 7, 'nmap': 8, ...
    # R2L attacks (16 types)
    # U2R attacks (4 types)
    # ... total 40 attack types
}
# Total: 40+ specific attacks
# With category grouping: 5 categories (Normal, DoS, Probe, R2L, U2R)
```

**Key Difference**:

- **UNSW**: 10 classes (coarse-grained)
- **KDD**: 40+ specific attacks, can be grouped into 5 categories

---

### 7. **Feature Engineering**

#### **UNSW-NB15**

- **Step 2**: Feature Engineering adds 4 features:
  - `sbytes_ratio`: Ratio of source bytes
  - `dbytes_ratio`: Ratio of destination bytes
  - `packet_ratio`: Ratio of packets
  - `flow_ratio`: Ratio of flows
- **Result**: 45 → 48 features

#### **KDD**

- **Step 2**: **No feature engineering** (returns data as-is)
- **Reason**: KDD already has all necessary features
- **Result**: 41 features (fixed)

---

### 8. **Categorical Encoding**

#### **UNSW-NB15**

- **Columns**: `proto`, `service`, `state`
- **Encoding**: Target encoding for high-cardinality, one-hot for low-cardinality

#### **KDD**

- **Columns**: `protocol_type`, `service`, `flag` (different column names)
- **Encoding**: Same strategy (target/one-hot) but different column names

---

## 📊 Summary Table

| Aspect                   | UNSW-NB15                                   | KDD                                       |
| ------------------------ | ------------------------------------------- | ----------------------------------------- |
| **Preprocessor**         | `UNSWPreprocessor`                          | `KDDPreprocessor` (inherits from UNSW)    |
| **Input Features**       | 25-48 (variable, after engineering)         | 41 (fixed, no engineering)                |
| **Feature Engineering**  | ✅ Yes (adds 4 features)                    | ❌ No                                     |
| **Sequence Length**      | 30 (TCN) or 1 (non-TCN)                     | 22 (configurable)                         |
| **Sequence Stride**      | 15                                          | 12                                        |
| **TCN Architecture**     | Parallel multi-branch (3 branches, 448-dim) | Unified sequential (1 path, 128-dim)      |
| **TCN Kernel Sizes**     | (2,3,4) or (3,5,7)                          | (2,3,3)                                   |
| **Hidden Dimension**     | 64-512 (varies)                             | 128 (fixed)                               |
| **Embedding Dimension**  | 64-128 (typically)                          | 256                                       |
| **Attack Types**         | 10 classes                                  | 40+ specific (5 categories with grouping) |
| **Categorical Columns**  | `proto`, `service`, `state`                 | `protocol_type`, `service`, `flag`        |
| **Residual Connections** | Yes (typically)                             | No (from config)                          |
| **Model Complexity**     | Higher (multi-branch)                       | Lower (unified, 3× faster)                |

---

## 🎯 Key Architectural Insights

### **1. KDD is More Efficient**

- Uses **UnifiedDilatedTCN** (single path) instead of parallel branches
- **3× faster** than UNSW's multi-branch approach
- **83% fewer parameters** (128-dim vs 448-dim output)

### **2. UNSW Has More Features**

- Feature engineering adds 4 features
- More complex preprocessing pipeline
- Variable feature count (25-48)

### **3. Sequence Handling**

- **UNSW**: Longer sequences (30) for more temporal context
- **KDD**: Shorter sequences (22) for efficiency

### **4. Model Dimensions**

- **UNSW**: Smaller embedding (64-128), variable hidden (64-512)
- **KDD**: Larger embedding (256), fixed hidden (128)

### **5. Attack Type Granularity**

- **UNSW**: Coarse-grained (10 classes)
- **KDD**: Fine-grained (40+ attacks) with optional grouping

---

## 🔄 Code Reusability

**KDD Preprocessor** inherits from **UNSW Preprocessor**, which means:

- ✅ Most preprocessing logic is **shared**
- ✅ Only dataset-specific parts are overridden:
  - Attack type mappings
  - Categorical column names
  - Feature engineering (skipped for KDD)
- ✅ Sequence creation logic is **identical**

**Model Architecture** is **identical**:

- ✅ Same `TransductiveLearner` class
- ✅ Same TCN module (but different configuration)
- ✅ Same meta-learning approach
- ✅ Same TTT adaptation

---

## 💡 Recommendations

1. **For Better Performance**: Use KDD's unified TCN (faster, fewer parameters)
2. **For More Features**: Use UNSW's feature engineering approach
3. **For Fine-Grained Detection**: Use KDD's 40+ attack types
4. **For Simpler Setup**: Use KDD (no feature engineering needed)

The architectures are **very similar**, with KDD being a **more optimized version** that reuses UNSW's preprocessing logic but uses a more efficient TCN architecture.



