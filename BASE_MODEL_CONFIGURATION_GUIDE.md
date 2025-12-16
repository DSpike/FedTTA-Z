# Base Model Configuration Guide

## 📍 **Where to Configure Base Model**

The base model configuration is in **`config.py`** - the `SystemConfig` class.

---

## 🎯 **Key Base Model Parameters**

### **1. Model Architecture** (Lines 483-500)

```python
# === MODEL CONFIGURATION ===
input_dim: int = 41  # Number of input features (KDD: 41, UNSW: 43, CICIDS2017: varies)
hidden_dim: int = 128  # Hidden layer dimension
embedding_dim: int = 256  # Embedding space dimension
num_classes: int = 2  # Binary classification (Normal vs Attack)

# TCN Configuration
use_tcn: bool = True  # Enable/disable TCN feature extraction
sequence_length: int = 22  # Sequence length for TCN
sequence_stride: int = 12  # Stride for sequence creation
tcn_kernel_sizes: tuple = (2, 3, 3)  # TCN kernel sizes
use_residual_connections: bool = False  # TCN residual connections
```

**Location**: `config.py` lines **483-500**

---

### **2. Training Hyperparameters** (Lines 20-22, 500-501)

```python
# === CENTRALIZED LEARNING CONFIGURATION ===
learning_rate: float = 0.0016387494099028342  # Learning rate for meta-training
batch_size: int = 256  # Batch size

# Meta-Learning Configuration
meta_epochs: int = 21  # Number of meta-training epochs
transductive_steps: int = 40  # Transductive refinement steps
```

**Location**: `config.py` lines **20-22, 500-501**

---

### **3. Meta-Learning Task Configuration** (Lines 636-638)

```python
k_shot: int = 152  # Support set size (few-shot learning)
n_query: int = 16  # Query set size
num_meta_tasks: int = 34  # Number of meta-tasks (default, can be overridden)
enforce_equal_support_composition: bool = False  # Equal class distribution in support
```

**Location**: `config.py` lines **636-638**

---

### **4. Loss Function Weights** (Lines 488-492)

```python
# Embedding Quality Losses
center_loss_weight: float = 0.08  # Center loss weight (intra-class compactness)
margin_loss_weight: float = 0.25  # Margin loss weight (inter-class separation)
prototype_margin: float = 4.5  # Prototype margin for separation
```

**Location**: `config.py` lines **488-492**

---

### **5. Advanced Features** (Lines 494-505)

```python
use_supervised_contrastive_loss: bool = True  # Contrastive learning
contrastive_loss_weight: float = 0.3
use_multi_prototype: bool = True  # Multi-prototype learning
prototypes_per_class: int = 3
use_mixup_augmentation: bool = True  # Data augmentation
```

**Location**: `config.py` lines **494-505**

---

## 📝 **How to Modify Base Model Configuration**

### **Option 1: Direct Edit in `config.py`**

1. Open `config.py`
2. Find the `SystemConfig` class
3. Modify the parameter values
4. Save and run

**Example:**
```python
# Increase model capacity
hidden_dim: int = 512  # Changed from 128
embedding_dim: int = 256  # Keep or increase to 512

# Increase training intensity
meta_epochs: int = 50  # Changed from 21
learning_rate: float = 0.0025  # Changed from 0.0016

# Increase support set
k_shot: int = 250  # Changed from 152
```

---

### **Option 2: Use `config_loader.py` (Dataset-Specific)**

If you want different configurations for different datasets:

1. Open `config_loader.py`
2. Modify the dataset-specific configuration dictionary
3. The system will automatically use it when you run with `--dataset DATASET_NAME`

**Example:**
```python
'KDD': {
    'hidden_dim': 128,
    'embedding_dim': 256,
    'meta_epochs': 21,
    'k_shot': 152,
    # ... other parameters
}
```

---

## 🎯 **Most Important Parameters for Base Model Performance**

### **High Impact (⭐⭐⭐⭐⭐):**

1. **`meta_epochs`** - Training duration (more = better, but slower)
2. **`learning_rate`** - Training speed (too high = unstable, too low = slow)
3. **`hidden_dim`** - Model capacity (larger = more powerful, but slower)
4. **`embedding_dim`** - Embedding quality (larger = better separation)
5. **`k_shot`** - Support set size (more = better prototypes)

### **Medium Impact (⭐⭐⭐):**

6. **`center_loss_weight`** - Embedding compactness
7. **`margin_loss_weight`** - Class separation
8. **`transductive_steps`** - Refinement iterations
9. **`num_meta_tasks`** - Meta-learning diversity

---

## 📊 **Recommended Configuration for High Performance**

```python
# Model Capacity
hidden_dim: int = 512  # 2x increase
embedding_dim: int = 256  # Keep or increase to 512

# Training Intensity
meta_epochs: int = 50  # 2.5x increase
learning_rate: float = 0.0025  # 67% increase

# Support Set
k_shot: int = 250  # 64% increase
num_meta_tasks: int = 100  # 3x increase

# Loss Weights
center_loss_weight: float = 0.08  # 4x increase
margin_loss_weight: float = 0.25  # 108% increase
prototype_margin: float = 4.5  # 80% increase

# Refinement
transductive_steps: int = 40  # 2x increase
```

---

## ⚠️ **Important Notes**

1. **Memory Constraints**: Larger models (higher `hidden_dim`, `embedding_dim`) use more GPU memory
2. **Training Time**: More epochs (`meta_epochs`) and larger support sets (`k_shot`) increase training time
3. **Dataset-Specific**: Some parameters may need adjustment based on dataset size and characteristics
4. **TTT vs Base**: Base model config affects TTT performance - better base model = better TTT starting point

---

## 🔍 **Where Configuration is Used**

- **Model Creation**: `models/transductive_fewshot_model.py` - Uses config to build model
- **Training**: `coordinators/centralized_coordinator.py` - Uses config for training loop
- **Meta-Tasks**: `models/transductive_fewshot_model.py` - Uses config for task creation

---

## ✅ **Quick Reference**

| Parameter | Location | Default | Impact |
|-----------|----------|---------|--------|
| `hidden_dim` | config.py:485 | 128 | ⭐⭐⭐⭐ |
| `embedding_dim` | config.py:486 | 256 | ⭐⭐⭐⭐ |
| `meta_epochs` | config.py:500 | 21 | ⭐⭐⭐⭐⭐ |
| `learning_rate` | config.py:20 | 0.0016 | ⭐⭐⭐⭐⭐ |
| `k_shot` | config.py:636 | 152 | ⭐⭐⭐⭐ |
| `center_loss_weight` | config.py:488 | 0.08 | ⭐⭐⭐ |
| `margin_loss_weight` | config.py:491 | 0.25 | ⭐⭐⭐ |

---

**Main Configuration File**: `config.py` (SystemConfig class)




