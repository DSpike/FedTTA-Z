# How config_loader.py and config.py Work Together

## 🔄 **Relationship Between Files**

### **`config.py`** (Base Configuration)

- **Purpose**: Defines the `SystemConfig` class with ALL configuration parameters
- **Contains**: Default values for ALL settings (KDD, UNSW, CICIDS2017, CICIDS2023, etc.)
- **Role**: Base class and default values

### **`config_loader.py`** (Dataset-Specific Override)

- **Purpose**: Selects dataset-specific values and overrides base config
- **Contains**: Only dataset-specific overrides (data_path, input_dim, hyperparameters, etc.)
- **Role**: Wrapper that creates `SystemConfig` instance and modifies only relevant fields

---

## 📊 **How They Work Together**

### **Step 1: `main.py` Calls `config_loader`**

```python
# In main.py (line 7730-7736):
try:
    from config_loader import get_dataset_config
    config = get_dataset_config()  # Uses config_loader
except ImportError:
    config = get_config()  # Fallback to config.py directly
```

### **Step 2: `config_loader` Creates Base Config from `config.py`**

```python
# In config_loader.py (line 148-149):
# Create base config with default values
base_config = SystemConfig()  # ← Creates instance from config.py
```

### **Step 3: `config_loader` Overrides Only Dataset-Specific Values**

```python
# In config_loader.py (line 151-154):
# Override with dataset-specific values
for key, value in dataset_config.items():
    if hasattr(base_config, key):
        setattr(base_config, key, value)  # Override only these fields
```

### **Step 4: Returns Modified `SystemConfig` Instance**

```python
# Returns SystemConfig instance with dataset-specific overrides
return base_config  # Still a SystemConfig from config.py, but with modified values
```

---

## 🎯 **What Gets Overridden?**

### **`config_loader.py` Overrides:**

- ✅ `data_path` and `test_path`
- ✅ `input_dim`
- ✅ `hidden_dim`, `embedding_dim`
- ✅ `sequence_length`, `sequence_stride`
- ✅ `tcn_kernel_sizes`
- ✅ `meta_epochs`, `k_shot`, `n_query`
- ✅ `learning_rate`
- ✅ `confidence_rejection_threshold`
- ✅ `zero_day_attack`
- ✅ `use_category_grouping`

### **`config.py` Provides (Not Overridden):**

- ✅ `SystemConfig` class definition
- ✅ All other parameters (TTT settings, thresholds, etc.)
- ✅ Attack types dictionary (for KDD)
- ✅ All default values for non-dataset-specific settings

---

## 📋 **Example Flow**

### **When Running `python main.py --dataset CICIDS2017`:**

1. **`main.py`** calls `get_dataset_config()` from `config_loader.py`

2. **`config_loader.py`**:

   - Detects `--dataset CICIDS2017`
   - Creates `SystemConfig()` instance (from `config.py`) with ALL defaults
   - Gets `DATASET_CONFIGS['CICIDS2017']` dictionary
   - Overrides only these fields:
     ```python
     base_config.data_path = "CICIDS2017_train.csv"
     base_config.test_path = "CICIDS2017_test.csv"
     base_config.input_dim = 78
     base_config.zero_day_attack = "PortScan"
     # ... etc
     ```
   - Returns modified `SystemConfig` instance

3. **Result**: A `SystemConfig` object with:
   - **Base values** from `config.py` (TTT settings, thresholds, etc.)
   - **Overridden values** from `config_loader.py` (dataset-specific settings)

---

## 🔍 **Key Points**

### **1. `config.py` is Still Used**

- ✅ `SystemConfig` class is defined in `config.py`
- ✅ All default values come from `config.py`
- ✅ All non-dataset-specific settings come from `config.py`

### **2. `config_loader.py` is a Wrapper**

- ✅ Doesn't replace `config.py`
- ✅ Only overrides dataset-specific values
- ✅ Still returns a `SystemConfig` instance from `config.py`

### **3. Both Files Are Needed**

- ❌ **Cannot remove `config.py`** - it defines the class
- ❌ **Cannot remove `config_loader.py`** - it provides dataset switching
- ✅ **Both work together** - `config_loader` uses `config.py` as base

---

## 💡 **Why This Design?**

### **Advantages:**

1. **Single Source of Truth**: `SystemConfig` class defined once in `config.py`
2. **Dataset Flexibility**: Easy switching via `config_loader.py`
3. **Backward Compatibility**: Can still use `config.py` directly (fallback)
4. **Maintainability**: Dataset-specific values in one place (`config_loader.py`)

### **Disadvantages:**

- Need to maintain both files
- Must ensure `config_loader` values match `SystemConfig` attributes

---

## 🔧 **Current Implementation**

```python
# main.py
config = get_dataset_config()  # From config_loader.py
# Returns: SystemConfig instance (from config.py) with dataset-specific overrides

# config_loader.py
base_config = SystemConfig()  # Creates from config.py
# Override dataset-specific values
return base_config  # Still a SystemConfig from config.py

# config.py
@dataclass
class SystemConfig:
    # All configuration parameters with defaults
    ...
```

---

## ✅ **Summary**

**Question**: Does `main.py` use only `config_loader.py` and not `config.py`?

**Answer**: **NO** - `main.py` uses **BOTH**:

- **`config_loader.py`**: Selects dataset-specific values
- **`config.py`**: Provides `SystemConfig` class and all default values
- **`config_loader.py`** creates a `SystemConfig` instance from `config.py` and overrides only dataset-specific fields

**Both files are required and work together!** 🎯



