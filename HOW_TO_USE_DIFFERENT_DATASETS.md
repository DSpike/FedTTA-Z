# How to Use Different Datasets with Different Configuration Settings

## 🎯 **Overview**

This guide explains how to easily switch between different datasets (KDD, UNSW-NB15, CICIDS2017, CICIDS2023) with their optimized configuration settings.

---

## 🚀 **Quick Start**

### **Method 1: Command-Line Argument (Recommended)**

```bash
# Run with KDD dataset
python main.py --dataset KDD

# Run with UNSW-NB15 dataset
python main.py --dataset UNSW

# Run with CICIDS2017 dataset
python main.py --dataset CICIDS2017

# Run with CICIDS2023 dataset
python main.py --dataset CICIDS2023
```

### **Method 2: Environment Variable**

```bash
# Windows PowerShell
$env:DATASET="KDD"; python main.py

# Windows CMD
set DATASET=KDD && python main.py

# Linux/Mac
export DATASET=KDD && python main.py
```

### **Method 3: Programmatic (In Code)**

```python
from config_loader import get_dataset_config

# Get KDD configuration
config = get_dataset_config('KDD')
system = BlockchainFederatedIncentiveSystem(config)

# Get UNSW configuration
config = get_dataset_config('UNSW')
system = BlockchainFederatedIncentiveSystem(config)
```

### **Method 4: Auto-Detection (Current Default)**

The system will auto-detect the dataset from `data_path` in `config.py`:

```python
# If config.py has:
data_path: str = "KDDTest+.txt"  # → Auto-detects KDD
data_path: str = "UNSW_NB15_training-set.csv"  # → Auto-detects UNSW
```

---

## 📋 **Available Datasets**

| Dataset        | Command                | Input Dim | Hidden Dim | Embedding Dim | Category Grouping |
| -------------- | ---------------------- | --------- | ---------- | ------------- | ----------------- |
| **KDD**        | `--dataset KDD`        | 41        | 128        | 256           | ✅ Yes            |
| **UNSW-NB15**  | `--dataset UNSW`       | 43        | 256        | 128           | ❌ No             |
| **CICIDS2017** | `--dataset CICIDS2017` | 78        | 256        | 128           | ✅ Yes            |
| **CICIDS2023** | `--dataset CICIDS2023` | 45        | 256        | 128           | ✅ Yes            |

---

## 🔧 **How It Works**

### **1. Config Loader (`config_loader.py`)**

The `config_loader.py` module provides:

- **Dataset-specific presets**: Pre-configured hyperparameters for each dataset
- **Auto-detection**: Automatically detects dataset from command-line args, environment, or data_path
- **Easy switching**: One command to switch between datasets

### **2. Configuration Presets**

Each dataset has optimized settings stored in `DATASET_CONFIGS`:

```python
DATASET_CONFIGS = {
    'KDD': {
        'input_dim': 41,
        'hidden_dim': 128,
        'embedding_dim': 256,
        'sequence_length': 22,
        'tcn_kernel_sizes': (2, 3, 3),
        'confidence_rejection_threshold': 0.90,
        # ... more settings
    },
    'UNSW': {
        'input_dim': 43,
        'hidden_dim': 256,
        'embedding_dim': 128,
        'sequence_length': 21,
        'tcn_kernel_sizes': (3, 3, 6),
        'confidence_rejection_threshold': 0.70,
        # ... more settings
    },
    # ... more datasets
}
```

### **3. Integration with main.py**

The `main.py` file has been updated to use the config loader:

```python
# In main.py
try:
    from config_loader import get_dataset_config
    config = get_dataset_config()  # Auto-detects dataset
except ImportError:
    config = get_config()  # Fallback to default
```

---

## 📝 **Step-by-Step Examples**

### **Example 1: Run KDD Dataset**

```bash
# Step 1: Run with KDD dataset
python main.py --dataset KDD

# Output:
# ✅ Loaded configuration for dataset: KDD
#    Data path: KDDTest+.txt
#    Hidden dim: 128, Embedding dim: 256
```

### **Example 2: Run UNSW Dataset**

```bash
# Step 1: Run with UNSW dataset
python main.py --dataset UNSW

# Output:
# ✅ Loaded configuration for dataset: UNSW
#    Data path: UNSW_NB15_training-set.csv
#    Hidden dim: 256, Embedding dim: 128
```

### **Example 3: Switch Between Datasets in Code**

```python
from config_loader import get_dataset_config
from main import BlockchainFederatedIncentiveSystem

# Run KDD
kdd_config = get_dataset_config('KDD')
kdd_system = BlockchainFederatedIncentiveSystem(kdd_config)
kdd_system.initialize_system()
# ... run KDD experiments

# Switch to UNSW
unsw_config = get_dataset_config('UNSW')
unsw_system = BlockchainFederatedIncentiveSystem(unsw_config)
unsw_system.initialize_system()
# ... run UNSW experiments
```

---

## ⚙️ **Customizing Dataset Configurations**

### **Option 1: Modify `config_loader.py`**

Edit the `DATASET_CONFIGS` dictionary in `config_loader.py`:

```python
DATASET_CONFIGS['KDD'] = {
    'hidden_dim': 256,  # Change from 128 to 256
    'learning_rate': 0.002,  # Change learning rate
    # ... modify other settings
}
```

### **Option 2: Override After Loading**

```python
from config_loader import get_dataset_config

config = get_dataset_config('KDD')
config.hidden_dim = 256  # Override specific setting
config.learning_rate = 0.002  # Override another setting
```

### **Option 3: Add New Dataset**

Add a new entry to `DATASET_CONFIGS`:

```python
DATASET_CONFIGS['MY_DATASET'] = {
    'input_dim': 50,
    'hidden_dim': 512,
    'embedding_dim': 256,
    'data_path': "my_dataset_train.csv",
    'test_path': "my_dataset_test.csv",
    # ... other settings
}
```

---

## 🔍 **Verification**

### **Check Current Configuration**

```python
from config_loader import get_dataset_config

config = get_dataset_config('KDD')
print(f"Dataset: KDD")
print(f"Input dim: {config.input_dim}")
print(f"Hidden dim: {config.hidden_dim}")
print(f"Embedding dim: {config.embedding_dim}")
print(f"Data path: {config.data_path}")
```

### **List Available Datasets**

```bash
python config_loader.py --list
```

Output:

```
Available datasets:
  - KDD:
      Data: KDDTest+.txt
      Input dim: 41
      Hidden dim: 128, Embedding dim: 256
  - UNSW:
      Data: UNSW_NB15_training-set.csv
      Input dim: 43
      Hidden dim: 256, Embedding dim: 128
  ...
```

---

## ⚠️ **Important Notes**

### **1. Dataset-Specific Requirements**

Each dataset may require:

- **Different preprocessors**: KDD uses `KDDPreprocessor`, UNSW uses `UNSWPreprocessor`
- **Different feature counts**: Verify `input_dim` matches actual features
- **Different attack types**: Check `zero_day_attack` is valid for the dataset

### **2. File Paths**

Ensure dataset files exist:

- **KDD**: `KDDTrain+.csv` and `KDDTest+.csv` in project root
- **UNSW**: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
- **CICIDS2017**: `CICIDS2017_training.csv` and `CICIDS2017_testing.csv`
- **CICIDS2023**: `CICIoT2023_training.csv` and `CICIoT2023_testing.csv`

### **3. Preprocessor Compatibility**

The system auto-selects preprocessors based on dataset name (see `main.py`):

- UNSW → `UNSWPreprocessor`
- KDD → `KDDPreprocessor`
- CICIDS2017 → `CICIDSPreprocessor`
- CICIDS2023 → `CICIDS2023Preprocessor`

---

## 🔄 **Migration from Old Approach**

### **Before (Manual Editing):**

```python
# Had to manually edit config.py each time
# config.py
input_dim: int = 41  # Change to 43 for UNSW
hidden_dim: int = 128  # Change to 256 for UNSW
data_path: str = "KDDTest+.txt"  # Change to UNSW path
```

### **After (Automatic):**

```bash
# Just use command-line argument
python main.py --dataset KDD
python main.py --dataset UNSW
```

---

## 📚 **Related Files**

- `config_loader.py` - Dataset-aware configuration loader
- `config.py` - Base configuration class
- `config_kdd_backup.py` - KDD settings backup
- `DATASET_CONFIG_COMPARISON.md` - Comparison of dataset settings
- `main.py` - Main execution file (updated to use config_loader)

---

## 💡 **Best Practices**

1. **Always specify dataset explicitly**: Use `--dataset` flag instead of relying on auto-detection
2. **Verify file paths**: Ensure dataset files exist before running
3. **Check input dimensions**: Verify `input_dim` matches actual feature count after preprocessing
4. **Test with small subset first**: Run with `quick_verify=True` to test configuration
5. **Keep backups**: Use `config_kdd_backup.py` as reference for original settings

---

## 🐛 **Troubleshooting**

### **Issue: "Unknown dataset 'XXX'"**

**Solution**: Check available datasets:

```bash
python config_loader.py --list
```

### **Issue: "File not found"**

**Solution**: Verify dataset files exist in project root:

```bash
ls *.csv *.txt  # Check for dataset files
```

### **Issue: "Input dimension mismatch"**

**Solution**: Check actual feature count after preprocessing and update `input_dim` in `DATASET_CONFIGS`.

---

## ✅ **Summary**

**To use different datasets with different configurations:**

1. **Use command-line argument**: `python main.py --dataset KDD`
2. **Or use environment variable**: `export DATASET=KDD && python main.py`
3. **Or use programmatic approach**: `get_dataset_config('KDD')`

**That's it!** The system automatically loads the correct configuration for each dataset. 🎉
