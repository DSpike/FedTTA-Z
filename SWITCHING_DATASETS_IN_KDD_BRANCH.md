# Switching Datasets in kdd-dataset-testing Branch

## ✅ **Yes, It's Appropriate!**

The `kdd-dataset-testing` branch is designed to support **multiple datasets** with centralized learning. Here's why:

---

## 🎯 **Why It's Appropriate**

### **1. Branch Supports Centralized Learning**

- ✅ Uses `CentralizedCoordinator` (not federated-specific)
- ✅ Works with any dataset that has a preprocessor
- ✅ Flexible architecture for dataset switching

### **2. Auto-Detection Infrastructure**

- ✅ `main.py` automatically detects dataset from file paths
- ✅ `config_loader.py` provides dataset-specific configurations
- ✅ Preprocessor selection is automatic

### **3. Multiple Preprocessors Available**

- ✅ `KDDPreprocessor` (for KDD)
- ✅ `UNSWPreprocessor` (for UNSW-NB15)
- ✅ `CICIDSPreprocessor` (for CICIDS2017)
- ✅ `CICIDS2023Preprocessor` (for CICIDS2023)

---

## 🔄 **How to Switch Datasets**

### **Method 1: Using config_loader (Recommended)**

```bash
# Switch to UNSW
python main.py --dataset UNSW

# Switch to CICIDS2017
python main.py --dataset CICIDS2017

# Switch to CICIDS2023
python main.py --dataset CICIDS2023

# Switch back to KDD
python main.py --dataset KDD
```

### **Method 2: Manual Config Edit**

Edit `config.py`:

```python
# For UNSW
data_path: str = "UNSW_NB15_training-set.csv"
test_path: str = "UNSW_NB15_testing-set.csv"
zero_day_attack: str = "DoS"
use_category_grouping: bool = False
input_dim: int = 43
# ... (other UNSW settings)

# For KDD
data_path: str = "KDDTrain+.csv"
test_path: str = "KDDTest+.csv"
zero_day_attack: str = "DoS"
use_category_grouping: bool = True
input_dim: int = 41
# ... (other KDD settings)
```

### **Method 3: Using switch_dataset_config.py**

```bash
python switch_dataset_config.py --dataset UNSW
python main.py
```

---

## ⚠️ **Important Considerations**

### **1. Preprocessor Compatibility**

The system auto-selects preprocessors based on file names:

```python
# In main.py (lines 452-478):
if 'KDD' in data_path.upper():
    → Uses KDDPreprocessor
elif 'CICIOT23' or 'CICIDS2023' in data_path:
    → Uses CICIDS2023Preprocessor
else:
    → Uses CICIDSPreprocessor (default)
```

**Note**: UNSW preprocessor is commented out in the else branch. If you want to use UNSW, you may need to:

- Add explicit UNSW detection, OR
- Use the config_loader which handles this

### **2. Configuration Must Match Dataset**

When switching, ensure:

- ✅ `input_dim` matches actual feature count
- ✅ `data_path` and `test_path` point to correct files
- ✅ `zero_day_attack` is valid for the dataset
- ✅ `use_category_grouping` matches dataset structure
- ✅ Hyperparameters are appropriate (or use config_loader)

### **3. File Availability**

Ensure dataset files exist:

- **KDD**: `KDDTrain+.csv`, `KDDTest+.csv`
- **UNSW**: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`
- **CICIDS2017**: `CICIDS2017_training.csv`, `CICIDS2017_testing.csv`
- **CICIDS2023**: `CICIoT2023_training.csv`, `CICIoT2023_testing.csv`

---

## 📊 **Current Branch Capabilities**

| Dataset        | Preprocessor              | Auto-Detection | Config Loader | Status          |
| -------------- | ------------------------- | -------------- | ------------- | --------------- |
| **KDD**        | ✅ KDDPreprocessor        | ✅ Yes         | ✅ Yes        | ✅ Ready        |
| **UNSW**       | ⚠️ UNSWPreprocessor       | ⚠️ Needs fix   | ✅ Yes        | ⚠️ May need fix |
| **CICIDS2017** | ✅ CICIDSPreprocessor     | ✅ Yes         | ✅ Yes        | ✅ Ready        |
| **CICIDS2023** | ✅ CICIDS2023Preprocessor | ✅ Yes         | ✅ Yes        | ✅ Ready        |

---

## 🔧 **Recommended Approach**

### **For Quick Testing:**

Use `config_loader.py`:

```bash
python main.py --dataset UNSW
```

### **For Production/Research:**

1. **Option A**: Use dedicated branches for each dataset

   - `kdd-dataset-testing` for KDD
   - `unsw-nb15-version` for UNSW (but it's federated-only)
   - `cicids2023-implementation` for CICIDS2023

2. **Option B**: Use this branch with config_loader
   - Switch datasets as needed
   - Keep track of which config was used for each experiment

---

## ✅ **Best Practice**

**For this branch (`kdd-dataset-testing`):**

1. ✅ **Yes, switch datasets** - The infrastructure supports it
2. ✅ **Use config_loader** - Ensures correct hyperparameters
3. ✅ **Document which dataset** - Keep track of experiments
4. ⚠️ **Verify preprocessor** - Check logs to ensure correct preprocessor is used
5. ⚠️ **Check file paths** - Ensure dataset files exist

---

## 🎯 **Summary**

**Can you switch datasets?** ✅ **YES**

**Is it appropriate?** ✅ **YES** - The branch is designed for centralized learning with multiple datasets

**How to do it?** Use `python main.py --dataset <DATASET_NAME>`

**Any concerns?** Just verify:

- Dataset files exist
- Correct preprocessor is selected (check logs)
- Configuration matches dataset (use config_loader)

---

## 💡 **Recommendation**

Since you're on `kdd-dataset-testing` branch which supports centralized learning, **switching datasets is perfectly fine and appropriate**. The branch name suggests KDD focus, but the code infrastructure supports multiple datasets.

**Just use the config_loader for clean switching:**

```bash
python main.py --dataset UNSW  # or KDD, CICIDS2017, CICIDS2023
```



