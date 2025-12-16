# Dataset Switching Guide

## 🎯 **Easy Dataset Switching - Now Implemented!**

Switching between datasets is now **as easy as changing one line** in `config.py`!

---

## ✅ **How to Switch Datasets**

### **Step 1: Change One Line in `config.py`**

```python
# config.py - Line 108

# For CICIDS2017 (current default):
dataset_type: DatasetType = DatasetType.CICIDS2017

# For UNSW-NB15:
dataset_type: DatasetType = DatasetType.UNSW_NB15

# For Edge-IIoT:
dataset_type: DatasetType = DatasetType.EDGE_IIOT
```

**That's it!** Everything else is auto-configured:
- ✅ Preprocessor selection
- ✅ Attack types dictionary
- ✅ Input dimension
- ✅ Data file paths
- ✅ Default zero-day attack

---

## 📋 **Supported Datasets**

### **1. CICIDS2017** (Default)
```python
dataset_type: DatasetType = DatasetType.CICIDS2017
```
- **Data files:** `CICIDS2017_train.csv`, `CICIDS2017_test.csv`
- **Input dimension:** 43 features
- **Default zero-day:** `PortScan` (label 10)
- **Attack types:** BENIGN, Bot, DDoS, DoS variants, PortScan, SSH-Patator, Web Attack, etc.

### **2. UNSW-NB15**
```python
dataset_type: DatasetType = DatasetType.UNSW_NB15
```
- **Data files:** `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`
- **Input dimension:** 196 features
- **Default zero-day:** `DoS` (label 4)
- **Attack types:** Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

### **3. Edge-IIoT**
```python
dataset_type: DatasetType = DatasetType.EDGE_IIOT
```
- **Data files:** `Edge-IIoTset.csv` (uses internal train/test split)
- **Input dimension:** 61 features
- **Default zero-day:** `DDoS_UDP`
- **Attack types:** (To be configured)

---

## 🔧 **Optional Overrides**

You can still override individual settings if needed:

```python
# config.py

# Change dataset
dataset_type: DatasetType = DatasetType.UNSW_NB15

# Optionally override specific settings:
data_path: str = "custom_train.csv"  # Override default path
test_path: str = "custom_test.csv"   # Override default path
zero_day_attack: str = "Exploits"    # Override default zero-day
input_dim: int = 200                 # Override default input dimension
```

**Note:** If you don't specify overrides, the system uses the dataset defaults automatically.

---

## 📊 **What Gets Auto-Configured**

When you change `dataset_type`, the following are automatically configured:

1. **Preprocessor Class**
   - CICIDS2017 → `CICIDSPreprocessor`
   - UNSW-NB15 → `UNSWPreprocessor`
   - Edge-IIoT → `EdgeIIoTPreprocessor`

2. **Attack Types Dictionary**
   - All attack type mappings for the selected dataset

3. **Input Dimension**
   - Feature count for the selected dataset

4. **Data File Paths**
   - Training and test file paths

5. **Default Zero-Day Attack**
   - Recommended zero-day attack for the dataset

6. **Default Zero-Day Label**
   - Integer label for the default zero-day attack

---

## 🚀 **Example: Switching from CICIDS2017 to UNSW-NB15**

### **Before (Old Way - Required 5-10 manual edits):**
1. Comment/uncomment preprocessor in `main.py`
2. Comment/uncomment attack_types in `config.py`
3. Change `data_path` and `test_path`
4. Change `input_dim`
5. Change `zero_day_attack`
6. Change default zero-day label

### **After (New Way - Just 1 line!):**
```python
# config.py - Line 108
dataset_type: DatasetType = DatasetType.UNSW_NB15
```

**Done!** Everything else is automatic.

---

## 🔍 **Verification**

After switching datasets, check the logs to verify:

```
Initializing UNSW_NB15 preprocessor...
   Module: preprocessing.blockchain_federated_unsw_preprocessor
   Class: UNSWPreprocessor
   Data path: UNSW_NB15_training-set.csv
   Test path: UNSW_NB15_testing-set.csv
✅ Preprocessor initialized: UNSWPreprocessor
```

---

## ⚠️ **Important Notes**

1. **File Availability:** Make sure the data files exist in your working directory
2. **Zero-Day Attack:** The default zero-day attack is a suggestion - you can override it
3. **Input Dimension:** The system will auto-update if preprocessing results in different feature count
4. **Backward Compatibility:** Old configs still work - if `dataset_type` is not set, it defaults to CICIDS2017

---

## 🎉 **Benefits**

- ✅ **One-line switching** - No more manual edits
- ✅ **Type-safe** - Enum prevents typos
- ✅ **Auto-configuration** - Everything set up automatically
- ✅ **Easy to extend** - Add new datasets to `DATASET_CONFIGS` dictionary
- ✅ **Backward compatible** - Old configs still work

---

## 📝 **Adding New Datasets**

To add a new dataset, edit `config.py`:

1. Add to `DatasetType` enum:
```python
class DatasetType(Enum):
    # ... existing ...
    NEW_DATASET = "new_dataset"
```

2. Add to `DATASET_CONFIGS`:
```python
DATASET_CONFIGS: Dict[DatasetType, Dict[str, Any]] = {
    # ... existing ...
    DatasetType.NEW_DATASET: {
        'data_path': "new_dataset_train.csv",
        'test_path': "new_dataset_test.csv",
        'input_dim': 100,
        'default_zero_day': "AttackType",
        'default_zero_day_label': 5,
        'attack_types': {
            'Normal': 0,
            'AttackType': 5,
            # ...
        },
        'preprocessor_module': 'preprocessing.new_preprocessor',
        'preprocessor_class': 'NewPreprocessor'
    }
}
```

That's it! The new dataset is now available.










