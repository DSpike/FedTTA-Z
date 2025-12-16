# Dataset Switching Analysis

## 🔍 **Current State: Partially Easy, But Needs Improvement**

---

## ✅ **What's Easy to Switch:**

### **1. Data File Paths** ✅
```python
# config.py - Easy to change
data_path: str = "CICIDS2017_train.csv"
test_path: str = "CICIDS2017_test.csv"
```
**Status:** ✅ Easy - Just change file paths in `config.py`

---

### **2. Zero-Day Attack Selection** ✅
```python
# config.py - Easy to change
zero_day_attack: str = "PortScan"  # CICIDS2017
# or
zero_day_attack: str = "Exploits"  # UNSW-NB15
```
**Status:** ✅ Easy - Just change the attack name

---

## ⚠️ **What's Hard to Switch (Needs Manual Changes):**

### **1. Preprocessor Selection** ❌ **HARDCODED**
```python
# main.py lines 452-457 - HARDCODED
logger.info("Initializing CICIDS2017 preprocessor...")
from blockchain_federated_cicids_preprocessor import CICIDSPreprocessor
self.preprocessor = CICIDSPreprocessor(
    data_path=self.config.data_path,
    test_path=self.config.test_path
)
'''
# UNSW-NB15 preprocessor (commented out - using CICIDS2017 instead)
'''
```
**Problem:** Must manually uncomment/comment and change imports

**Impact:** 🔴 **HIGH** - Requires code changes in `main.py`

---

### **2. Attack Types Dictionary** ❌ **HARDCODED**
```python
# config.py lines 41-58 - HARDCODED
attack_types = {
    'BENIGN': 0,
    'Bot': 1,
    # ... CICIDS2017 attacks
}
'''
# UNSW-NB15 dataset attack types (commented out)
attack_types = {
    'Normal': 0,
    'Fuzzers': 1,
    # ... UNSW-NB15 attacks
}
'''
```
**Problem:** Must manually comment/uncomment entire dictionary

**Impact:** 🔴 **HIGH** - Easy to make mistakes, error-prone

---

### **3. Input Dimension** ❌ **HARDCODED**
```python
# config.py line 81
input_dim: int = 43  # CICIDS2017 specific (43 features)
```
**Problem:** Different datasets have different feature counts
- CICIDS2017: 43 features
- UNSW-NB15: Different feature count
- Edge-IIoT: Different feature count

**Impact:** 🟡 **MEDIUM** - Must manually update for each dataset

---

### **4. Zero-Day Attack Label Default** ❌ **HARDCODED**
```python
# config.py line 78
return self.attack_types.get(self.zero_day_attack, 10)  # Default to PortScan=10 (CICIDS2017)
```
**Problem:** Default value (10) is CICIDS2017-specific

**Impact:** 🟡 **MEDIUM** - May cause errors if attack not found

---

## 📊 **Current Dataset Support:**

### **Supported Datasets:**
1. ✅ **CICIDS2017** - Currently active
2. ✅ **UNSW-NB15** - Available but commented out
3. ✅ **Edge-IIoT** - Has separate `main_edgeiiot.py` file

### **Switching Process (Current):**

**To switch from CICIDS2017 to UNSW-NB15:**

1. **Change `config.py`:**
   ```python
   data_path: str = "UNSW_NB15_training-set.csv"
   test_path: str = "UNSW_NB15_testing-set.csv"
   zero_day_attack: str = "Exploits"
   
   # Comment out CICIDS attack_types
   # Uncomment UNSW-NB15 attack_types
   attack_types = {
       'Normal': 0,
       'Fuzzers': 1,
       # ...
   }
   
   input_dim: int = <UNSW_NB15_feature_count>
   ```

2. **Change `main.py`:**
   ```python
   # Comment out CICIDS preprocessor
   # Uncomment UNSW-NB15 preprocessor
   from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
   self.preprocessor = UNSWPreprocessor(...)
   ```

3. **Update zero_day_attack_label default:**
   ```python
   return self.attack_types.get(self.zero_day_attack, 4)  # Default to DoS=4 (UNSW-NB15)
   ```

**Total Changes:** ~5-10 manual edits across 2 files

---

## 🎯 **Recommended Improvements:**

### **Option 1: Dataset Enum/Config (Recommended)** ⭐

Create a `DatasetConfig` class:

```python
# config.py
from enum import Enum

class DatasetType(Enum):
    CICIDS2017 = "cicids2017"
    UNSW_NB15 = "unsw_nb15"
    EDGE_IIOT = "edge_iiot"

@dataclass
class DatasetConfig:
    dataset_type: DatasetType = DatasetType.CICIDS2017
    
    # Auto-configure based on dataset type
    @property
    def attack_types(self):
        if self.dataset_type == DatasetType.CICIDS2017:
            return {
                'BENIGN': 0,
                'Bot': 1,
                # ...
            }
        elif self.dataset_type == DatasetType.UNSW_NB15:
            return {
                'Normal': 0,
                'Fuzzers': 1,
                # ...
            }
    
    @property
    def default_input_dim(self):
        if self.dataset_type == DatasetType.CICIDS2017:
            return 43
        elif self.dataset_type == DatasetType.UNSW_NB15:
            return <UNSW_feature_count>
    
    @property
    def preprocessor_class(self):
        if self.dataset_type == DatasetType.CICIDS2017:
            from blockchain_federated_cicids_preprocessor import CICIDSPreprocessor
            return CICIDSPreprocessor
        elif self.dataset_type == DatasetType.UNSW_NB15:
            from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
            return UNSWPreprocessor
```

**Usage:**
```python
# config.py - Just change one line!
dataset_type: DatasetType = DatasetType.UNSW_NB15
```

**Benefits:**
- ✅ Single point of configuration
- ✅ No manual commenting/uncommenting
- ✅ Type-safe (enum prevents typos)
- ✅ Auto-configures attack_types, input_dim, preprocessor

---

### **Option 2: Configuration File (YAML/JSON)**

Create `dataset_configs.yaml`:

```yaml
datasets:
  cicids2017:
    data_path: "CICIDS2017_train.csv"
    test_path: "CICIDS2017_test.csv"
    input_dim: 43
    attack_types:
      BENIGN: 0
      Bot: 1
      # ...
    preprocessor: "blockchain_federated_cicids_preprocessor.CICIDSPreprocessor"
    default_zero_day: "PortScan"
  
  unsw_nb15:
    data_path: "UNSW_NB15_training-set.csv"
    test_path: "UNSW_NB15_testing-set.csv"
    input_dim: <count>
    attack_types:
      Normal: 0
      Fuzzers: 1
      # ...
    preprocessor: "preprocessing.blockchain_federated_unsw_preprocessor.UNSWPreprocessor"
    default_zero_day: "DoS"
```

**Usage:**
```python
# config.py
dataset_name: str = "unsw_nb15"  # Just change this!
```

**Benefits:**
- ✅ No code changes needed
- ✅ Easy to add new datasets
- ✅ Can be modified without touching code

---

### **Option 3: Minimal Changes (Quick Fix)**

Just add a `dataset_type` parameter:

```python
# config.py
dataset_type: str = "CICIDS2017"  # or "UNSW_NB15"

# main.py
if config.dataset_type == "CICIDS2017":
    from blockchain_federated_cicids_preprocessor import CICIDSPreprocessor
    self.preprocessor = CICIDSPreprocessor(...)
elif config.dataset_type == "UNSW_NB15":
    from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
    self.preprocessor = UNSWPreprocessor(...)
```

**Benefits:**
- ✅ Minimal code changes
- ✅ Quick to implement
- ⚠️ Still need to manually update attack_types

---

## 📝 **Summary:**

### **Current Ease of Switching:**
- 🔴 **HARD** - Requires 5-10 manual edits across 2 files
- ⚠️ **Error-prone** - Easy to forget a change
- ⚠️ **No validation** - Typos not caught until runtime

### **After Recommended Improvements:**
- ✅ **EASY** - Change 1 line in config
- ✅ **Safe** - Type checking prevents errors
- ✅ **Maintainable** - Clear structure

### **Recommendation:**
Implement **Option 1 (Dataset Enum/Config)** for best balance of:
- Ease of use
- Type safety
- Code maintainability
- Backward compatibility

---

## 🚀 **Quick Assessment:**

**Question:** "Is my code easy to switch between datasets?"

**Answer:** 
- ❌ **Currently: NO** - Requires manual edits in 2 files
- ✅ **Can be improved to: YES** - With recommended changes, just change 1 line

**Effort to Improve:**
- **Option 1 (Enum):** ~30 minutes
- **Option 2 (YAML):** ~1 hour
- **Option 3 (Quick Fix):** ~10 minutes










