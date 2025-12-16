# Dataset Detection and Loading Flow
**Complete trace of how the system determines which dataset to use**

---

## 🔍 Complete Dataset Detection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN.PY STARTUP                         │
│                         (Line 7771)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              config_loader.get_dataset_config()                 │
│                    (config_loader.py:114)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Dataset Detection Priority Order  │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────┴────────────────────┐
        │                                      │
        ▼                                      │
┌──────────────────┐  NO                      │
│ 1. Command-line? │────────────────────┐     │
│  --dataset XXX   │                    │     │
└──────────────────┘                    │     │
  (Line 127-130)                        ▼     │
        │ YES                    ┌──────────────────┐  NO
        └───────────────────────▶│ 2. Environment?  │─────────┐
                                 │  DATASET=XXX     │         │
                                 └──────────────────┘         │
                                   (Line 133-135)             │
                                         │ YES                │
                                         └───────────┐        │
                                                     ▼        ▼
                                          ┌─────────────────────────┐
                                          │ 3. Auto-detect from     │
                                          │    config.py data_path  │
                                          └────────┬────────────────┘
                                                   │ (Lines 138-148)
                                                   ▼
                    ┌──────────────────────────────────────────────────┐
                    │ Read config.py (config_with_grouping.py):       │
                    │   data_path = "CICIDS2017_train.csv"            │
                    │                                                  │
                    │ Check data_path string:                          │
                    │   ✓ Contains "CICIDS2017" → dataset = 'CICIDS2017'│
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────────┐
                    │ Load DATASET_CONFIGS['CICIDS2017']              │
                    │ from config_loader.py (lines 58-92)             │
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────────┐
                    │ Create base SystemConfig() from config.py       │
                    │ (config_loader.py line 168)                     │
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────────┐
                    │ Override with CICIDS2017-specific values        │
                    │ (config_loader.py lines 171-173)                │
                    │   - input_dim: 78                                │
                    │   - hidden_dim: 512                              │
                    │   - embedding_dim: 128                           │
                    │   - data_path: "CICIDS2017_train.csv"           │
                    │   - zero_day_attack: "PortScan"                 │
                    │   - etc.                                         │
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────────┐
                    │ Return configured SystemConfig object            │
                    │ ✅ Dataset: CICIDS2017                           │
                    └──────────────────────────────────────────────────┘
```

---

## 📍 Where Dataset is Defined/Identified

### **Primary Definition Location: `config.py` (or `config_with_grouping.py`)**

**Lines 51-52:**
```python
data_path: str = "CICIDS2017_train.csv"
test_path: str = "CICIDS2017_test.csv"
```

**This is the MASTER setting that determines everything!**

### **Detection Logic: `config_loader.py`**

**Lines 137-148** - Auto-detection based on filename:
```python
# Auto-detect from data_path in default config
if dataset_name is None:
    default_config = SystemConfig()  # ← Loads config.py
    data_path = default_config.data_path.upper()  # ← Reads "CICIDS2017_TRAIN.CSV"

    if 'KDD' in data_path or 'NSL' in data_path:
        dataset_name = 'KDD'
    elif 'UNSW' in data_path:
        dataset_name = 'UNSW'
    elif 'CICIOT23' in data_path or 'CICIDS2023' in data_path:
        dataset_name = 'CICIDS2023'
    elif 'CICIDS2017' in data_path or 'CICIDS' in data_path:
        dataset_name = 'CICIDS2017'  # ✅ This matches!
```

---

## 🎯 Three Ways to Control Dataset Selection

### **Method 1: Command-Line Argument (Highest Priority)**
```bash
python main.py --dataset CICIDS2017
python main.py --dataset KDD
python main.py --dataset UNSW
```
- **File**: `config_loader.py` lines 127-130
- **Priority**: #1 (overrides everything)

### **Method 2: Environment Variable (Second Priority)**
```bash
export DATASET=CICIDS2017
python main.py
```
- **File**: `config_loader.py` lines 133-135
- **Priority**: #2 (used if no --dataset flag)

### **Method 3: config.py data_path (Fallback - Currently Active)**
Edit `config.py` lines 51-52:
```python
# For KDD:
data_path: str = "KDDTrain+.csv"
test_path: str = "KDDTest+.csv"

# For CICIDS2017:
data_path: str = "CICIDS2017_train.csv"  # ✅ CURRENT
test_path: str = "CICIDS2017_test.csv"   # ✅ CURRENT

# For UNSW:
data_path: str = "UNSW_NB15_training-set.csv"
test_path: str = "UNSW_NB15_testing-set.csv"
```
- **File**: `config.py` lines 51-52
- **Priority**: #3 (used if no flag or env var)
- **Current status**: ✅ Set to CICIDS2017

---

## 🔄 What Happens After Detection

Once dataset is identified, `config_loader.py` loads optimized hyperparameters:

```python
DATASET_CONFIGS = {
    'CICIDS2017': {
        'input_dim': 78,           # ← CICIDS2017-specific
        'hidden_dim': 512,         # ← Optuna-optimized
        'embedding_dim': 128,      # ← Optuna-optimized
        'data_path': "CICIDS2017_train.csv",
        'test_path': "CICIDS2017_test.csv",
        'zero_day_attack': "PortScan",  # ← CICIDS2017-specific
        'k_shot': 200,             # ← Optuna-optimized
        'ttt_lr': 0.002,          # ← Optuna-optimized
        # ... 20+ more optimized parameters
    }
}
```

All these values **override** the base config.py defaults.

---

## 📊 Current Configuration State

### **Active Settings:**
```
Source File: config.py (config_with_grouping.py)
└─ data_path = "CICIDS2017_train.csv"  ← MASTER CONTROL
   └─ Detected as: CICIDS2017
      └─ Loads: config_loader.DATASET_CONFIGS['CICIDS2017']
         └─ Final Config:
            ├─ Dataset: CICIDS2017
            ├─ Zero-day: PortScan
            ├─ Input dim: 78
            ├─ Hidden dim: 512
            ├─ Embedding dim: 128
            └─ All Optuna-optimized hyperparameters
```

### **Verification:**
```python
# Check current dataset detection
from config_loader import get_dataset_config
config = get_dataset_config()
print(f"Detected dataset: {config.data_path}")
# Output: CICIDS2017_train.csv
```

---

## 🎛️ How to Switch Datasets

### **Option A: Command-Line (Temporary)**
```bash
# Switch to KDD for one run
python main.py --dataset KDD

# Switch to UNSW for one run
python main.py --dataset UNSW
```

### **Option B: Edit config.py (Permanent)**
Edit `config.py` lines 51-52:
```python
# Change from CICIDS2017 to KDD:
data_path: str = "KDDTrain+.csv"
test_path: str = "KDDTest+.csv"
```

### **Option C: Environment Variable (Session)**
```bash
# Unix/Linux/Mac:
export DATASET=KDD
python main.py

# Windows:
set DATASET=KDD
python main.py
```

---

## 🔍 Debug: Trace Dataset Loading

Add this at the start of your run to see the detection process:
```python
import sys
print("=" * 60)
print("DATASET DETECTION TRACE")
print("=" * 60)

# Check config.py default
from config import SystemConfig
base = SystemConfig()
print(f"1. config.py data_path: {base.data_path}")

# Check config_loader detection
from config_loader import get_dataset_config
config = get_dataset_config()
print(f"2. Detected dataset: {config.data_path}")
print(f"3. Zero-day attack: {config.zero_day_attack}")
print(f"4. Input dim: {config.input_dim}")
print(f"5. Hidden dim: {config.hidden_dim}")
print("=" * 60)
```

---

## ✅ Summary

**Q: Where is the dataset defined and identified?**

**A: Three-step process:**

1. **Defined in**: `config.py` line 51
   ```python
   data_path: str = "CICIDS2017_train.csv"  ← MASTER SETTING
   ```

2. **Detected by**: `config_loader.py` lines 138-148
   - Reads the `data_path` from config.py
   - Checks if filename contains "CICIDS2017"
   - Sets `dataset_name = 'CICIDS2017'`

3. **Configuration loaded from**: `config_loader.py` lines 58-92
   - Loads DATASET_CONFIGS['CICIDS2017']
   - Applies all Optuna-optimized hyperparameters
   - Returns final configured SystemConfig object

**Current State**: ✅ CICIDS2017 (as defined in config.py line 51)
