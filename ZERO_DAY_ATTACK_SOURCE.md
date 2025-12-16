# Zero-Day Attack Configuration Source
**Where does main.py get the zero_day_attack value?**

---

## ✅ Answer: config_loader.py (NOT config.py)

**main.py receives: `zero_day_attack = "PortScan"`**
**Source: `config_loader.py` line 72**

---

## 🔍 Step-by-Step Trace

### Step 1: config.py Defines Base Value
**File**: `config.py` line 55
```python
zero_day_attack: str = "DoS"  # Base default value
```
**Status**: ⚠️ This value gets **OVERRIDDEN** (not used)

---

### Step 2: config_loader.py Has Dataset-Specific Value
**File**: `config_loader.py` lines 58-72
```python
DATASET_CONFIGS = {
    'CICIDS2017': {
        'input_dim': 78,
        'hidden_dim': 512,
        'embedding_dim': 128,
        ...
        'zero_day_attack': "PortScan",  # ← CICIDS2017-specific value
        ...
    }
}
```
**Status**: ✅ This is the value that gets used

---

### Step 3: config_loader.py Overrides Base Config
**File**: `config_loader.py` lines 167-175
```python
def get_dataset_config(dataset_name: Optional[str] = None) -> SystemConfig:
    ...
    # Create base config with default values
    base_config = SystemConfig()  # ← Loads config.py (zero_day_attack = "DoS")

    # Override with dataset-specific values
    for key, value in dataset_config.items():  # ← dataset_config = DATASET_CONFIGS['CICIDS2017']
        if hasattr(base_config, key):
            setattr(base_config, key, value)  # ← OVERWRITES zero_day_attack from "DoS" → "PortScan"

    return base_config
```

**What happens**:
1. `base_config = SystemConfig()` creates config with `zero_day_attack = "DoS"`
2. Loop finds `'zero_day_attack': "PortScan"` in DATASET_CONFIGS['CICIDS2017']
3. `setattr(base_config, 'zero_day_attack', "PortScan")` overwrites the value
4. Returns config with `zero_day_attack = "PortScan"`

---

### Step 4: main.py Receives Overridden Config
**File**: `main.py` line 7772
```python
config = get_dataset_config()  # ← Returns config with zero_day_attack = "PortScan"
```

---

## 📊 Configuration Override Verification

```
┌─────────────────────────────────────────────────────────┐
│ config.py (Base Config)                                 │
│ zero_day_attack = "DoS"                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ config_loader.py loads base config                      │
│ base_config = SystemConfig()                            │
│ → zero_day_attack = "DoS"  ← Initial value              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ config_loader.py applies dataset-specific overrides     │
│ DATASET_CONFIGS['CICIDS2017']['zero_day_attack']        │
│ → zero_day_attack = "PortScan"  ← OVERRIDDEN            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ main.py receives final config                           │
│ config.zero_day_attack = "PortScan"  ← FINAL VALUE      │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Which File Controls Zero-Day Attack?

### For CICIDS2017 Dataset:
**Answer: `config_loader.py` line 72**
```python
'zero_day_attack': "PortScan",  # ← THIS IS THE ACTIVE VALUE
```

### For KDD Dataset:
**Answer: `config_loader.py` line 38**
```python
'zero_day_attack': "DoS",  # ← Would be used if running KDD
```

### For UNSW Dataset:
**Answer: `config_loader.py` line 55**
```python
'zero_day_attack': "DoS",  # ← Would be used if running UNSW
```

---

## ⚠️ Common Confusion

**Q: I changed `zero_day_attack` in `config.py` to "neptune", why isn't it working?**

**A**: Because `config_loader.py` **overwrites** that value with the dataset-specific setting!

### If You Want to Change Zero-Day Attack:

**Option 1: Edit config_loader.py (Recommended)**
```python
# config_loader.py line 72
'CICIDS2017': {
    ...
    'zero_day_attack': "SSH-Patator",  # ← Change this
    ...
}
```

**Option 2: Edit config.py AND disable config_loader**
Modify `main.py` line 7771-7775 to use only config.py:
```python
# Comment out config_loader
# from config_loader import get_dataset_config
# config = get_dataset_config()

# Use base config directly
from config import get_config
config = get_config()
```
**Not recommended** - you'll lose all Optuna-optimized hyperparameters!

---

## 🔬 Runtime Verification

To verify which value is actually being used:
```python
from config_loader import get_dataset_config
config = get_dataset_config()
print(f"Zero-day attack: {config.zero_day_attack}")
# Output: PortScan
```

---

## 📝 All Zero-Day Attack Settings

### Current Values in Your System:

| File | Location | Value | Status |
|------|----------|-------|--------|
| **config_loader.py** | Line 38 (KDD) | "DoS" | Inactive (different dataset) |
| **config_loader.py** | Line 55 (UNSW) | "DoS" | Inactive (different dataset) |
| **config_loader.py** | Line 72 (CICIDS2017) | **"PortScan"** | ✅ **ACTIVE** |
| **config_loader.py** | Line 108 (CICIDS2023) | "DDoS" | Inactive (different dataset) |
| **config.py** | Line 55 | "DoS" | ❌ **OVERRIDDEN** |
| **config_with_grouping.py** | Line 55 | "DoS" | ❌ **NOT USED** |

---

## ✅ Summary

**Question: Does main.py take zero_day_attack from config.py or config_loader.py?**

**Answer**:
- **Source**: `config_loader.py` line 72
- **Value**: `"PortScan"`
- **Process**: config.py value ("DoS") is loaded first, then immediately overridden by config_loader.py
- **Result**: main.py receives "PortScan" from config_loader.py

**To change zero-day attack**: Edit `config_loader.py` line 72, NOT config.py
