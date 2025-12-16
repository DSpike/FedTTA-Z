# Meta-Task Configuration Source

## 📋 **Source: Configuration File (NOT Hard-Coded)**

The meta-task configuration comes from **`config.py`** (configuration file), **NOT** hard-coded values.

---

## 🔍 **Evidence**

### **1. Configuration Definition (`config.py`):**

**Location**: `config.py`, line 221

```python
class SystemConfig:
    # ... other configs ...
    
    num_meta_tasks: int = 20  # Reduced from 50 to 20 for faster training
```

✅ **Defined in configuration file** as a class attribute with default value `20`.

---

### **2. Usage in Code (`coordinators/simple_fedavg_coordinator.py`):**

**Location**: `coordinators/simple_fedavg_coordinator.py`, line 2590

```python
local_meta_tasks = create_meta_tasks(
    self.train_data,
    self.train_labels,
    n_way=self.config.n_way,
    k_shot=self.config.k_shot,
    n_query=self.config.n_query,
    n_tasks=self.config.num_meta_tasks,  # ← Read from config
    phase="training",
    ...
)
```

✅ **Read from `self.config.num_meta_tasks`** (not hard-coded).

---

### **3. Usage in Code (`main.py`):**

**Location**: `main.py`, line 1258

```python
local_meta_tasks = create_meta_tasks(
    client.train_data,
    client.train_labels,
    n_way=self.config.n_way,
    k_shot=self.config.k_shot,
    n_query=self.config.n_query,
    n_tasks=self.config.num_meta_tasks,    # ← Read from config
    phase="training",
    ...
)
```

✅ **Read from `self.config.num_meta_tasks`** (not hard-coded).

---

## ✅ **Answer**

### **Configuration Source:**
- ✅ **From Configuration File**: `config.py`
- ❌ **NOT Hard-Coded**: No hard-coded values in the code

### **How It Works:**
1. **Definition**: `num_meta_tasks: int = 20` in `config.py` (SystemConfig class)
2. **Access**: `self.config.num_meta_tasks` throughout the codebase
3. **Flexibility**: Can be changed by modifying `config.py` or passing different config instances

---

## 🔧 **How to Change It**

### **Option 1: Edit `config.py`**
```python
# In config.py
num_meta_tasks: int = 30  # Change from 20 to 30
```

### **Option 2: Pass Custom Config**
```python
from config import SystemConfig

custom_config = SystemConfig()
custom_config.num_meta_tasks = 30
system = BlockchainFederatedIncentiveSystem(custom_config)
```

### **Option 3: Environment Variable (if supported)**
```python
# In config.py SystemConfig class
num_meta_tasks: int = int(os.getenv('NUM_META_TASKS', 20))
```

---

## 📊 **Summary**

| Aspect | Source |
|--------|--------|
| **Definition** | `config.py` (SystemConfig class) |
| **Default Value** | `20` |
| **Usage** | `self.config.num_meta_tasks` |
| **Hard-Coded?** | ❌ No |
| **Configurable?** | ✅ Yes |

---

## 🎯 **Conclusion**

**All meta-task configuration comes from `config.py`**, not hard-coded values. The system is fully configurable through the configuration file.










