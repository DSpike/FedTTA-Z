# Dataset Switching - Quick Reference

## 🚀 **One-Line Commands**

```bash
# KDD Dataset
python main.py --dataset KDD

# UNSW-NB15 Dataset
python main.py --dataset UNSW

# CICIDS2017 Dataset
python main.py --dataset CICIDS2017

# CICIDS2023 Dataset
python main.py --dataset CICIDS2023
```

## 📊 **Configuration Differences**

| Setting                  | KDD     | UNSW    | CICIDS2017 | CICIDS2023 |
| ------------------------ | ------- | ------- | ---------- | ---------- |
| **Input Dim**            | 41      | 43      | 78         | 45         |
| **Hidden Dim**           | 128     | 256     | 256        | 256        |
| **Embedding Dim**        | 256     | 128     | 128        | 128        |
| **Sequence Length**      | 22      | 21      | 25         | 20         |
| **TCN Kernels**          | (2,3,3) | (3,3,6) | (3,5,7)    | (3,4,5)    |
| **Confidence Threshold** | 0.90    | 0.70    | 0.75       | 0.72       |
| **Category Grouping**    | ✅      | ❌      | ✅         | ✅         |

## 🔧 **Alternative Methods**

### **Environment Variable:**

```bash
# Windows
set DATASET=KDD && python main.py

# Linux/Mac
export DATASET=KDD && python main.py
```

### **In Python Code:**

```python
from config_loader import get_dataset_config
config = get_dataset_config('KDD')
```

## ✅ **Verification**

```bash
# List all available datasets
python config_loader.py --list
```

---

**For detailed documentation, see `HOW_TO_USE_DIFFERENT_DATASETS.md`**



