# Branch Learning Mode Summary

## 📊 Overview

This document summarizes which learning mode (Centralized vs Federated) each branch uses.

---

## 🔍 Branch Analysis

### 1. **`kdd-dataset-testing`** (Current Branch)

- **Dataset**: KDD/NSL-KDD (currently configured for UNSW-NB15)
- **Learning Mode**: ✅ **Centralized Learning**
- **Coordinator**: `CentralizedCoordinator`
- **Config Flag**: `use_federated_learning: bool = False`
- **Status**: Fully implemented with centralized learning support

---

### 2. **`unsw-nb15-version`**

- **Dataset**: UNSW-NB15
- **Learning Mode**: ❌ **Federated Learning Only**
- **Coordinator**: `SimpleFedAVGCoordinator`
- **Config Flag**: Not present (defaults to federated)
- **Status**: No centralized learning implementation

---

### 3. **`master`** (CICIDS2017)

- **Dataset**: CICIDS2017
- **Learning Mode**: ❌ **Federated Learning Only**
- **Coordinator**: `SimpleFedAVGCoordinator`
- **Config**: `num_clients: int = 4`, `num_rounds: int = 20`
- **Status**: Federated learning only, no centralized option

---

### 4. **`cicids2023-implementation`** (CICIoT2023)

- **Dataset**: CICIoT2023 (CICIDS2023)
- **Learning Mode**: ✅ **Centralized Learning** (with option to switch)
- **Coordinator**: `CentralizedCoordinator` (default)
- **Config Flag**: `use_federated_learning` flag exists (can switch modes)
- **Status**: Supports both modes, defaults to centralized

---

## 📋 Summary Table

| Branch                          | Dataset     | Learning Mode  | Coordinator               | Can Switch?       |
| ------------------------------- | ----------- | -------------- | ------------------------- | ----------------- |
| **`kdd-dataset-testing`**       | KDD/NSL-KDD | ✅ Centralized | `CentralizedCoordinator`  | ✅ Yes (via flag) |
| **`unsw-nb15-version`**         | UNSW-NB15   | ❌ Federated   | `SimpleFedAVGCoordinator` | ❌ No             |
| **`master`**                    | CICIDS2017  | ❌ Federated   | `SimpleFedAVGCoordinator` | ❌ No             |
| **`cicids2023-implementation`** | CICIoT2023  | ✅ Centralized | `CentralizedCoordinator`  | ✅ Yes (via flag) |

---

## 🎯 Key Findings

### **Centralized Learning Branches:**

1. ✅ **`kdd-dataset-testing`** - Fully centralized
2. ✅ **`cicids2023-implementation`** - Centralized with switch option

### **Federated Learning Only Branches:**

1. ❌ **`unsw-nb15-version`** - Federated only
2. ❌ **`master`** (CICIDS2017) - Federated only

---

## 💡 Recommendations

### **For Centralized Learning:**

- Use **`kdd-dataset-testing`** branch (current)
- Or use **`cicids2023-implementation`** branch

### **For Federated Learning:**

- Use **`unsw-nb15-version`** branch
- Or use **`master`** branch (CICIDS2017)

### **To Add Centralized Learning to Other Branches:**

1. Copy `coordinators/centralized_coordinator.py` from `kdd-dataset-testing` branch
2. Add `use_federated_learning: bool = False` to `config.py`
3. Update `main.py` to check the flag and use `CentralizedCoordinator` when `False`

---

## 🔄 Migration Path

If you want to run **CICIDS2017** or **UNSW-NB15** with centralized learning:

1. **Option 1**: Switch to `kdd-dataset-testing` branch and change dataset config
2. **Option 2**: Copy centralized coordinator to the target branch and add the flag
3. **Option 3**: Use `cicids2023-implementation` branch and change dataset config

---

## ✅ Current Status

**You are currently on `kdd-dataset-testing` branch with:**

- ✅ Centralized learning enabled
- ✅ UNSW-NB15 dataset configured
- ✅ All features working

This is the **best branch** for centralized learning experiments! 🚀



