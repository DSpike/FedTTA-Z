# 🎯 Centralized Learning Implementation Plan

## 📊 **Goal**

Enable centralized learning mode **without changing** the existing federated learning code. This allows you to:

- Run experiments in centralized mode (all data in one place)
- Switch easily between federated and centralized modes
- Keep all existing code intact

---

## 🔧 **Implementation Strategy**

### **Step 1: Add Configuration Flag**

Add to `config.py`:

```python
use_federated_learning: bool = True  # Set to False for centralized learning
```

### **Step 2: Create Centralized Coordinator**

Create `coordinators/centralized_coordinator.py` that:

- Maintains same interface as `SimpleFedAVGCoordinator`
- Uses ALL training data directly (no client splitting)
- Reuses existing meta-learning training code
- Reuses existing TTT adaptation code

### **Step 3: Update main.py**

Modify coordinator initialization to check the flag:

```python
if config.use_federated_learning:
    self.coordinator = SimpleFedAVGCoordinator(...)
else:
    self.coordinator = CentralizedCoordinator(...)
```

### **Step 4: Ensure Compatibility**

Both coordinators must have:

- `distribute_data()` method
- `run_federated_round()` method (even if centralized)
- `adapt_to_test_data()` method for TTT
- Same model interface

---

## ✅ **Benefits**

- ✅ **Zero changes** to federated learning code
- ✅ **Easy switching** via one config flag
- ✅ **Same model architecture** and training
- ✅ **Fair comparison** between modes
- ✅ **All features work** (TTT, evaluation, visualization)

---

## 📝 **Next Steps**

I'll now implement this step-by-step.








