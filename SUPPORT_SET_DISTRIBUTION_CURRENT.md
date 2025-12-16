# Support Set Distribution During Training Phase

## 📊 **Current Configuration**

Based on `config.py`:
- **`k_shot = 129`** (samples per class)
- **`n_way = 2`** (binary: Normal vs Attack)
- **`enforce_equal_support_composition = False`** ⚠️ (Optimized from Optuna)
- **`include_all_attack_types_in_support = True`** ✅ (Enabled)

---

## 🔍 **Important Note**

**`enforce_equal_support_composition = False`** means the code logic for equal composition is **disabled**. However, the actual behavior depends on the code path taken.

---

## 📋 **Actual Support Set Distribution**

### **When `enforce_equal_support_composition = False`:**

The code will follow a **different path** (not the equal composition block). However, based on the implementation:

**Most Likely Behavior:**
1. Normal samples: Up to `k_shot` samples (or all available if less)
2. Attack samples: Varies based on data availability

**Without Equal Composition Enforcement:**
- Support set may have **unequal distribution** of Normal vs Attack
- Distribution depends on:
  - Available samples per class in client's local data
  - Random selection logic
  - Data availability constraints

---

## 🎯 **Expected Distribution (If Equal Composition Was Enabled)**

**If `enforce_equal_support_composition = True`** (which it's currently NOT):

### **With `k_shot = 129` and `include_all_attack_types_in_support = True`:**

**Support Set Composition Per Meta-Task:**

| Component | Count | Details |
|-----------|-------|---------|
| **Normal Support** | **129 samples** | Label 0 (all from Normal class) |
| **Attack Support** | **129 samples** | Label 1 (distributed across ALL attack types) |

**Attack Type Distribution (129 samples total):**
- If **8 attack types** available (excluding zero-day):
  - `samples_per_type = 129 // 8 = 16` samples per type
  - `remaining = 129 % 8 = 1` extra sample
  - **Result:**
    - 1 attack type: **17 samples** (16 + 1)
    - 7 attack types: **16 samples** each
  - **All attack labels remapped to 1** (binary classification)

**Total Support Set:**
- **Normal: 129 samples (50%)**
- **Attack: 129 samples (50%)**
- **Total: 258 samples per meta-task**

---

## 🔍 **To Verify Current Distribution**

Since `enforce_equal_support_composition = False`, the actual distribution may vary. To check the real distribution during training, you would need to:

1. **Add logging** in `create_meta_tasks()` function
2. **Run training** and check logs for actual support set composition
3. **Check the code path** that's executed when `enforce_equal_support_composition = False`

---

## 📊 **Example Distribution Scenarios**

### **Scenario 1: Equal Composition (If Enabled)**
```
Support Set (258 samples):
├─ Normal: 129 samples (50%)
└─ Attack: 129 samples (50%)
    ├─ Attack Type 1: 17 samples
    ├─ Attack Type 2: 16 samples
    ├─ Attack Type 3: 16 samples
    ├─ Attack Type 4: 16 samples
    ├─ Attack Type 5: 16 samples
    ├─ Attack Type 6: 16 samples
    ├─ Attack Type 7: 16 samples
    └─ Attack Type 8: 16 samples
```

### **Scenario 2: Current Configuration (enforce_equal_support_composition = False)**
```
Support Set (variable):
├─ Normal: Variable (up to 129, or all available if less)
└─ Attack: Variable (depends on data availability and selection logic)
```

---

## 💡 **Recommendation**

To ensure **equal 50-50 distribution**, consider:

```python
# In config.py, change:
enforce_equal_support_composition: bool = True  # Enable equal composition
```

This would ensure:
- **129 Normal samples** (50%)
- **129 Attack samples** (50%)
- **Total: 258 samples** per meta-task support set

---

## 📝 **Key Takeaways**

1. **Current Config:** `enforce_equal_support_composition = False`
2. **Expected:** Variable distribution (may not be 50-50)
3. **If Equal Enabled:** 129 Normal + 129 Attack = 258 total (50-50 split)
4. **Attack Samples:** With `include_all_attack_types_in_support = True`, attack samples are uniformly distributed across all available attack types

---

**Next Step:** Add logging to verify actual distribution during training runs.










