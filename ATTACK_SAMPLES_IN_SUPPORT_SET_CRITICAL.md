# ⚠️ CRITICAL: Attack Samples in Support Set - Current Configuration

## 🎯 **Your Question:**

**"Does the attack support set guarantee to include all attack types?"**

## ❌ **Answer: NO** (With Current Configuration)

---

## 📊 **Current Configuration**

```python
enforce_equal_support_composition: bool = False  # ⚠️ This disables the feature!
include_all_attack_types_in_support: bool = True  # ⚠️ This is IGNORED!
```

---

## 🔍 **Code Logic Analysis**

### **The Problem:**

The `include_all_attack_types_in_support` logic is **ONLY checked** inside this block:

```python
if n_way == 2 and enforce_equal_support_composition:  # Line 1521
    # ... Normal samples ...
    # ... Attack samples logic including include_all_attack_types_in_support ...
```

### **When `enforce_equal_support_composition = False`:**

The code takes the **else path** (line 1615):

```python
else:
    # For n_way != 2, use original random selection
    if len(available_labels) >= n_way:
        task_classes = torch.randperm(len(available_labels))[:n_way]  # Random selection!
        selected_labels = available_labels[task_classes]
    # ...
    for label in selected_labels:
        # Just takes k_shot samples from each selected class
        # NO CHECK for include_all_attack_types_in_support!
```

**This means:**

- ❌ `include_all_attack_types_in_support = True` is **IGNORED**
- ❌ It just **randomly selects** `n_way` classes (e.g., Normal=0 and one Attack type)
- ❌ Takes `k_shot` samples from **each selected class**
- ❌ **Does NOT include all attack types**

---

## 📋 **Actual Behavior (Current Config)**

### **What Happens:**

1. Code path: `else` block (line 1615) because `enforce_equal_support_composition = False`
2. Random selection: Selects `n_way = 2` random classes from available labels
3. Typical selection:
   - Class 1: Normal (label 0)
   - Class 2: **One random Attack type** (e.g., Fuzzers=1, or Generic=6, etc.)
4. Support set:
   - Normal: `k_shot = 129` samples
   - Attack: `k_shot = 129` samples from **ONE attack type only**
   - **Total: 258 samples**

### **Result:**

- ✅ Normal: 129 samples (50%)
- ❌ Attack: 129 samples from **ONE attack type** (NOT all attack types!)
- ❌ **Other attack types are excluded from this meta-task**

---

## ✅ **To Enable All Attack Types in Support Set**

You need **BOTH** flags enabled:

```python
enforce_equal_support_composition: bool = True  # ✅ Enable this!
include_all_attack_types_in_support: bool = True  # ✅ Keep this!
```

### **With Both Enabled:**

**Support Set Composition:**

- Normal: 129 samples (label 0)
- Attack: 129 samples **uniformly distributed across ALL 8 attack types**
  - If 8 attack types: ~16 samples per type (129 ÷ 8 = 16.125)
  - Distribution: 1 type gets 17 samples, 7 types get 16 samples each

**Result:**

- ✅ Normal: 129 samples (50%)
- ✅ Attack: 129 samples from **ALL attack types** (uniformly distributed)
- ✅ **All attack types included in every meta-task**

---

## 🔄 **Code Flow Comparison**

### **Current (enforce_equal_support_composition = False):**

```
1. Check: if n_way == 2 and enforce_equal_support_composition
   ❌ FALSE → Skip this block

2. Take else path (line 1615)
   - Randomly select 2 classes
   - Take k_shot from each
   - ❌ include_all_attack_types_in_support is NEVER checked!
```

### **If Enabled (enforce_equal_support_composition = True):**

```
1. Check: if n_way == 2 and enforce_equal_support_composition
   ✅ TRUE → Enter this block

2. Add Normal samples (k_shot)

3. Check: if include_all_attack_types_in_support and labels_for_attack_types is not None
   ✅ TRUE → Sample from ALL attack types uniformly
   ✅ All attack types included!
```

---

## 💡 **Recommendation**

**To guarantee all attack types in support set:**

Change in `config.py`:

```python
enforce_equal_support_composition: bool = True  # Change from False to True
include_all_attack_types_in_support: bool = True  # Keep True
```

**This will ensure:**

- ✅ Equal 50-50 Normal/Attack distribution
- ✅ All 8 attack types included in every meta-task
- ✅ Uniform distribution of attack samples across all types

---

## 📊 **Summary Table**

| Configuration                                                                               | Equal Composition | All Attack Types Included? |
| ------------------------------------------------------------------------------------------- | ----------------- | -------------------------- |
| `enforce_equal_support_composition = False`<br>`include_all_attack_types_in_support = True` | ❌ Not guaranteed | ❌ **NO** (ignored)        |
| `enforce_equal_support_composition = True`<br>`include_all_attack_types_in_support = False` | ✅ Yes (50-50)    | ❌ NO (one random type)    |
| `enforce_equal_support_composition = True`<br>`include_all_attack_types_in_support = True`  | ✅ Yes (50-50)    | ✅ **YES** (all types)     |

---

## 🎯 **Current Status**

**With your current config:**

- ❌ Attack samples are **NOT guaranteed** to include all attack types
- ❌ Each meta-task likely includes samples from **one random attack type** only
- ⚠️ The `include_all_attack_types_in_support = True` flag is **being ignored**!

**Fix:** Enable `enforce_equal_support_composition = True` to activate the all-attack-types feature.
