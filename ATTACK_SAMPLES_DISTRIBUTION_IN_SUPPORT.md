# Attack Samples Distribution in Support Set

## 📊 **Support Set Attack Samples: How Are They Distributed?**

### **With `k_shot = 169` and `include_all_attack_types_in_support=True`:**

**Total Attack Samples**: **169 samples**

**Available Attack Types**: **8 types** (excluding zero-day)
- Fuzzers (1)
- Analysis (2)
- DoS (4)
- Exploits (5)
- Generic (6)
- Reconnaissance (7)
- Shellcode (8)
- Worms (9)

---

## 🎯 **Distribution Calculation**

### **Uniform Distribution Logic:**

```python
samples_per_type = k_shot // num_attack_types  # 169 // 8 = 21
remaining_samples = k_shot % num_attack_types  # 169 % 8 = 1
```

**Result:**
- **7 attack types**: Get **21 samples each**
- **1 attack type**: Gets **22 samples** (21 + 1 remaining)
- **Total**: `7 × 21 + 1 × 22 = 147 + 22 = 169` ✅

---

## 📋 **Exact Distribution**

### **With 169 Attack Samples Across 8 Types:**

| Attack Type | Samples | Percentage |
|-------------|---------|------------|
| **Fuzzers (1)** | 21 | 12.4% |
| **Analysis (2)** | 21 | 12.4% |
| **DoS (4)** | 21 | 12.4% |
| **Exploits (5)** | 21 | 12.4% |
| **Generic (6)** | 21 | 12.4% |
| **Reconnaissance (7)** | 21 | 12.4% |
| **Shellcode (8)** | 21 | 12.4% |
| **Worms (9)** | 22 | 13.0% |
| **Total** | **169** | **100%** |

---

## ✅ **Answer to Your Question**

**YES!** The 169 attack samples are **nearly equally distributed** across all 8 attack types:

- **7 attack types**: **21 samples each** (~12.4%)
- **1 attack type**: **22 samples** (~13.0% - gets the extra sample)
- **Distribution**: **Uniform (as equal as possible)**

---

## 🔍 **Why Nearly Equal (Not Perfectly Equal)?**

Because `169 ÷ 8 = 21.125`, we can't have perfectly equal distribution:
- **Base samples per type**: `21` (integer division)
- **Remaining samples**: `1` (modulo)
- **Solution**: Distribute the 1 remaining sample to one attack type (randomly selected)

---

## 📊 **Visual Breakdown**

### **Attack Samples in Support Set (169 total):**

```
Fuzzers:         ████████████ 21 (12.4%)
Analysis:        ████████████ 21 (12.4%)
DoS:             ████████████ 21 (12.4%)
Exploits:        ████████████ 21 (12.4%)
Generic:         ████████████ 21 (12.4%)
Reconnaissance:  ████████████ 21 (12.4%)
Shellcode:       ████████████ 21 (12.4%)
Worms:           ████████████▌ 22 (13.0%) ← Gets extra sample
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:           169 samples (100%)
```

---

## 💡 **Simple Answer**

**YES!** The 169 attack samples are **uniformly distributed** across all 8 attack types:
- Each attack type gets **21-22 samples** (nearly equal)
- All attack types are represented in every meta-task
- This ensures the model learns from **all attack patterns** in each task

---

## 🎯 **Complete Support Set Breakdown**

### **Support Set (338 samples total):**

| Component | Count | Distribution |
|-----------|-------|--------------|
| **Normal** | 169 | 100% Normal (label 0) |
| **Attack** | 169 | **Uniformly distributed** across 8 types:<br>• 7 types: 21 samples each<br>• 1 type: 22 samples<br>All remapped to label 1 |
| **Total** | **338** | 50% Normal, 50% Attack |

---

## 📝 **Key Points**

1. ✅ **All 8 attack types** are included in every meta-task
2. ✅ **Nearly equal distribution**: 21-22 samples per attack type
3. ✅ **Uniform sampling**: Each attack type contributes equally
4. ✅ **Labels remapped**: All attack samples have label 1 (binary classification)










