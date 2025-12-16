# Clear Meta-Task Composition: Normal vs Attack Percentages

## 📊 **Support Set Composition**

### **With `k_shot = 169`:**

| Component  | Count   | Percentage |
| ---------- | ------- | ---------- |
| **Normal** | **169** | **50%**    |
| **Attack** | **169** | **50%**    |
| **Total**  | **338** | **100%**   |

✅ **Support Set: 50% Normal, 50% Attack** (Equal composition)

---

## 📋 **Query Set Composition**

### **With `n_query = 18` and Natural Distribution:**

**Total Query Samples**: `18 × 2 = 36 samples`

### **Typical Distribution (varies by client):**

| Component  | Count      | Percentage  | Notes            |
| ---------- | ---------- | ----------- | ---------------- |
| **Normal** | **~11-12** | **~31-33%** | Varies by client |
| **Attack** | **~24-25** | **~67-69%** | Varies by client |
| **Total**  | **36**     | **100%**    | Fixed total      |

⚠️ **Query Set: ~32% Normal, ~68% Attack** (Natural distribution, imbalanced)

---

## 🎯 **Complete Meta-Task Summary**

### **With `k_shot = 169` and `n_query = 18`:**

| Component           | Normal | Normal % | Attack | Attack % | Total |
| ------------------- | ------ | -------- | ------ | -------- | ----- |
| **Support Set**     | 169    | 50.0%    | 169    | 50.0%    | 338   |
| **Query Set**       | ~12    | ~33.3%   | ~24    | ~66.7%   | 36    |
| **Total Meta-Task** | ~181   | ~48.4%   | ~193   | ~51.6%   | 374   |

---

## 📊 **Visual Breakdown**

### **Support Set (338 samples):**

```
Normal:  ████████████████████ 169 (50%)
Attack:  ████████████████████ 169 (50%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:   338 samples
```

### **Query Set (36 samples):**

```
Normal:  ████  ~12 (~33%)
Attack:  ████████████  ~24 (~67%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:   36 samples
```

### **Complete Meta-Task (374 samples):**

```
Normal:  ████████████████████▌ ~181 (~48%)
Attack:  █████████████████████ ~193 (~52%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:   374 samples
```

---

## 🔍 **Why Different Percentages?**

### **Support Set (50/50):**

- ✅ **Enforced equal composition**
- ✅ Ensures balanced learning
- ✅ Model sees equal Normal/Attack examples during training

### **Query Set (32/68):**

- ✅ **Natural distribution** (matches real-world data)
- ✅ Reflects actual class imbalance in cybersecurity
- ✅ Tests model on realistic scenarios

---

## 💡 **Simple Answer**

### **Support Set:**

- **Normal: 50%** (169 samples)
- **Attack: 50%** (169 samples)

### **Query Set:**

- **Normal: ~33%** (~12 samples)
- **Attack: ~67%** (~24 samples)

### **Overall Meta-Task:**

- **Normal: ~48%** (~181 samples)
- **Attack: ~52%** (~193 samples)

---

## 📝 **Key Points**

1. **Support Set**: Always **50/50** (equal Normal/Attack)
2. **Query Set**: Typically **33/67** (natural distribution, imbalanced)
3. **Total**: Roughly **48/52** (slightly more attack samples due to query set imbalance)









