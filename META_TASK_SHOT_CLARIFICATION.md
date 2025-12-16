# Meta-Task Shot Count Clarification

## 📊 **Current Configuration**

### **Meta-Task Structure:**
- **`n_way = 2`**: Two classes per task (Normal + Attack)
- **`k_shot = 169`** (or optimized value): Number of support samples **per class**

---

## 🎯 **Support Set Composition**

### **Per Meta-Task:**

With `k_shot = 169`:

```
Support Set = [k_shot Normal samples] + [k_shot Attack samples]
            = [169 Normal] + [169 Attack]
            = 338 total support samples
```

### **Terminology:**
- **"k-shot"**: Refers to **samples per class**, not total samples
- **With n_way=2**: It's a **"2-way, 169-shot"** meta-task
- **Total support samples**: `n_way × k_shot = 2 × 169 = 338 samples`

---

## 📋 **Complete Meta-Task Breakdown**

### **With `k_shot = 169`:**

| Component | Count | Details |
|-----------|-------|---------|
| **Normal Support** | 169 samples | Label 0 |
| **Attack Support** | 169 samples | Label 1 (from ALL 8 attack types) |
| **Total Support** | **338 samples** | `2 × 169` |
| **Query Set** | ~36 samples | `2 × n_query` (where `n_query = 18`) |

---

## 🔢 **Is It "200-Shot"?**

**No, it's not "200-shot"** in the traditional few-shot learning sense.

### **Correct Terminology:**
- ✅ **"2-way, 169-shot"** meta-task
- ✅ **338 total support samples** per task
- ❌ **NOT "200-shot"** (would imply 200 samples per class)

### **If You Want "200-Shot" Per Class:**
You would need:
```python
k_shot = 200  # 200 samples per class
```

This would result in:
- **Support Set**: `200 Normal + 200 Attack = 400 total support samples`
- **Meta-task type**: "2-way, 200-shot"

---

## 📊 **Comparison Table**

| k_shot | Normal Support | Attack Support | Total Support | Meta-Task Type |
|--------|----------------|----------------|---------------|----------------|
| **100** | 100 | 100 | **200** | 2-way, 100-shot |
| **150** | 150 | 150 | **300** | 2-way, 150-shot |
| **169** | 169 | 169 | **338** | 2-way, 169-shot (current optimized) |
| **200** | 200 | 200 | **400** | 2-way, 200-shot |

---

## 🎯 **Summary**

**Current Setup:**
- **Type**: 2-way, 169-shot meta-task
- **Total Support Samples**: 338 per task
- **Composition**: 169 Normal + 169 Attack (from all 8 attack types)

**If you're seeing "200":**
- That might be from the unit test (`k_shot = 100` → `2 × 100 = 200` total)
- Or you might want to change `k_shot` to 100 for testing

**The "shot" count refers to samples per class, not total samples.**










