# Query Set Composition in Meta-Tasks

## 📊 **Query Set Configuration**

### **Current Settings:**

- **`n_query = 18`** (optimized value): Number of query samples **per class**
- **`normal_query_ratio = 0.8`**: 80% Normal samples in query set (for training phase)

---

## 🔍 **How Query Set is Created**

The query set uses **natural distribution** based on available data, not forced ratios (after a scientific fix).

### **Query Set Creation Logic:**

1. **Calculate natural distribution** from available data:

   ```python
   natural_normal_ratio = len(normal_indices) / total_available
   natural_attack_ratio = len(attack_indices) / total_available
   ```

2. **Sample query set** maintaining natural distribution:

   - `target_normal_count = int(total_query_samples * natural_normal_ratio)`
   - `target_attack_count = total_query_samples - target_normal_count`

3. **Total query samples**: `n_query × n_way = 18 × 2 = 36 samples`

---

## 📋 **Expected Query Set Composition**

### **With `n_query = 18` and `n_way = 2`:**

**Total Query Samples**: `18 × 2 = 36 samples`

### **Distribution (Natural, based on data availability):**

Since training data typically has:

- **Normal**: ~31.94% (56,000 / 175,341)
- **Attack**: ~68.06% (119,341 / 175,341) - excluding zero-day

**Expected Query Set:**

- **Normal samples**: `36 × 0.32 ≈ 11-12 samples` (~32%)
- **Attack samples**: `36 × 0.68 ≈ 24-25 samples` (~68%)

**Note**: The `normal_query_ratio = 0.8` parameter exists but the implementation uses natural distribution.

---

## 🎯 **Complete Meta-Task Composition**

### **With `k_shot = 169` and `n_query = 18`:**

| Component       | Normal   | Attack   | Total   |
| --------------- | -------- | -------- | ------- |
| **Support Set** | 169      | 169      | **338** |
| **Query Set**   | ~11-12   | ~24-25   | **36**  |
| **Total**       | ~180-181 | ~193-194 | **374** |

---

## 🔧 **Why Natural Distribution?**

The code comment explains:

```python
# SCIENTIFIC FIX: Use natural class distribution instead of artificial ratios
# Sample query set with realistic distribution based on available data
```

This ensures:

- ✅ Realistic test scenarios (matches real-world data distribution)
- ✅ Better generalization (model sees realistic class imbalance)
- ✅ Maintains semantic meaning of class relationships

---

## 📊 **Actual Distribution May Vary**

The actual distribution depends on:

1. **Available samples** in client's local data
2. **Data sampling** (random selection from available)
3. **Client heterogeneity** (different clients may have different distributions)

---

## 🧪 **To Verify Query Set Composition**

You can check the actual composition by:

1. Running training and checking logs (should show query set distribution)
2. Inspecting meta-task dictionaries (contains `query_x`, `query_y`)
3. Adding debug logging to print query set composition per task

Would you like me to add code to log the actual query set composition during training?









