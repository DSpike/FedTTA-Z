# Query Set Composition in Meta-Tasks

## 📊 **Current Configuration**

### **Settings:**
- **`n_query = 18`**: Query samples per class (optimized value)
- **`n_way = 2`**: Two classes (Normal + Attack)
- **Total Query Samples**: `n_query × n_way = 18 × 2 = 36 samples`

---

## 🔍 **Query Set Distribution Logic**

**Important**: The query set uses **natural distribution** (not the `normal_query_ratio` parameter)!

### **Distribution Calculation:**

```python
# Calculate natural distribution from available data
natural_normal_ratio = len(normal_indices) / total_available
natural_attack_ratio = len(attack_indices) / total_available

# Sample based on natural distribution
target_normal_count = int(total_query_samples * natural_normal_ratio)
target_attack_count = total_query_samples - target_normal_count
```

**Note**: The `normal_query_ratio = 0.8` parameter is logged but **NOT used** in the actual sampling (after a scientific fix to use natural distribution).

---

## 📋 **Expected Query Set Composition**

### **Typical Training Data Distribution:**

In UNSW-NB15 training data (per client, after Dirichlet distribution):
- **Normal samples**: ~31-32% (varies by client)
- **Attack samples**: ~68-69% (varies by client, excluding zero-day)

### **Expected Query Set (with `n_query = 18`, `n_way = 2`):**

**Total Query Samples**: **36 samples**

| Component | Count | Percentage | Calculation |
|-----------|-------|------------|-------------|
| **Normal** | ~11-12 | ~31-33% | `36 × 0.32 ≈ 11-12` |
| **Attack** | ~24-25 | ~67-69% | `36 × 0.68 ≈ 24-25` |
| **Total** | **36** | **100%** | `18 × 2 = 36` |

**Note**: Actual counts vary by client (due to Dirichlet non-IID distribution) and per task (due to random sampling).

---

## 🎯 **Complete Meta-Task Composition Summary**

### **With `k_shot = 169` and `n_query = 18`:**

| Component | Normal | Attack | Total | Type |
|-----------|--------|--------|-------|------|
| **Support Set** | 169 | 169 | **338** | Equal (50/50) |
| **Query Set** | ~11-12 | ~24-25 | **36** | Natural (~32/68) |
| **Total Meta-Task** | ~180-181 | ~193-194 | **374** | Mixed |

---

## 📊 **Why Natural Distribution?**

The code comment explains:
```python
# SCIENTIFIC FIX: Use natural class distribution instead of artificial ratios
# Sample query set with realistic distribution based on available data
```

**Benefits:**
- ✅ **Realistic scenarios**: Matches real-world class imbalance
- ✅ **Better generalization**: Model learns to handle imbalanced query sets
- ✅ **Semantic meaning**: Preserves natural class relationships

---

## 🔧 **Actual Distribution Varies By:**

1. **Client heterogeneity**: Different clients have different data distributions (Dirichlet alpha)
2. **Data availability**: Natural ratio calculated from client's local data
3. **Random sampling**: Each task samples independently

---

## 📝 **Example Per Client:**

### **Client with Balanced Data (50/50):**
- Normal: 18 samples (~50%)
- Attack: 18 samples (~50%)
- **Total**: 36 samples

### **Client with Imbalanced Data (30/70):**
- Normal: 11 samples (~31%)
- Attack: 25 samples (~69%)
- **Total**: 36 samples

### **Client with Very Imbalanced Data (20/80):**
- Normal: 7 samples (~19%)
- Attack: 29 samples (~81%)
- **Total**: 36 samples

---

## 🧪 **To See Actual Distribution:**

The code logs the actual distribution at DEBUG level:
```
logger.debug(f"Query set distribution: {query_normal_count}/{total_query} Normal ({actual_normal_ratio:.1%}), target: {normal_query_ratio:.1%}")
```

To see this in training logs, set logging level to DEBUG.

---

## ⚠️ **Important Note:**

The `normal_query_ratio = 0.8` parameter is:
- ✅ **Logged** in info messages
- ❌ **NOT used** in actual query set creation
- ℹ️ **Kept for backward compatibility** (may be used in future)

The actual distribution is **natural** (based on available data ratios).










