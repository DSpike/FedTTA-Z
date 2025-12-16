# Zero-Day Attacks in Query Set

## 🔍 **Answer: It Depends on the Phase**

### **Training/Validation Phase:**

**❌ NO - Zero-day attacks are EXCLUDED from query set**

### **Testing Phase:**

**✅ YES - Zero-day attacks are INCLUDED in query set** (this is where they should be for evaluation)

---

## 📊 **Code Analysis**

### **How Query Set is Created:**

The query set uses `attack_indices` which is created at line 1502:

```python
# For attack samples, exclude zero-day attack if specified
# Use multiclass labels if available for zero-day exclusion, otherwise use binary
if include_all_attack_types_in_support and labels_for_attack_types is not None:
    # Use multiclass labels to exclude zero-day
    if zero_day_attack_label is not None:
        attack_mask = (data_y != 0) & (labels_for_attack_types != zero_day_attack_label)
    else:
        attack_mask = data_y != 0
else:
    # Use binary labels (fallback)
    if zero_day_attack_label is not None:
        attack_mask = (data_y != 0) & (data_y != zero_day_attack_label)
    else:
        attack_mask = data_y != 0
attack_indices = torch.where(attack_mask)[0]
```

**Key Point**: `attack_indices` explicitly excludes zero-day attacks when `zero_day_attack_label is not None`.

### **Query Set Sampling (lines 1688-1693):**

```python
# Sample attack samples for query set (from all available attack samples, excluding zero-day if specified)
if len(attack_indices) >= target_attack_count:
    attack_query_indices = attack_indices[torch.randperm(len(attack_indices))[:target_attack_count]]
else:
    attack_query_indices = attack_indices
```

**The query set samples from `attack_indices`, which already excludes zero-day attacks.**

---

## 🎯 **Phase-Specific Behavior**

### **1. Training Phase (`phase="training"`):**

```python
# For training phase, exclude zero-day attack if specified
if phase in ["training", "validation"] and zero_day_attack_label is not None:
    # Filter out zero-day attack from available labels
    available_labels = unique_labels[unique_labels != zero_day_attack_label]
```

**Result**:

- ❌ Zero-day attacks are **excluded** from both support and query sets
- ✅ Model learns from Normal + 8 known attack types only
- ✅ Model never sees zero-day patterns during training

### **2. Testing Phase (`phase="testing"`):**

```python
else:
    available_labels = unique_labels  # No filtering
```

**Result**:

- ✅ Zero-day attacks can be included in query set
- ✅ Model is tested on zero-day samples it has never seen
- ✅ This evaluates true zero-day detection capability

---

## 📋 **Summary**

| Phase          | Support Set              | Query Set                           | Zero-Day             |
| -------------- | ------------------------ | ----------------------------------- | -------------------- |
| **Training**   | Normal + 8 known attacks | Normal + 8 known attacks            | ❌ Excluded          |
| **Validation** | Normal + 8 known attacks | Normal + 8 known attacks            | ❌ Excluded          |
| **Testing**    | Normal + 8 known attacks | Normal + 8 known attacks + Zero-day | ✅ Included in query |

---

## ✅ **Why This is Correct**

1. **Training Phase Exclusion**:

   - Model must learn from known attacks only
   - Zero-day should be completely unseen during training
   - Ensures proper zero-day detection evaluation

2. **Testing Phase Inclusion**:

   - Zero-day appears in test/query set
   - Model is evaluated on truly unseen attack patterns
   - Tests generalization to unknown attacks

3. **Consistent Exclusion in Support Set**:
   - Support set never includes zero-day (regardless of phase)
   - This is correct because support set represents "known patterns"
   - Zero-day is by definition "unknown"

---

## 🔍 **Verification Needed**

To verify this behavior, check:

1. What `phase` parameter is passed to `create_meta_tasks` during training
2. Whether zero-day samples exist in training data (they shouldn't after zero-day split)
3. Whether query sets during testing include zero-day samples



