# Support Set Composition: Normal vs Attack Traffic

## 📊 **How Support Set is Composed**

The support set composition depends on the `enforce_equal_support_composition` parameter and `n_way` setting.

---

## 🔍 **Current Configuration**

- **`n_way = 2`** (Binary classification: Normal vs Attack)
- **`enforce_equal_support_composition = True`** (Default, can be optimized)
- **`k_shot = 150`** (or optimized value: 100-200)

---

## ✅ **With `enforce_equal_support_composition = True`**

### **Step 1: Class Selection**
For each meta-task:
1. **Always selects Normal (label 0)**
2. **Always selects ONE random Attack class** (excluding zero-day)
3. **Result**: `selected_labels = [0, attack_class]` where `attack_class != zero_day_label`

### **Step 2: Support Set Creation**
For each selected class:
- Gets all samples with that label from client's local data
- Randomly selects `k_shot` samples (or all available if fewer than `k_shot`)
- Adds to support set

### **Step 3: Final Support Set**
```
Support Set = [k_shot Normal samples] + [k_shot Attack samples]
Total Support Samples = 2 * k_shot
```

**Example with `k_shot = 150`:**
- ✅ **150 Normal samples** (label 0)
- ✅ **150 Attack samples** (label = selected attack class, e.g., Fuzzers=1)
- ✅ **Total: 300 support samples**
- ✅ **Composition: 50% Normal, 50% Attack** (Equal)

---

## 📋 **Code Flow**

```python
# Line 1491-1510: Class Selection
if n_way == 2 and enforce_equal_support_composition:
    # 1. Always select Normal (0)
    normal_label = torch.tensor([0])
    
    # 2. Get available attack labels (exclude zero-day)
    attack_labels = available_labels[(available_labels != 0) & 
                                     (available_labels != zero_day_attack_label)]
    
    # 3. Select one random attack class
    selected_attack_label = random.choice(attack_labels)
    selected_labels = [normal_label, selected_attack_label]
    # Result: [0, 1] or [0, 2] etc. (always Normal + one Attack)

# Line 1525-1544: Support Set Creation
for label in selected_labels:  # Loop: [0, attack_class]
    # Get all samples with this label
    class_indices = torch.where(data_y == label)[0]
    
    # Select k_shot samples (or all if fewer available)
    if len(class_indices) >= k_shot:
        support_indices = random.sample(class_indices, k_shot)
    else:
        support_indices = class_indices  # Use all available
    
    # Add to support set
    support_x_list.append(data_x[support_indices])
    support_y_list.append(data_y[support_indices])

# Line 1546-1550: Combine Support Sets
support_x = torch.cat(support_x_list, dim=0)  # Concatenate Normal + Attack
support_y = torch.cat(support_y_list, dim=0)  # Concatenate labels

# Result: support_x has k_shot Normal + k_shot Attack samples
#         support_y has k_shot zeros + k_shot attack_labels
```

---

## 🎯 **Verification**

The code verifies equal composition at lines 1552-1560:

```python
if n_way == 2 and enforce_equal_support_composition:
    support_normal_count = (support_y == 0).sum().item()      # Count Normal
    support_attack_count = (support_y != 0).sum().item()      # Count Attack
    
    if support_normal_count != support_attack_count:
        logger.warning(f"⚠️  Unequal: {support_normal_count} Normal vs {support_attack_count} Attack")
    else:
        logger.debug(f"✅ Equal: {support_normal_count} Normal, {support_attack_count} Attack")
```

**Expected Output:**
- ✅ Equal: 150 Normal, 150 Attack (when `k_shot = 150`)
- ✅ Equal: 115 Normal, 115 Attack (when `k_shot = 115`)
- ⚠️ Unequal: Only happens if client doesn't have enough samples for one class

---

## ⚠️ **When Composition May Be Unequal**

### **Scenario 1: Insufficient Normal Samples**
If client has only 50 Normal samples but `k_shot = 150`:
- ✅ Uses all 50 Normal samples
- ✅ Uses 150 Attack samples (if available)
- ⚠️ **Result: 50 Normal, 150 Attack** (Unequal)

### **Scenario 2: Insufficient Attack Samples**
If client has only 80 Attack samples but `k_shot = 150`:
- ✅ Uses 150 Normal samples (if available)
- ✅ Uses all 80 Attack samples
- ⚠️ **Result: 150 Normal, 80 Attack** (Unequal)

### **Scenario 3: Insufficient Samples for Both**
If client has only 30 Normal and 40 Attack:
- ✅ Uses all 30 Normal samples
- ✅ Uses all 40 Attack samples
- ⚠️ **Result: 30 Normal, 40 Attack** (Unequal, but uses all available)

**These scenarios are logged as warnings and handled gracefully.**

---

## 🔄 **What Happens Without Equal Composition Enforcement**

If `enforce_equal_support_composition = False`:

```python
# Random selection (line 1517-1519)
task_classes = torch.randperm(len(available_labels))[:n_way]
selected_labels = available_labels[task_classes]
```

**Possible outcomes:**
- ✅ `[0, 1]` - Normal + Attack (Good)
- ❌ `[1, 2]` - Attack + Attack (Bad - no Normal)
- ❌ `[0, 0]` - Normal + Normal (Bad - no Attack) - Shouldn't happen but theoretically possible
- ❌ `[1, 3]` - Attack + Attack (Bad - no Normal)

**Result**: Unpredictable composition, may lack Normal or Attack samples entirely.

---

## 📊 **Summary**

### **With Equal Composition Enforcement (`enforce_equal_support_composition = True`):**

| Component | Count | Percentage |
|-----------|-------|------------|
| **Normal Samples** | `k_shot` | 50% |
| **Attack Samples** | `k_shot` | 50% |
| **Total Support** | `2 * k_shot` | 100% |

**Guaranteed:**
- ✅ Always has Normal samples
- ✅ Always has Attack samples
- ✅ Equal representation (when sufficient samples available)
- ✅ Zero-day attack excluded from training

**Example with `k_shot = 150`:**
- Support Set: 150 Normal + 150 Attack = **300 samples total**
- Composition: **50% Normal, 50% Attack** (Balanced)

---

## 🎯 **Why Equal Composition Matters**

1. **Balanced Learning**: Model sees equal Normal/Attack examples per task
2. **Prevents Bias**: Avoids overfitting to majority class
3. **Better Prototypes**: Balanced support sets create more accurate class prototypes
4. **Consistent Training**: Same composition across all meta-tasks
5. **Predictable Behavior**: Easier to debug and understand model behavior

---

## 🔧 **Configuration Impact**

The `enforce_equal_support_composition` parameter is now **optimizable** in Optuna:
- **`True`**: Guarantees balanced Normal/Attack support sets
- **`False`**: Random selection (may lack Normal or Attack)

Optuna will determine which setting performs better for your specific use case.










