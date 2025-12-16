# Including All Attack Types in Each Meta-Task

## 🎯 **Goal**

Currently, each meta-task includes:

- ✅ Normal samples (label 0)
- ✅ ONE randomly selected attack class (e.g., Fuzzers OR Generic OR Exploits)

**Desired**: Include **ALL available attack types** in each meta-task along with Normal samples.

---

## 🔍 **Current Implementation**

### **Current Behavior:**

```python
# Line 1501-1510: Selects ONE random attack class
if len(attack_labels) > 0:
    attack_label_idx = torch.randint(0, len(attack_labels), (1,))
    selected_attack_label = attack_labels[attack_label_idx]  # Only ONE attack class
    selected_labels = torch.cat([normal_label, selected_attack_label])  # [0, attack_class]
```

**Result**: Each meta-task has Normal + ONE attack type

---

## ✅ **Solution Options**

### **Option 1: Binary Classification with Mixed Attack Samples (Recommended)**

**Approach**: Keep `n_way=2` (binary classification) but sample Attack support set from **ALL attack types** (proportionally or uniformly).

**Support Set Composition:**

- `k_shot` Normal samples (label 0)
- `k_shot` Attack samples **drawn from ALL available attack types** (labels 1,2,4,5,6,7,8,9 - excluding zero-day 3)
  - Distribution: Proportional to available samples OR uniform across types

**Implementation:**

```python
# Modified logic in create_meta_tasks function
if n_way == 2 and enforce_equal_support_composition:
    # Normal (0) must always be selected
    normal_label = torch.tensor([0], dtype=available_labels.dtype, device=available_labels.device)

    # Get ALL available attack labels (exclude Normal=0 and zero-day)
    if zero_day_attack_label is not None:
        all_attack_labels = available_labels[(available_labels != 0) & (available_labels != zero_day_attack_label)]
    else:
        all_attack_labels = available_labels[available_labels != 0]

    # NEW: Select Normal + ALL attack types
    selected_labels = torch.cat([normal_label, all_attack_labels])  # [0, 1, 2, 4, 5, 6, 7, 8, 9]

    # For support set creation:
    # Normal: k_shot samples from label 0
    # Attack: k_shot samples TOTAL, distributed across all attack types

    # Create support set
    support_x_list = []
    support_y_list = []

    # 1. Add Normal samples (k_shot)
    normal_mask = data_y == 0
    normal_indices = torch.where(normal_mask)[0]
    if len(normal_indices) >= k_shot:
        shuffled_normal = normal_indices[torch.randperm(len(normal_indices))][:k_shot]
        support_x_list.append(data_x[shuffled_normal])
        support_y_list.append(data_y[shuffled_normal])
    else:
        support_x_list.append(data_x[normal_indices])
        support_y_list.append(data_y[normal_indices])

    # 2. Add Attack samples (k_shot TOTAL, distributed across all attack types)
    # Option A: Uniform distribution (each attack type gets k_shot / num_attack_types)
    # Option B: Proportional distribution (each attack type contributes proportionally)

    # Option A: Uniform (recommended for balanced exposure)
    num_attack_types = len(all_attack_labels)
    samples_per_attack_type = max(1, k_shot // num_attack_types)  # At least 1 per type
    remaining_samples = k_shot - (samples_per_attack_type * num_attack_types)

    attack_x_list = []
    attack_y_list = []

    for attack_label in all_attack_labels:
        attack_mask = data_y == attack_label
        attack_indices = torch.where(attack_mask)[0]

        if len(attack_indices) >= samples_per_attack_type:
            shuffled_attack = attack_indices[torch.randperm(len(attack_indices))][:samples_per_attack_type]
            attack_x_list.append(data_x[shuffled_attack])
            attack_y_list.append(data_y[shuffled_attack])
        else:
            # Use all available
            attack_x_list.append(data_x[attack_indices])
            attack_y_list.append(data_y[attack_indices])

    # Add remaining samples (if any) to balance
    if remaining_samples > 0:
        # Randomly select from all attack types for remaining samples
        all_attack_mask = (data_y != 0) & (data_y != zero_day_attack_label)
        all_attack_indices = torch.where(all_attack_mask)[0]
        if len(all_attack_indices) >= remaining_samples:
            shuffled_remaining = all_attack_indices[torch.randperm(len(all_attack_indices))][:remaining_samples]
            attack_x_list.append(data_x[shuffled_remaining])
            attack_y_list.append(data_y[shuffled_remaining])

    # Combine all attack samples
    if attack_x_list:
        support_x_list.append(torch.cat(attack_x_list, dim=0))
        support_y_list.append(torch.cat(attack_y_list, dim=0))

    # Final support set
    support_x = torch.cat(support_x_list, dim=0)
    support_y = torch.cat(support_y_list, dim=0)

    # Important: For binary classification, remap all attack labels to 1
    # OR keep original labels (multi-class) but use binary classifier
    # Decision needed: Keep multi-class labels or remap to binary?
```

**Pros:**

- ✅ Maintains binary classification (`n_way=2`)
- ✅ Model sees ALL attack types in every meta-task
- ✅ Minimal architecture changes needed
- ✅ Balanced exposure to all attack patterns

**Cons:**

- ⚠️ Attack samples have different labels (1,2,4,5,6,7,8,9) but classifier outputs binary (Normal/Attack)
- ⚠️ Need to decide: Keep multi-class labels OR remap all attacks to label 1?

**Decision Needed:**

1. **Keep multi-class attack labels** (1,2,4,5,6,7,8,9) → Model sees variety, but classifier treats all as "Attack"
2. **Remap all attacks to label 1** → Cleaner binary classification, but loses attack type information

---

### **Option 2: Multi-Way Classification (n_way=9)**

**Approach**: Change to 9-way classification (Normal + 8 attack types).

**Support Set Composition:**

- `k_shot` Normal samples (label 0)
- `k_shot` Fuzzers samples (label 1)
- `k_shot` Analysis samples (label 2)
- `k_shot` DoS samples (label 4)
- `k_shot` Exploits samples (label 5)
- `k_shot` Generic samples (label 6)
- `k_shot` Reconnaissance samples (label 7)
- `k_shot` Shellcode samples (label 8)
- `k_shot` Worms samples (label 9)
- **Total**: `9 × k_shot` support samples

**Implementation:**

```python
# Change n_way from 2 to 9 in config.py
n_way: int = 9  # Normal + 8 attack types

# Modify create_meta_tasks:
if n_way == 9 and enforce_all_attack_types:
    # Always select: Normal + ALL 8 attack types
    normal_label = torch.tensor([0], dtype=available_labels.dtype, device=available_labels.device)

    # Get ALL available attack labels (exclude zero-day)
    if zero_day_attack_label is not None:
        all_attack_labels = available_labels[(available_labels != 0) & (available_labels != zero_day_attack_label)]
    else:
        all_attack_labels = available_labels[available_labels != 0]

    selected_labels = torch.cat([normal_label, all_attack_labels])  # [0, 1, 2, 4, 5, 6, 7, 8, 9]

    # Create support set for each class (9 classes total)
    support_x_list = []
    support_y_list = []

    for label in selected_labels:
        class_mask = data_y == label
        class_indices = torch.where(class_mask)[0]

        if len(class_indices) >= k_shot:
            shuffled_indices = class_indices[torch.randperm(len(class_indices))][:k_shot]
            support_x_list.append(data_x[shuffled_indices])
            support_y_list.append(data_y[shuffled_indices])
        else:
            support_x_list.append(data_x[class_indices])
            support_y_list.append(data_y[class_indices])

    support_x = torch.cat(support_x_list, dim=0)
    support_y = torch.cat(support_y_list, dim=0)
```

**Required Changes:**

1. **Config**: Change `n_way` from 2 to 9
2. **Model Architecture**: Change classifier output from 2 to 9 classes
   ```python
   # In TransductiveLearner.__init__
   self.classifier = nn.Linear(embedding_dim, num_classes)  # Change num_classes from 2 to 9
   ```
3. **Loss Function**: FocalLoss already supports multi-class, but verify
4. **Evaluation**: Update evaluation to handle 9-class predictions (may need to aggregate to binary for comparison)

**Pros:**

- ✅ Model learns to distinguish between attack types
- ✅ All attack types always present in every meta-task
- ✅ More fine-grained learning

**Cons:**

- ⚠️ Major architecture change (classifier output size)
- ⚠️ 9× more support samples per task (9 × k_shot vs 2 × k_shot)
- ⚠️ May require more training data per client
- ⚠️ Evaluation becomes multi-class (need aggregation to binary for comparison)

---

### **Option 3: Hybrid Approach (Recommended for Binary Classification)**

**Approach**: Keep `n_way=2` but ensure Attack samples come from ALL attack types proportionally.

**Support Set Composition:**

- `k_shot` Normal samples (label 0)
- `k_shot` Attack samples, **uniformly distributed** across all 8 attack types
  - Each attack type contributes: `k_shot // 8` samples (or proportionally)
  - Total: `k_shot` Attack samples with variety

**Key Difference from Option 1:**

- Remap all attack labels to 1 for binary classification
- But sample from ALL attack types to ensure diversity

**Implementation:**

```python
# Modified create_meta_tasks for hybrid approach
if n_way == 2 and enforce_equal_support_composition and include_all_attack_types:
    # Normal (0)
    normal_label = torch.tensor([0], dtype=available_labels.dtype, device=available_labels.device)

    # Get ALL available attack labels
    if zero_day_attack_label is not None:
        all_attack_labels = available_labels[(available_labels != 0) & (available_labels != zero_day_attack_label)]
    else:
        all_attack_labels = available_labels[available_labels != 0]

    # Create support set
    support_x_list = []
    support_y_list = []

    # 1. Normal samples (k_shot)
    normal_mask = data_y == 0
    normal_indices = torch.where(normal_mask)[0]
    if len(normal_indices) >= k_shot:
        shuffled_normal = normal_indices[torch.randperm(len(normal_indices))][:k_shot]
        support_x_list.append(data_x[shuffled_normal])
        # Keep label 0 for Normal
        support_y_list.append(data_y[shuffled_normal])
    else:
        support_x_list.append(data_x[normal_indices])
        support_y_list.append(data_y[normal_indices])

    # 2. Attack samples (k_shot TOTAL, from all attack types)
    num_attack_types = len(all_attack_labels)
    samples_per_type = k_shot // num_attack_types  # Uniform distribution
    remaining = k_shot % num_attack_types

    attack_x_list = []
    attack_y_list = []

    for i, attack_label in enumerate(all_attack_labels):
        # Add one extra sample to first 'remaining' attack types for balance
        samples_needed = samples_per_type + (1 if i < remaining else 0)

        attack_mask = data_y == attack_label
        attack_indices = torch.where(attack_mask)[0]

        if len(attack_indices) >= samples_needed:
            shuffled = attack_indices[torch.randperm(len(attack_indices))][:samples_needed]
            attack_x_list.append(data_x[shuffled])
            # IMPORTANT: Remap all attack labels to 1 for binary classification
            attack_y_list.append(torch.ones(samples_needed, dtype=data_y.dtype, device=data_y.device) * 1)
        else:
            attack_x_list.append(data_x[attack_indices])
            attack_y_list.append(torch.ones(len(attack_indices), dtype=data_y.dtype, device=data_y.device) * 1)

    # Combine attack samples
    if attack_x_list:
        support_x_list.append(torch.cat(attack_x_list, dim=0))
        support_y_list.append(torch.cat(attack_y_list, dim=0))

    # Final support set
    support_x = torch.cat(support_x_list, dim=0)
    support_y = torch.cat(support_y_list, dim=0)

    # Verify: support_y should contain only 0 (Normal) and 1 (Attack)
    # But attack samples come from ALL attack types for diversity
```

**Pros:**

- ✅ Maintains binary classification (`n_way=2`)
- ✅ All attack types included in every meta-task
- ✅ Clean binary labels (0=Normal, 1=Attack)
- ✅ Minimal architecture changes

**Cons:**

- ⚠️ Loses attack type distinction (all mapped to 1)
- ⚠️ Slightly more complex sampling logic

---

## 📊 **Comparison Table**

| Approach                   | n_way | Support Samples | Attack Labels                  | Architecture Change | Complexity |
| -------------------------- | ----- | --------------- | ------------------------------ | ------------------- | ---------- |
| **Current**                | 2     | `2 × k_shot`    | ONE random type                | None                | Low        |
| **Option 1**               | 2     | `2 × k_shot`    | ALL types (multi-class labels) | Minimal             | Medium     |
| **Option 2**               | 9     | `9 × k_shot`    | ALL types (separate labels)    | Major               | High       |
| **Option 3 (Recommended)** | 2     | `2 × k_shot`    | ALL types (binary labels)      | Minimal             | Low-Medium |

---

## 🎯 **Recommendation**

**For Binary Classification Goal**: Use **Option 3 (Hybrid Approach)**

- Maintains `n_way=2` (binary classification)
- Includes all attack types in every meta-task
- Minimal code changes
- Clean binary labels

**For Multi-Class Goal**: Use **Option 2**

- Better attack type discrimination
- Requires architecture changes
- More support samples needed

---

## 🔧 **Implementation Steps for Option 3**

1. **Add configuration flag** in `config.py`:

   ```python
   include_all_attack_types_in_support: bool = True  # Include all attack types in support set
   ```

2. **Modify `create_meta_tasks` function** in `models/transductive_fewshot_model.py`:

   - Add parameter: `include_all_attack_types: bool = False`
   - Implement hybrid sampling logic (as shown above)

3. **Update function calls** in `main.py` and `coordinators/simple_fedavg_coordinator.py`:

   ```python
   local_meta_tasks = create_meta_tasks(
       ...,
       include_all_attack_types=self.config.include_all_attack_types_in_support
   )
   ```

4. **Test and verify**:
   - Check support set contains samples from all attack types
   - Verify labels are binary (0 or 1)
   - Ensure training works correctly

---

## 📝 **Next Steps**

Would you like me to:

1. Implement **Option 3 (Hybrid Approach)**?
2. Implement **Option 2 (Multi-Way Classification)**?
3. Show you the exact code changes needed?









