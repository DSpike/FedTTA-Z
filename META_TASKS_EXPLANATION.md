# Meta-Tasks Configuration: How Many Are Used?

## 📊 **Number of Meta-Tasks**

### **Current Configuration:**

- **`num_meta_tasks: int = 20`** (per client, per federated round)

**Location**: `config.py`, line 221

```python
num_meta_tasks: int = 20  # Reduced from 50 to 20 for faster training
```

---

## 🎯 **Usage in Training**

### **Per Client Per Round:**

Each client creates **20 meta-tasks** from their local training data:

```python
# Location: coordinators/simple_fedavg_coordinator.py, line 2590
local_meta_tasks = create_meta_tasks(
    self.train_data,
    self.train_labels,
    n_way=self.config.n_way,              # 2 (Normal vs Attack)
    k_shot=self.config.k_shot,            # 150 (or optimized value)
    n_query=self.config.n_query,          # 10 (or optimized value)
    n_tasks=self.config.num_meta_tasks,   # 20 meta-tasks
    phase="training",
    normal_query_ratio=0.8,
    zero_day_attack_label=self.config.zero_day_attack_label,
    enforce_equal_support_composition=True
)
```

### **Total Meta-Tasks Across All Clients:**

If you have **N clients** and **R federated rounds**:

**Total Meta-Tasks = N clients × 20 tasks × R rounds**

**Example:**

- **8 clients** × **20 tasks** × **5 rounds** = **800 total meta-tasks** (across all training)
- **Per round**: **8 clients** × **20 tasks** = **160 meta-tasks**

---

## 📋 **What Each Meta-Task Contains**

Each of the 20 meta-tasks has:

1. **Support Set**:

   - `k_shot` Normal samples (e.g., 150)
   - `k_shot` Attack samples (e.g., 150)
   - **Total**: `2 × k_shot` = 300 samples

2. **Query Set**:
   - `n_query` samples per class (e.g., 10)
   - **Total**: `2 × n_query` = 20 samples
   - Natural distribution (not forced ratio)

**Total Samples Per Meta-Task**: `(2 × k_shot) + (2 × n_query)` = `300 + 20` = **320 samples**

---

## 🔢 **Total Training Samples Per Client**

Per client, per round:

- **Support Set**: `20 tasks × 300 samples` = **6,000 support samples**
- **Query Set**: `20 tasks × 20 samples` = **400 query samples**
- **Total**: **6,400 samples per client per round**

**Note**: These samples are drawn from the client's local training data, so there may be overlap across tasks.

---

## 📊 **Why 20 Meta-Tasks?**

The configuration comment states:

```python
num_meta_tasks: int = 20  # Reduced from 50 to 20 for faster training
```

**Reasons:**

1. **Training Speed**: 20 tasks per client is faster than 50
2. **Sufficient Diversity**: 20 tasks provide enough variation to learn robust patterns
3. **Balance**: Trade-off between training time and learning quality

---

## ⚙️ **Is This Optimizable?**

**Yes!** In `optimize_hyperparameters.py`, `num_meta_tasks` is **not currently in the search space**.

If you want to optimize it, you could add:

```python
# In optimize_hyperparameters.py, suggest_hyperparameters method
num_meta_tasks = trial.suggest_int('num_meta_tasks', 10, 50, step=5)
# Search space: 10, 15, 20, 25, 30, 35, 40, 45, 50
```

**Current Status**: Fixed at **20** (not optimized).

---

## 📈 **Summary**

| Parameter                              | Value                                               |
| -------------------------------------- | --------------------------------------------------- |
| **Meta-Tasks Per Client**              | 20                                                  |
| **Support Samples Per Task**           | `2 × k_shot` (e.g., 300)                            |
| **Query Samples Per Task**             | `2 × n_query` (e.g., 20)                            |
| **Total Samples Per Task**             | `(2 × k_shot) + (2 × n_query)` (e.g., 320)          |
| **Total Support Samples Per Client**   | `20 × (2 × k_shot)` (e.g., 6,000)                   |
| **Total Query Samples Per Client**     | `20 × (2 × n_query)` (e.g., 400)                    |
| **Total Samples Per Client Per Round** | `20 × [(2 × k_shot) + (2 × n_query)]` (e.g., 6,400) |

---

## 🔍 **Example Calculation**

With current optimized configuration:

- `k_shot = 100` (optimized)
- `n_query = 10` (optimized)
- `num_meta_tasks = 20`
- `num_clients = 8`
- `num_rounds = 5`

**Per Meta-Task:**

- Support: `2 × 100` = 200 samples
- Query: `2 × 10` = 20 samples
- **Total**: 220 samples

**Per Client Per Round:**

- Total: `20 × 220` = **4,400 samples**

**Total Across All Training:**

- `8 clients × 20 tasks × 5 rounds × 220 samples` = **176,000 sample-task interactions**
- (Note: This counts interactions, not unique samples due to sampling with replacement)

---

## 📝 **Note on Sampling**

Meta-tasks are created by **randomly sampling** from the client's local data:

- **Support Set**: Randomly selects `k_shot` Normal and `k_shot` Attack samples
- **Query Set**: Randomly selects query samples with natural distribution
- **Overlap**: Different tasks may share some samples (sampling with replacement)

This ensures:

- ✅ Diverse task composition
- ✅ Different attack types across tasks
- ✅ Robust learning from various data combinations









