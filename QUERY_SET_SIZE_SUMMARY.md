# Query Set Size Summary

## 📊 **Query Set Sizes in Your System**

There are **TWO different query sets** used at different stages:

---

### **1. Meta-Learning Query Set (During Training)**

**Configuration:**
- `n_query: int = 20` (from `config.py` line 229)

**Purpose:**
- Used during federated meta-learning training
- Small query set per meta-task (20 samples)
- Used for transductive learning within each task

**Usage:**
- Each meta-task has a support set + query set
- Query set: 20 samples per task
- Total queries across all tasks: `num_meta_tasks × n_query = 34 × 20 = 680` queries (distributed across tasks)

---

### **2. TTT Adaptation Query Set (During Test-Time Training)**

**Configuration:**
- `ttt_adaptation_query_size: int = 1198` (from `config.py` line 104)

**Purpose:**
- Used for test-time training (TTT) adaptation on test data
- Larger query set for better adaptation

**Actual Size Used (From Full Run):**
- **Actual query set size: 736 samples**
- Limited by test set size: `min(1198, len(X_test)) = 736`

**Composition:**
- **Zero-day samples**: 184 (25.0%)
- **Non-zero-day samples**: 552 (75.0%)
- **Total**: **736 samples**

**From Code** (`main.py` lines 3374-3379):
```python
ttt_query_size = getattr(self.config, 'ttt_adaptation_query_size', 750)
query_size = min(ttt_query_size, len(X_test))  # Limited by test set size
query_indices = torch.randperm(len(X_test))[:query_size]
query_x = torch.FloatTensor(X_test[query_indices]).to(self.device)
```

**Note:** The config specifies 1198, but the actual test set only has 736 samples, so all 736 samples are used for TTT adaptation.

---

## 📋 **Summary Table**

| Query Set Type | Config Value | Actual Size | Purpose |
|----------------|--------------|-------------|---------|
| **Meta-Learning** | `n_query = 20` | 20 per task | Training (transductive meta-learning) |
| **TTT Adaptation** | `ttt_adaptation_query_size = 1198` | **736 samples** | Test-time training adaptation |

---

## 🎯 **Key Points**

1. **Meta-Learning Query Set**: Small (20 samples per task) - used during federated training
2. **TTT Adaptation Query Set**: Large (**736 samples**) - uses the full test set for adaptation
3. **TTT Query Set Composition**: 184 zero-day (25%) + 552 non-zero-day (75%)
4. **Test Set Size**: 736 samples total (matches TTT query set)

---

## ✅ **From Latest Full Run**

**TTT Adaptation Query Set:**
- **Size**: 736 samples (full test set)
- **Zero-day**: 184 samples (25.0%)
- **Non-zero-day**: 552 samples (75.0%)
- **Source**: Complete test set (no further sampling)

This is the query set used for TTT adaptation and final evaluation! 🎯









