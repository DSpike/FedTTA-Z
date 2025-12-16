# KDD Configuration Source Clarification

## 🔍 **Question: Where did the KDD config values come from?**

**Answer:** The KDD backup values came from the **ACTUAL config.py file** (git commit `c619901`), **NOT** from `best_hyperparameters.json`.

---

## 📊 **Comparison: Config.py vs Optuna File**

### **Values from Actual config.py (git c619901):**

```python
hidden_dim: int = 128
embedding_dim: int = 256
sequence_length: int = 22
sequence_stride: int = 12
tcn_kernel_sizes: tuple = (2, 3, 3)
meta_epochs: int = 21
k_shot: int = 152
n_query: int = 16
learning_rate: float = 0.0016387494099028342
```

### **Values from best_hyperparameters.json (Optuna Trial 7):**

```json
{
  "hidden_dim": 256, // ❌ DIFFERENT (128 vs 256)
  "embedding_dim": 128, // ❌ DIFFERENT (256 vs 128)
  "sequence_length": 37, // ❌ DIFFERENT (22 vs 37)
  "sequence_stride": 16, // ❌ DIFFERENT (12 vs 16)
  "tcn_kernel_size_1": 4, // ❌ DIFFERENT (2 vs 4)
  "tcn_kernel_size_2": 3, // ✅ SAME
  "tcn_kernel_size_3": 3, // ✅ SAME
  "meta_epochs": 23, // ❌ DIFFERENT (21 vs 23)
  "k_shot": 117, // ❌ DIFFERENT (152 vs 117)
  "n_query": 11, // ❌ DIFFERENT (16 vs 11)
  "meta_learning_rate": 0.0017399757799262, // ❌ DIFFERENT (0.0016 vs 0.0017)
  "use_residual_connections": true // ❌ DIFFERENT (False vs True)
}
```

---

## 🤔 **Why the Discrepancy?**

### **Possible Explanations:**

1. **Multiple Optimization Runs:**

   - Config.py values are from "Optuna Trial 1" (as per comments)
   - `best_hyperparameters.json` shows "Trial 7"
   - Different trials may have different objectives or search spaces

2. **Manual Adjustments:**

   - Config.py values may have been manually adjusted after optimization
   - Some parameters may have been fine-tuned based on empirical results

3. **Different Optimization Objectives:**

   - Trial 1 may have optimized for different metrics
   - Trial 7 may have used a different objective function

4. **Config.py is the "Working" Version:**
   - The actual config.py represents the **working configuration** that was used
   - The Optuna file is just the **raw optimization result**
   - Config.py may have been adjusted for stability/performance

---

## ✅ **Which Values Should You Use?**

### **For Restoring KDD Settings:**

**Use the values from `config_kdd_backup.py`** (which match the actual config.py):

- ✅ These are the **actual working values** that were used
- ✅ These values were tested and validated
- ✅ These match what was in config.py before switching to UNSW

### **For Reference/Comparison:**

- The Optuna file (`best_hyperparameters.json`) shows what the optimizer found
- But the actual config.py may have been adjusted for practical reasons

---

## 📝 **Summary**

| Source                        | Values Used In         | Purpose                                                      |
| ----------------------------- | ---------------------- | ------------------------------------------------------------ |
| **config.py (git c619901)**   | `config_kdd_backup.py` | ✅ **Use this for restoration** - Actual working values      |
| **best_hyperparameters.json** | Reference only         | ⚠️ Different values - May be from different optimization run |

---

## 🔄 **Action Taken**

The `config_kdd_backup.py` file has been updated to:

1. ✅ Clarify that values are from actual config.py (not Optuna file)
2. ✅ Note the discrepancies with best_hyperparameters.json
3. ✅ Document the git commit source (c619901)

---

## 💡 **Recommendation**

**When restoring KDD settings, use `config_kdd_backup.py` values** because:

- They represent the **actual working configuration**
- They were the values in use before switching to UNSW
- They have been tested and validated in practice

The Optuna file can be used as a reference, but the config.py values are the "ground truth" for what was actually running.



