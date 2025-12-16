# Test Set Zero-Day Sample Inclusion

## ✅ **YES - Test Set DOES Contain Zero-Day Samples**

The test set is **designed to include zero-day samples** as part of a balanced composition.

---

## 📊 **Test Set Target Composition**

According to the code in `main.py` (`_stratified_test_subset` function, line 558):

### **Target Distribution:**
- **40% Normal/BENIGN samples**
- **35% Non-zero-day attacks** (known attacks seen during training)
- **25% Zero-day attacks** (unseen attacks - Backdoor in your case)

### **Code Reference:**
```python
# TARGET: 40% Normal, 35% Non-zero-day attacks, 25% Zero-day attacks
zero_day_target_percentage = getattr(self, '_temp_zero_day_target', 0.25)  # 25% zero-day target
```

---

## 🔍 **How Zero-Day Samples Are Selected**

### **1. Pre-Sequence Sampling (Before Sequence Creation)**
- **Target:** 30% zero-day (higher target to compensate for sequence creation dilution)
- **Process:**
  1. Identifies zero-day samples using multiclass labels or attack categories
  2. Randomly selects zero-day samples to meet target percentage
  3. Fills remaining slots with non-zero-day samples (stratified sampling)

**Code Location:** `main.py` line 905-906
```python
self._temp_zero_day_target = 0.30  # Target 30% before sequences
X_test_subset, y_test_subset, y_test_multiclass_original, test_attack_cat_original = self._stratified_test_subset(...)
```

### **2. Post-Sequence Filtering (After Sequence Creation)**
- **Target:** 25% zero-day sequences
- **Process:**
  1. Creates sequences from test subset
  2. Filters sequences to maintain target distribution (25% zero-day, 75% non-zero-day)
  3. Ensures balanced composition after sequence creation

**Code Location:** `main.py` line 963
```python
target_zero_day_percentage = 0.25  # Target 25% zero-day (40% Normal, 35% Non-zero-day, 25% Zero-day)
```

---

## 📈 **Actual Composition in Your Runs**

Based on the code logic and your Backdoor run:

### **Expected Test Set Composition:**
```
Total Test Samples: ~140 sequences
├── 40% Normal (BENIGN): ~56 sequences
├── 35% Non-zero-day attacks: ~49 sequences  
└── 25% Zero-day attacks (Backdoor): ~35 sequences ✅
```

### **From Your Recent Backdoor Run:**
- **Total samples:** 140
- **Zero-day samples:** 35 (25%)
- **Normal samples:** 64 (45.7%) 
- **Non-zero-day attacks:** 41 (29.3%)

**Note:** Your actual distribution shows slightly more Normal (45.7% vs 40% target) and slightly less zero-day (25% vs target), but this is within acceptable variance.

---

## 🔬 **Verification in Code**

### **Zero-Day Sample Selection Logic:**
```python
# Get indices of zero-day and non-zero-day samples
if y_multiclass_np is not None:
    zero_day_indices = np.where(y_multiclass_np == zero_day_label)[0]
    non_zero_day_indices = np.where(y_multiclass_np != zero_day_label)[0]

# Select zero-day samples (target already adjusted based on availability)
if available_zero_day > 0:
    actual_zero_day_count = min(zero_day_target_count, available_zero_day)
    # Randomly select zero-day samples
    np.random.seed(42)
    selected_zero_day_indices = np.random.choice(zero_day_indices, size=actual_zero_day_count, replace=False)
```

### **Logging:**
The system logs the zero-day composition:
```python
logger.info(f"   Zero-day samples: {zero_day_count}/{len(X_subset)} ({actual_percentage:.1f}%) [TARGET: {100*zero_day_target_percentage:.1f}%]")
```

---

## ✅ **Why Test Set Contains Zero-Day Samples**

### **1. Zero-Day Detection Evaluation**
- Test set is the **ONLY dataset** that contains zero-day samples
- Used to evaluate zero-day detection capability
- Validates that TTT adaptation improves zero-day detection

### **2. Realistic Evaluation**
- Real-world scenarios include both known and unknown attacks
- Test set composition (40% Normal, 35% Known, 25% Zero-day) reflects realistic deployment

### **3. Fair Comparison**
- Both base model and TTT model are evaluated on the same test set
- Allows direct comparison of zero-day detection improvement

---

## 🎯 **Separation of Concerns**

### **Validation Set:**
- ❌ **NO zero-day samples**
- Only contains Normal + Known attacks
- Used for training monitoring and hyperparameter tuning

### **Test Set:**
- ✅ **CONTAINS zero-day samples** (25% target)
- Contains Normal + Known + Zero-day attacks
- Used for final zero-day detection evaluation

### **Training Set:**
- ❌ **NO zero-day samples** (except 1% leakage for realism)
- Only contains Normal + Known attacks
- Model is trained without seeing zero-day attacks

---

## 📊 **How to Verify in Your Run**

Look for these log messages in your output:

```
🔍 Stratified test subset: 140 samples
   Zero-day samples: 35/140 (25.0%) [TARGET: 25.0%]
```

Or during sequence filtering:
```
✅ After post-sequence filtering: 35/140 zero-day sequences (25.0%) [TARGET: 25%]
```

---

## ✅ **Summary**

| Question | Answer |
|----------|--------|
| **Does test set contain zero-day samples?** | ✅ **YES** - Approximately 25% of test set |
| **What is the target percentage?** | **25% zero-day** (40% Normal, 35% Non-zero-day, 25% Zero-day) |
| **How are zero-day samples selected?** | Randomly selected from available zero-day samples using stratified sampling |
| **Is this the only dataset with zero-day?** | ✅ **YES** - Training and validation sets exclude zero-day samples |
| **What is the purpose?** | Evaluate zero-day detection capability of base and TTT models |

---

**Conclusion:** The test set **definitely contains zero-day samples** (target: 25%) as part of the evaluation strategy for zero-day attack detection.

---

*Documentation Date: December 2, 2025*  
*Code Reference: `main.py` lines 539-724, 898-1024*









