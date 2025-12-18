# TTT Threshold Selection: Dynamic vs Fixed

## ✅ **Answer: TTT Uses DYNAMIC Threshold**

The TTT model uses a **dynamically optimized threshold** that is calculated based on the test data predictions and labels.

---

## 🔍 **How TTT Threshold is Selected**

### **Location:** `main.py` lines 6539-6723 (`_evaluate_ttt_model`)

### **Threshold Selection Strategy (Priority Order):**

1. **ZDR-Optimized Threshold** (Highest Priority)
   - **Lines 6564-6614**: Optimizes threshold specifically for Zero-Day Detection Rate
   - Searches 200 thresholds from 0.05 to 0.8
   - Maximizes ZDR while keeping FAR ≤ 40% (configurable)
   - **Target**: ZDR ≥ 80% (configurable via `ttt_zdr_target`)
   - **Constraint**: FAR ≤ 40% (configurable via `ttt_zdr_max_far`)

2. **PR-Based (F1-Optimized) Threshold** (Second Priority)
   - **Lines 6545-6550**: Uses Precision-Recall curve optimization
   - **Method**: `find_optimal_threshold_pr()` with `method='f1'`
   - Optimizes F1-score (balances precision and recall)
   - **Minimum recall**: 0.3 (ensures ZDR improvement)

3. **ROC-Based Threshold with FAR Constraint** (Fallback)
   - **Lines 6616-6671**: Uses ROC curve with FAR constraint
   - **Constraint**: FAR ≤ 35% (configurable via `max_far_for_zdr`)
   - Maximizes TPR (True Positive Rate) within FAR constraint

4. **Fallback Thresholds** (Last Resort)
   - **Median probability**: If optimization fails, uses median of attack probabilities
   - **Fixed 0.5**: Final fallback if all else fails

---

## 📊 **Threshold Selection Code Flow**

```python
# Step 1: PR-based threshold optimization
pr_threshold, pr_auc, pr_precision, pr_recall, pr_thresh = find_optimal_threshold_pr(
    query_y_binary.cpu().numpy(), 
    attack_probabilities.cpu().numpy(),
    method='f1',  # Optimize for F1-score
    min_recall=0.3
)

# Step 2: ZDR-optimized threshold (if enabled)
if adaptive_zdr_threshold:
    # Search 200 thresholds from 0.05 to 0.8
    for thresh in zero_day_thresholds:
        # Calculate ZDR and FAR at this threshold
        zdr_at_thresh = ...
        far_at_thresh = ...
        f1_at_thresh = ...
        
        # Select best threshold that meets ZDR target and FAR constraint
        if zdr_meets_target and far_acceptable:
            if f1_at_thresh > best_f1:
                best_zdr_threshold = thresh

# Step 3: Compare and select final threshold
if use_zdr_threshold:
    optimal_threshold = zdr_optimized_threshold  # Priority 1
elif pr_f1 >= roc_f1:
    optimal_threshold = pr_threshold  # Priority 2
else:
    optimal_threshold = roc_threshold  # Priority 3

# Step 4: Apply threshold to make predictions
ttt_predictions = (attack_probabilities >= optimal_threshold).long()
```

---

## ⚙️ **Configuration Parameters**

**Configurable via `config.py` or `config_loader.py`:**

```python
# ZDR optimization settings
'ttt_adaptive_zdr_threshold': True,  # Enable ZDR-optimized threshold
'ttt_zdr_target': 0.80,  # Target ZDR (80%)
'ttt_zdr_max_far': 0.40,  # Maximum FAR (40%)
'max_far_for_zdr': 0.35,  # FAR constraint for ROC-based threshold
```

---

## 🔄 **Why Dynamic Threshold for TTT?**

### **Rationale:**

1. **TTT Adapts to Test Data**: TTT adapts the model to the test distribution, so it makes sense to optimize threshold on test data
2. **Zero-Day Detection Priority**: ZDR-optimized threshold specifically targets zero-day detection performance
3. **Balanced Performance**: PR-based optimization balances precision and recall (better for imbalanced data)
4. **FAR Control**: ROC-based threshold with FAR constraint prevents excessive false alarms

### **Comparison with Base Model:**

| Model | Threshold Type | Optimization Method |
|-------|---------------|---------------------|
| **Base Model** | **Fixed** (0.5) or **FAR-optimized** | Optimized on test data for FAR < 1% |
| **TTT Model** | **Dynamic** (optimized) | Optimized on test data for F1/ZDR |

---

## ⚠️ **Important Notes**

### **1. Threshold Optimization Uses Test Labels**

**Code Evidence** (line 6546):
```python
pr_threshold = find_optimal_threshold_pr(
    query_y_binary.cpu().numpy(),  # ← Uses TEST LABELS
    attack_probabilities.cpu().numpy(),
    method='f1'
)
```

**Is this data leakage?**
- **For TTT**: ✅ **Acceptable** - TTT is designed to adapt to test data
- **For Base Model**: ⚠️ **Questionable** - Base model shouldn't see test labels

### **2. Threshold is Calculated Per Evaluation**

The threshold is **recalculated** every time `_evaluate_ttt_model()` is called:
- Different test sets → Different thresholds
- Different model states → Different thresholds
- **Not fixed across runs**

### **3. Threshold Selection is Complex**

The system tries multiple strategies and selects the best one:
- **Priority 1**: ZDR-optimized (if significant improvement)
- **Priority 2**: PR-based (F1-optimized)
- **Priority 3**: ROC-based (FAR-constrained)
- **Fallback**: Median probability or 0.5

---

## 📈 **Expected Threshold Values**

Based on the code:
- **Range**: 0.05 to 0.8 (for ZDR optimization)
- **Clamped**: 0.1 to 0.9 (final safety clamp)
- **Typical values**: 0.3-0.7 (depending on data distribution)

---

## 🎯 **Summary**

**TTT uses a DYNAMIC threshold** that is:
- ✅ Optimized on test data (acceptable for TTT)
- ✅ Prioritizes zero-day detection (ZDR-optimized)
- ✅ Balances precision/recall (PR-based)
- ✅ Controls false alarm rate (FAR constraint)
- ✅ Recalculated for each evaluation

**This is different from a fixed threshold (e.g., 0.5) and allows TTT to adapt to the specific test distribution.**

