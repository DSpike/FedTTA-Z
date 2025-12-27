# Metrics Prediction Method Analysis

## 🔍 **Question**: Do zero-day metrics (accuracy, precision, F1, recall) and known attack metrics use **threshold** or **argmax** predictions?

---

## 📊 **Analysis Results**

### **1. Base Model - Zero-Day Metrics**

**Location**: `main.py` lines 3622-3636

**Method**: **ARGMAX** (converted to binary)

```python
# Line 3626-3630
zero_day_predictions_valid = zero_day_predictions.cpu().numpy()[zero_day_valid_mask]
zero_day_actual_valid = zero_day_actual.cpu().numpy()[zero_day_valid_mask]
# Convert multiclass predictions (0-9) to binary (0=Normal, 1=Attack)
zero_day_y_true_bin = (zero_day_actual_valid != 0).astype(int)
zero_day_y_pred_bin = (zero_day_predictions_valid != 0).astype(int)  # ARGMAX-based

# Calculate metrics using ARGMAX-based binary predictions
zero_day_accuracy = (torch.tensor(zero_day_y_pred_bin) == torch.tensor(zero_day_y_true_bin)).float().mean().item()
zero_day_precision = precision_score(zero_day_y_true_bin, zero_day_y_pred_bin, zero_division=0)
zero_day_recall = recall_score(zero_day_y_true_bin, zero_day_y_pred_bin, zero_division=0)
zero_day_f1 = f1_score(zero_day_y_true_bin, zero_day_y_pred_bin, zero_division=0)
```

**Source of `zero_day_predictions`**: 
- Line 3352: `base_predictions = torch.argmax(base_logits, dim=1)` - **ARGMAX**

**Result**: ✅ Base model zero-day metrics use **ARGMAX** predictions

---

### **2. TTT Model - Zero-Day Metrics**

**Location**: `main.py` lines 7213-7239

**Method**: **THRESHOLD-BASED**

```python
# Line 7231-7232
zero_day_predictions = ttt_predictions_np[is_zero_day_np]  # Uses ttt_predictions_np
zero_day_actual = query_y_np[is_zero_day_np]

# Line 7237-7239: ZDR calculation
zero_day_tp = ((zero_day_predictions == 1) & (zero_day_actual == 1)).sum()
zero_day_fn = ((zero_day_predictions == 0) & (zero_day_actual == 1)).sum()
zero_day_detection_rate = zero_day_tp / (zero_day_tp + zero_day_fn) if (zero_day_tp + zero_day_fn) > 0 else 0.0
```

**Source of `ttt_predictions_np`**:
- Line 6851: `ttt_predictions = (attack_probabilities >= optimal_threshold).long()` - **THRESHOLD-BASED**
- Line 6854: `ttt_predictions_np = ttt_predictions.cpu().numpy()`

**Result**: ✅ TTT model zero-day metrics use **THRESHOLD-BASED** predictions

**Note**: The code only calculates **ZDR** for zero-day samples, not accuracy/precision/F1/recall separately. These would need to be added.

---

### **3. Overall Metrics (All Samples)**

**Location**: `main.py` lines 7180-7194

**TTT Model Metrics**:
- Line 7181: `ttt_accuracy = accuracy_score(query_y_np[ttt_valid_mask_for_metrics], ttt_predictions_np[ttt_valid_mask_for_metrics])`
- Uses `ttt_predictions_np` which is **THRESHOLD-BASED** (from line 6851)

**Base Model Metrics**:
- Line 7184: `base_accuracy = accuracy_score(query_y_np[ttt_valid_mask_for_metrics], base_predictions_np[ttt_valid_mask_for_metrics])`
- Uses `base_predictions_np` which comes from:
  - Line 6471: `base_predictions = torch.argmax(base_logits, dim=1)` - **ARGMAX**

**Result**: 
- ✅ TTT overall metrics use **THRESHOLD-BASED** predictions
- ✅ Base overall metrics use **ARGMAX** predictions

---

### **4. Known Attack Metrics**

**Location**: Not explicitly calculated separately, but included in overall metrics

**Method**: Same as overall metrics
- **Base model**: **ARGMAX** (line 6471)
- **TTT model**: **THRESHOLD-BASED** (line 6851)

---

## ⚠️ **CRITICAL INCONSISTENCY IDENTIFIED**

### **Problem**: Base and TTT models use **different prediction methods**!

| Metric Type | Base Model | TTT Model |
|------------|-----------|-----------|
| **Zero-Day Metrics** | ARGMAX | THRESHOLD |
| **Overall Metrics** | ARGMAX | THRESHOLD |
| **Known Attack Metrics** | ARGMAX | THRESHOLD |

### **Impact**:

1. **Metrics are NOT directly comparable** between base and TTT models
2. **Base model** uses argmax (multiclass → binary conversion)
3. **TTT model** uses threshold (probability-based binary classification)
4. **This explains why**:
   - Scatter plot shows good separation (uses argmax during TTT adaptation)
   - ZDR is zero (uses threshold, which may be too high for zero-day samples)

---

## ✅ **Recommendations**

### **Option 1: Make Both Use Threshold (Recommended)**

**For Base Model**:
- Calculate `base_attack_probs` from probabilities
- Apply same threshold optimization as TTT
- Use threshold-based predictions for all metrics

**For TTT Model**:
- Already uses threshold ✅

**Result**: Both models use same prediction method → **Fair comparison**

### **Option 2: Make Both Use Argmax**

**For Base Model**:
- Already uses argmax ✅

**For TTT Model**:
- Change to: `ttt_predictions = torch.argmax(adapted_logits, dim=1)`
- Convert to binary: `ttt_predictions_binary = (ttt_predictions != 0).long()`

**Result**: Both models use same prediction method → **Fair comparison**

### **Option 3: Report Both Methods**

- Calculate metrics using **both** threshold and argmax
- Report both sets of metrics
- Explain the difference

---

## 📋 **Summary**

| Metric | Base Model | TTT Model | Consistent? |
|--------|-----------|-----------|-------------|
| Zero-Day Accuracy | ARGMAX | Not calculated separately | ❌ |
| Zero-Day Precision | ARGMAX | Not calculated separately | ❌ |
| Zero-Day Recall | ARGMAX | Not calculated separately | ❌ |
| Zero-Day F1 | ARGMAX | Not calculated separately | ❌ |
| Zero-Day ZDR | ARGMAX | **THRESHOLD** | ❌ **INCONSISTENT** |
| Overall Accuracy | ARGMAX | **THRESHOLD** | ❌ **INCONSISTENT** |
| Overall Precision | ARGMAX | **THRESHOLD** | ❌ **INCONSISTENT** |
| Overall Recall | ARGMAX | **THRESHOLD** | ❌ **INCONSISTENT** |
| Overall F1 | ARGMAX | **THRESHOLD** | ❌ **INCONSISTENT** |
| Known Attack Metrics | ARGMAX | **THRESHOLD** | ❌ **INCONSISTENT** |

**Conclusion**: All metrics are **INCONSISTENT** between base and TTT models. This is a **critical issue** that makes comparisons unfair!



