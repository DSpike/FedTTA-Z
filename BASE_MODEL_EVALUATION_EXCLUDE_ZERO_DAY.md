# Base Model Performance Evaluation - Excluding Zero-Day Samples

## 🎯 **Implementation Summary**

The base model performance evaluation plot now **excludes zero-day samples** by default, evaluating only on **Normal + Known Attacks** (what the base model was trained on). This provides a fairer evaluation that matches the training distribution.

---

## ✅ **What Changed**

### **1. Modified `evaluate_base_model_only()` Function**

Added `exclude_zero_day` parameter:
- **Default:** `False` (backward compatible - includes all test samples)
- **When `True`:** Filters out zero-day samples before evaluation

**Location:** `main.py` line 2831

```python
def evaluate_base_model_only(self, exclude_zero_day: bool = False) -> Dict[str, Any]:
    """
    Evaluate ONLY the base model (transductive meta-learning) without TTT adaptation
    
    Args:
        exclude_zero_day: If True, evaluate only on Normal + Known Attacks (excludes zero-day samples).
                          If False, evaluate on all test samples including zero-day.
                          Default: False (evaluate on all samples for backward compatibility)
    """
```

---

### **2. Filtering Logic**

When `exclude_zero_day=True`:
1. **Identifies zero-day samples** using the zero-day mask
2. **Filters test set** to exclude zero-day samples
3. **Evaluates base model** on filtered test set (Normal + Known Attacks only)
4. **Logs evaluation mode** for transparency

**Key Code:**
```python
if exclude_zero_day:
    non_zero_day_mask = ~zero_day_mask
    X_test_filtered = X_test_tensor[non_zero_day_mask]
    y_test_filtered = y_test_tensor[non_zero_day_mask]
    zero_day_mask_filtered = torch.zeros(len(X_test_filtered), dtype=torch.bool, device=self.device)
    logger.info(f"🔍 Base model evaluation mode: EXCLUDING zero-day samples")
```

---

### **3. Updated Base Model Performance Plot Generation**

The base model performance bar chart now uses the filtered evaluation:

**Location:** `main.py` line 2683

**Before:**
```python
base_results = evaluation_results['base_model']
plot_paths['base_model_performance_barchart'] = self.visualizer.plot_base_model_performance_barchart(
    base_results
)
```

**After:**
```python
# Re-evaluate base model EXCLUDING zero-day samples for fair evaluation
logger.info("🔍 Re-evaluating base model EXCLUDING zero-day samples for base model performance plot...")
base_results_no_zeroday = self.evaluate_base_model_only(exclude_zero_day=True)

plot_paths['base_model_performance_barchart'] = self.visualizer.plot_base_model_performance_barchart(
    base_results_no_zeroday
)
```

---

## 📊 **Evaluation Modes**

### **Mode 1: Base Model Performance Plot (Excludes Zero-Day)**
- **Purpose:** Show how well base model performs on what it was trained on
- **Test Set:** Normal + Known Attacks only (excludes zero-day)
- **Use Case:** Fair evaluation of base model capability on seen data

### **Mode 2: Zero-Day Performance Plot (Zero-Day Only)**
- **Purpose:** Show zero-day detection capability
- **Test Set:** Zero-day attacks only
- **Use Case:** Evaluate zero-day detection performance

### **Mode 3: Performance Comparison Plot (Includes All)**
- **Purpose:** Compare base vs TTT on full test set
- **Test Set:** Normal + Known Attacks + Zero-Day (all samples)
- **Use Case:** Overall system performance comparison

---

## 🔍 **Benefits**

### **1. Fairer Evaluation**
- Base model is evaluated on **what it was trained on** (Normal + Known Attacks)
- Excludes zero-day samples which the model hasn't seen during training
- Provides realistic baseline performance

### **2. Clear Separation of Concerns**
- **Base Model Plot:** Normal + Known Attacks (training distribution)
- **Zero-Day Plot:** Zero-day attacks only (detection capability)
- **Comparison Plot:** Full test set (overall system performance)

### **3. Better Performance Interpretation**
- Base model performance is no longer "dragged down" by zero-day samples
- Can clearly see:
  - How well base model performs on known patterns
  - How well TTT improves zero-day detection
  - Overall system performance on mixed test set

---

## 📈 **Expected Results**

### **Base Model Performance Plot (Excludes Zero-Day)**
- **Expected Accuracy:** Higher (e.g., 70-85% instead of 60-70%)
- **Rationale:** Model is evaluated on patterns it has seen during training
- **Interpretation:** Shows base model's capability on known attack types

### **Zero-Day Performance Plot (Zero-Day Only)**
- **Expected Accuracy:** Lower initially (e.g., 50-70%)
- **Rationale:** Model hasn't seen these attacks during training
- **Interpretation:** Shows zero-day detection challenge and TTT improvement

### **Performance Comparison Plot (Includes All)**
- **Expected Accuracy:** Moderate (e.g., 65-75%)
- **Rationale:** Balanced mix of known and unknown attacks
- **Interpretation:** Overall system performance in realistic scenario

---

## 🎯 **Evaluation Mode Tracking**

Results now include `evaluation_mode` field:
- `'exclude_zero_day'`: Zero-day samples excluded from evaluation
- `'include_all'`: All test samples included in evaluation

This helps track which evaluation mode was used for each metric.

---

## 🔧 **Technical Details**

### **Metrics Calculated on Filtered Test Set:**
When `exclude_zero_day=True`, all metrics are calculated on the filtered test set:
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC, AUC-PR
- Confusion Matrix
- MCC (Matthews Correlation Coefficient)
- FAR (False Alarm Rate)

### **Zero-Day Metrics:**
- When `exclude_zero_day=True`: Zero-day metrics are set to 0.0 (no zero-day samples)
- When `exclude_zero_day=False`: Zero-day metrics calculated on zero-day samples only

---

## ✅ **Backward Compatibility**

- **Default behavior unchanged:** `exclude_zero_day=False` by default
- **Existing code still works:** All existing calls to `evaluate_base_model_only()` continue to work
- **New functionality:** Base model performance plot uses `exclude_zero_day=True` automatically

---

## 📝 **Usage Example**

```python
# Evaluate base model on Normal + Known Attacks only (excludes zero-day)
base_results_no_zeroday = system.evaluate_base_model_only(exclude_zero_day=True)

# Evaluate base model on all test samples (includes zero-day)
base_results_all = system.evaluate_base_model_only(exclude_zero_day=False)
```

---

## 🎓 **Scientific Rationale**

### **Why Exclude Zero-Day from Base Model Evaluation?**

1. **Training Distribution Match:**
   - Base model was trained on Normal + Known Attacks
   - Evaluation should match training distribution for fair assessment

2. **Separate Concerns:**
   - Base model performance: How well it classifies known patterns
   - Zero-day detection: How well it detects unseen patterns
   - These are different capabilities and should be evaluated separately

3. **Clearer Performance Interpretation:**
   - Base model performance on known patterns shows model quality
   - Zero-day performance shows detection capability
   - Mixing them makes it unclear which capability is being measured

4. **Standard Practice:**
   - In zero-day detection research, base model is typically evaluated on known patterns
   - Zero-day detection is evaluated separately as a distinct capability

---

## 📊 **Expected Impact on Results**

### **Base Model Performance Plot:**
- **Before:** 66.4% accuracy (includes zero-day samples)
- **After:** ~70-85% accuracy (excludes zero-day samples) ✅
- **Interpretation:** Shows true base model capability on known patterns

### **Zero-Day Performance Plot:**
- **No change:** Still evaluates only on zero-day samples
- **Interpretation:** Shows zero-day detection capability

### **Performance Comparison Plot:**
- **No change:** Still evaluates on full test set
- **Interpretation:** Shows overall system performance

---

## ✅ **Summary**

The base model performance evaluation now **excludes zero-day samples** by default for the base model performance plot, providing:

1. ✅ **Fairer evaluation** on training distribution
2. ✅ **Clearer separation** of base model vs zero-day detection
3. ✅ **Better interpretation** of performance metrics
4. ✅ **Backward compatibility** with existing code
5. ✅ **Scientific rigor** matching standard practices

---

*Implementation Date: December 2, 2025*  
*Related Files: `main.py` (lines 2831, 2683, 2940-2964)*









