# How to Check TTT Overfitting

## 📋 **Quick Guide**

I've created a diagnostic tool that automatically checks for TTT overfitting after evaluation. The check is **automatically integrated** into `main.py`.

---

## 🔍 **What Gets Checked**

### **1. Normal Sample Performance Degradation**

- ✅ Base model accuracy on Normal samples
- ✅ TTT model accuracy on Normal samples
- ✅ **Flag if**: Accuracy drops >5%

### **2. False Positive Rate Increase**

- ✅ Base model FP rate (Normal samples classified as attacks)
- ✅ TTT model FP rate
- ✅ **Flag if**: FP rate increases >5%

### **3. Zero-Day Trade-off**

- ✅ Zero-day performance improvement
- ✅ Overall performance change
- ✅ **Flag if**: Zero-day improves but overall decreases

### **4. Performance Discrepancy**

- ✅ Zero-day accuracy vs Overall accuracy
- ✅ **Flag if**: Gap >10% (zero-day much higher than overall)

---

## 📊 **How to Use**

### **Automatic Check (Already Integrated)**

The check runs **automatically** after TTT evaluation in `main.py`. Just run your system normally:

```bash
python main.py
```

**Look for this section in the logs:**

```
================================================================================
TTT OVERFITTING DIAGNOSTIC
================================================================================
✅ Status: HEALTHY
   Severity: NONE

📊 Normal Sample Performance:
   Base Model Accuracy: 85.23%
   TTT Model Accuracy:  82.15%
   Accuracy Drop:       3.08%
   Base FP Rate:        2.45%
   TTT FP Rate:         3.12%
   FP Rate Increase:    0.67%

💡 Recommendations:
   1. TTT adaptation appears healthy. No significant overfitting detected.
================================================================================
```

---

### **Manual Check (Using Diagnostic Script)**

If you want to check overfitting manually:

```python
from check_ttt_overfitting import check_ttt_overfitting, print_overfitting_report

# After evaluation
base_results = system.evaluate_base_model_only()
adapted_results = system.evaluate_adapted_model(adapted_model)

# Get test data
X_test = system.preprocessed_data['X_test']
y_test = system.preprocessed_data['y_test']
zero_day_mask = ...  # Your zero-day mask

# Check overfitting
results = check_ttt_overfitting(
    base_results=base_results,
    ttt_results=adapted_results,
    X_test=X_test,
    y_test=y_test,
    zero_day_mask=zero_day_mask,
    threshold=0.05  # 5% drop threshold
)

# Print report
print_overfitting_report(results)
```

---

## ⚠️ **Interpreting Results**

### **Status: HEALTHY** ✅

- No significant performance degradation
- Normal samples still perform well
- False positive rate under control
- **Action**: Continue with current TTT settings

### **Status: OVERFITTING** (Severity: MEDIUM) ⚠️

**Flags Detected:**

- Normal accuracy dropped 5-10%
- FP rate increased 5-10%
- Zero-day trade-off detected

**Recommended Actions:**

1. **Reduce TTT adaptation steps**: Lower `ttt_base_steps` in `config.py` (e.g., from 258 to 150)
2. **Reduce entropy weight**: Lower `entropy_weight` in `config.py` (e.g., from 0.67 to 0.4)
3. **Lower learning rate**: Reduce `ttt_lr` slightly (e.g., from 0.00015 to 0.0001)

### **Status: OVERFITTING** (Severity: HIGH) ⚠️⚠️

**Flags Detected:**

- Normal accuracy dropped >10%
- FP rate increased >10%
- Multiple flags triggered

**Recommended Actions:**

1. **Significantly reduce TTT intensity**: `ttt_base_steps` → 100, `entropy_weight` → 0.3
2. **Consider balanced adaptation**: Separate adaptation for Normal vs Attack samples
3. **Review TTT configuration**: Check if TTT is necessary for your use case

---

## 📈 **Example Output**

### **Healthy TTT Adaptation:**

```
✅ Status: HEALTHY
   Severity: NONE

📊 Normal Sample Performance:
   Base Model Accuracy: 88.50%
   TTT Model Accuracy:  86.20%
   Accuracy Drop:       2.30% ✅ (acceptable)
   Base FP Rate:        1.80%
   TTT FP Rate:         2.10%
   FP Rate Increase:    0.30% ✅ (acceptable)

🎯 Zero-Day Performance:
   Base Model Accuracy:     62.30%
   TTT Model Accuracy:      78.50%
   Accuracy Improvement:    16.20% ✅

📊 Overall Performance:
   Base Model Accuracy:  75.40%
   TTT Model Accuracy:   78.20%
   Accuracy Change:      2.80% ✅

💡 Recommendations:
   1. TTT adaptation appears healthy. No significant overfitting detected.
```

### **Overfitting Detected:**

```
⚠️ Status: OVERFITTING
   Severity: MEDIUM

⚠️ Overfitting Flags Detected:
   - Normal Accuracy Degradation
   - False Positive Increase
   - Zero Day Tradeoff

📊 Normal Sample Performance:
   Base Model Accuracy: 90.20%
   TTT Model Accuracy:  72.50%
   Accuracy Drop:       17.70% ⚠️ (significant drop)
   Base FP Rate:        2.10%
   TTT FP Rate:         12.50%
   FP Rate Increase:    10.40% ⚠️ (high false alarms)

🎯 Zero-Day Performance:
   Base Model Accuracy:     58.30%
   TTT Model Accuracy:      82.10%
   Accuracy Improvement:    23.80% ✅

📊 Overall Performance:
   Base Model Accuracy:  76.80%
   TTT Model Accuracy:   74.20%
   Accuracy Change:      -2.60% ⚠️ (overall decreased)

💡 Recommendations:
   1. Normal sample accuracy dropped by 17.7%. Consider reducing TTT adaptation intensity.
   2. False positive rate increased by 10.4%. TTT may be overfitting to attack patterns. Reduce entropy_weight.
   3. Zero-day performance improved but overall performance decreased. TTT is specializing too much.
```

---

## 🔧 **Fixing Overfitting**

### **Step 1: Reduce TTT Intensity**

Edit `config.py`:

```python
# Before (aggressive)
ttt_base_steps = 258
ttt_lr = 0.0001518747922672249
entropy_weight = 0.6705241236872915

# After (balanced)
ttt_base_steps = 150  # Reduced from 258
ttt_lr = 0.0001  # Reduced from 0.00015
entropy_weight = 0.4  # Reduced from 0.67
```

### **Step 2: Re-run Evaluation**

```bash
python main.py
```

### **Step 3: Check Overfitting Report Again**

Look for improvement in:

- Normal sample accuracy (should be higher)
- False positive rate (should be lower)
- Overall performance (should improve)

---

## 🎯 **Key Metrics to Monitor**

| Metric                      | Healthy Range | Warning   | Critical |
| --------------------------- | ------------- | --------- | -------- |
| **Normal Accuracy Drop**    | < 5%          | 5-10%     | > 10%    |
| **FP Rate Increase**        | < 3%          | 3-8%      | > 8%     |
| **Overall Accuracy Change** | > 0%          | -2% to 0% | < -2%    |
| **Zero-Day Improvement**    | > 5%          | 0-5%      | < 0%     |

---

## 📝 **Summary**

1. ✅ **Automatic check** runs after TTT evaluation
2. ✅ **Check logs** for "TTT OVERFITTING DIAGNOSTIC" section
3. ✅ **Review flags** and recommendations
4. ✅ **Adjust TTT config** if overfitting detected
5. ✅ **Re-run** to verify improvement

**The diagnostic tool is already integrated - just run your system and check the logs!** 🎯



