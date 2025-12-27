# TTT Model Zero Performance on Zero-Day Samples - Root Cause & Fix

## 🔍 **Problem Summary**

TTT model was showing **zero performance** (0.0) for zero-day samples in evaluation plots, even though zero-day samples exist in the test set.

---

## 🎯 **Root Cause**

### **Primary Issue: Wrong Zero-Day Label Used**

The TTT evaluation code (`evaluate_adapted_model`) was using the **wrong label** to identify zero-day samples:

**Before Fix:**
```python
# Line 3998 (WRONG):
zero_day_attack_label = attack_types.get(zero_day_attack, 1)  # Returns 10 (fine-grained label)
```

**Problem:**
- In **grouped mode** (`use_category_grouping=True`), PortScan maps to category "PortScan" with **label 4**
- But `attack_types.get("PortScan")` returns **label 10** (fine-grained label)
- Test set has labels: `[0, 1, 2, 4, 5, 6]` (label 4 is present, label 10 is NOT)
- Code was looking for label 10 → **zero zero-day samples found** → **zero performance**

---

## 📊 **Impact on TTT Evaluation**

### **What Happened:**

1. **Base Model Evaluation** (also had same bug):
   - Used label 10 instead of 4
   - Found 0 zero-day samples
   - Zero-day metrics = 0.0

2. **TTT Model Evaluation** (same bug):
   - Used label 10 instead of 4
   - Found 0 zero-day samples
   - Zero-day metrics = 0.0

3. **Result:**
   - Both models showed zero performance on zero-day samples
   - Zero-day detection rate = 0.0
   - Zero-day accuracy = 0.0
   - All zero-day metrics = 0.0

---

## ✅ **Fixes Applied**

### **Fix 1: Base Model Evaluation** (`evaluate_base_model_only`)
**Location:** `main.py` line 3071-3085

**Before:**
```python
zero_day_attack_label = attack_types.get(zero_day_attack, 1)  # Wrong: returns 10
```

**After:**
```python
# CRITICAL: Get the numeric label for zero-day attack
# In grouped mode, use config's zero_day_attack_label which handles category mapping correctly
if hasattr(self.config, 'zero_day_attack_label'):
    zero_day_attack_label = self.config.zero_day_attack_label  # Correct: returns 4 in grouped mode
else:
    zero_day_attack_label = attack_types.get(zero_day_attack, 1)  # Fallback
```

### **Fix 2: TTT Model Evaluation** (`evaluate_adapted_model`)
**Location:** `main.py` line 3997-4008

**Before:**
```python
zero_day_attack_label = attack_types.get(zero_day_attack, 1)  # Wrong: returns 10
```

**After:**
```python
# CRITICAL: Get the numeric label for zero-day attack
# In grouped mode, use config's zero_day_attack_label which handles category mapping correctly
if hasattr(self.config, 'zero_day_attack_label'):
    zero_day_attack_label = self.config.zero_day_attack_label  # Correct: returns 4 in grouped mode
else:
    zero_day_attack_label = attack_types.get(zero_day_attack, 1)  # Fallback
```

### **Fix 3: Zero-Day Detection Evaluation** (`evaluate_zero_day_detection`)
**Location:** `main.py` line 5211-5242

**Before:**
```python
# For TCN sequences, incorrectly marked ALL attacks as zero-day
zero_day_mask = (y_test_tensor != 0).to(torch.bool)  # Wrong: marks all attacks
```

**After:**
```python
# Uses correct zero_day_attack_label (4) to identify PortScan samples
if 'y_test_multiclass' in self.preprocessed_data:
    zero_day_mask = (y_test_multiclass_seq == zero_day_attack_label)  # Correct: uses label 4
```

---

## 🔧 **How the Fix Works**

### **Config Property Logic:**

In `config.py`, the `zero_day_attack_label` property (lines 404-423):

```python
@property
def zero_day_attack_label(self) -> int:
    if self.use_category_grouping and self.category_types:
        if self.zero_day_attack in self.category_types:
            return self.category_types.get(self.zero_day_attack, 0)  # Returns 4 for PortScan
        else:
            category = self.attack_category_mapping.get(self.zero_day_attack)
            if category:
                return self.category_types.get(category, 0)
    
    # Fine-grained mode: Return specific attack label
    return self.attack_types.get(self.zero_day_attack, 0)  # Returns 10 for PortScan
```

**For CICIDS2017 in grouped mode:**
- `zero_day_attack = "PortScan"`
- `use_category_grouping = True`
- `category_types = {'PortScan': 4, ...}`
- **Result:** Returns **4** (correct for grouped mode)

---

## 📈 **Expected Results After Fix**

### **Before Fix:**
- Zero-day samples found: **0**
- Zero-day detection rate: **0.0**
- Zero-day accuracy: **0.0**
- All zero-day metrics: **0.0**

### **After Fix:**
- Zero-day samples found: **56** (from 224 test sequences, 25% target)
- Zero-day detection rate: **> 0.0** (actual performance)
- Zero-day accuracy: **> 0.0** (actual performance)
- Zero-day metrics: **Real values** based on actual performance

---

## 🎯 **Why This Matters**

1. **Correct Evaluation**: TTT model can now be properly evaluated on zero-day samples
2. **Fair Comparison**: Base vs TTT comparison will show actual differences
3. **Performance Metrics**: Zero-day detection rate, accuracy, precision, recall will be meaningful
4. **Research Validity**: Results will reflect actual zero-day detection capability

---

## ✅ **Verification**

After running the code, check the diagnostic output:

```
🔍 DIAGNOSTIC: Zero-day identification data availability:
   Zero-day attack name: 'PortScan'
   Zero-day attack label (from config): 4  ← Should be 4, not 10
   Config use_category_grouping: True
   Zero-day label 4 in y_test_multiclass: True  ← Should be True
```

If you see:
- ✅ `Zero-day attack label: 4` (not 10)
- ✅ `Zero-day label 4 in y_test_multiclass: True`
- ✅ `Found X PortScan samples in test_attack_cat_original`

Then the fix is working correctly!

---

## 📋 **Summary**

**Root Cause:** TTT evaluation used wrong label (10 instead of 4) to identify zero-day samples in grouped mode.

**Fix:** Use `self.config.zero_day_attack_label` property which correctly handles grouped mode.

**Result:** TTT model will now correctly identify and evaluate zero-day samples, showing real performance metrics instead of zeros.



