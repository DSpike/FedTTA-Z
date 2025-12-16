# Equal Support Set Composition Feature

## Overview

This feature ensures that for `n_way=2` (binary classification), the support set (k-shot) has **equal composition** of Normal and Attack samples, excluding zero-day samples.

## What It Does

### **Before (Random Selection):**
- Randomly selects 2 classes from available labels
- Could select: `[0, 1]` (Normal + Attack) ✅ Good
- Could select: `[1, 2]` (Attack + Attack) ❌ Bad - no Normal samples
- Could select: `[0, 0]` (Normal + Normal) ❌ Bad - no Attack samples
- Support set composition: **Unpredictable**

### **After (Equal Composition Enforcement):**
- Always selects: Normal (0) + One Attack class (excluding zero-day)
- Guaranteed: `[0, attack_class]` where `attack_class != zero_day_label`
- Support set composition: **Always balanced**
- Support set: `k_shot` Normal samples + `k_shot` Attack samples = **Equal composition**

---

## Benefits

1. **Balanced Training:** Ensures model sees equal numbers of Normal and Attack examples in each meta-task
2. **Consistent Learning:** Prevents bias towards one class due to unbalanced support sets
3. **Better Generalization:** Balanced support sets lead to more robust prototypes
4. **Predictable Behavior:** Same composition across all meta-tasks

---

## Implementation

### **Code Changes:**

1. **Function Signature:**
   ```python
   def create_meta_tasks(..., enforce_equal_support_composition: bool = True):
   ```

2. **Logic:**
   ```python
   if n_way == 2 and enforce_equal_support_composition:
       # Always select Normal (0)
       normal_label = torch.tensor([0])
       
       # Select one random Attack class (excluding zero-day)
       attack_labels = available_labels[(available_labels != 0) & 
                                       (available_labels != zero_day_attack_label)]
       selected_attack_label = random.choice(attack_labels)
       
       selected_labels = [normal_label, selected_attack_label]
   ```

3. **Verification:**
   - Logs support set composition for debugging
   - Warns if composition is unequal (shouldn't happen when enabled)

---

## Configuration

### **In `config.py`:**
```python
enforce_equal_support_composition: bool = True  # Default: Enabled
```

### **In Optimization:**
This parameter is now included in the optimization search space:
- **Parameter:** `enforce_equal_support_composition`
- **Type:** Boolean (True/False)
- **Search Space:** `[True, False]`
- **Purpose:** Optimize whether equal composition improves performance

---

## Usage

### **Automatic (Default):**
Enabled by default in `config.py`. No action needed.

### **Disable:**
```python
config.enforce_equal_support_composition = False
```

### **During Optimization:**
Optuna will automatically test both `True` and `False` values to find the optimal setting.

---

## Example

### **Without Equal Composition (enforce=False):**
```
Task 1: Support set = [Normal, Fuzzers] ✅ Balanced
Task 2: Support set = [Fuzzers, Generic] ❌ No Normal
Task 3: Support set = [Normal, Generic] ✅ Balanced
Task 4: Support set = [Normal, Fuzzers] ✅ Balanced
```

**Result:** Inconsistent composition across tasks

### **With Equal Composition (enforce=True):**
```
Task 1: Support set = [Normal, Fuzzers] ✅ Always Normal + Attack
Task 2: Support set = [Normal, Generic] ✅ Always Normal + Attack
Task 3: Support set = [Normal, Exploits] ✅ Always Normal + Attack
Task 4: Support set = [Normal, Fuzzers] ✅ Always Normal + Attack
```

**Result:** Consistent, balanced composition

---

## Impact on Training

### **Expected Benefits:**
- More stable training (consistent support set composition)
- Better class balance in meta-learning
- Improved zero-day detection (balanced exposure to Normal/Attack)

### **Optimization:**
- Optuna will test both `True` and `False` values
- Determines if equal composition improves or harms performance
- Best value will be included in optimized configuration

---

## Logging

When enabled, you'll see:
```
✅ Equal support set composition: 169 Normal, 169 Attack
```

If disabled and unbalanced composition occurs:
```
⚠️  Unequal support set composition: 169 Normal vs 0 Attack (expected equal)
```

---

## Notes

- **Only affects `n_way=2`:** For other n_way values, uses original random selection
- **Excludes zero-day:** Zero-day attack is always excluded from training
- **Fallback:** If no attack labels available (excluding zero-day), falls back to random selection
- **Backward compatible:** Default is `True`, but can be disabled for comparison










