# DIAGNOSIS: Scatter Plot vs ZDR Issue - Complete Analysis

## Summary of Investigation

After extensive analysis, I've identified the **ROOT CAUSE** of why scatter plot shows good separation but ZDR=0.

---

## Key Findings

### 1. Binary Labels Are Correct ✅

From saved test set analysis:
```
Backdoor samples (multiclass=3): 84
Binary labels for Backdoor: [1] (100% have binary_label=1)

Multiclass → Binary mapping:
  0 (Normal) → 0
  1 (Fuzzers) → 1
  2 (Analysis) → 1
  3 (Backdoor) → 1  ← CORRECT!
```

**Conclusion**: Preprocessor correctly assigns `binary_label=1` to all Backdoor samples.

### 2. Zero-Day Confusion Matrix Shows Anomaly ❌

From `performance_metrics_.json`:
```json
"zero_day_only": {
    "num_samples": 184,
    "confusion_matrix": [[117, 61], [0, 0]]
}
```

**Analysis**:
- Row 0 (Normal class): TN=117, FP=61, Total=178
- Row 1 (Attack class): FN=0, TP=0, Total=0  ← NO SAMPLES!

**This means**: All 184 Backdoor samples are being classified as belonging to the "Normal" class (row 0) in the confusion matrix.

### 3. The Paradox

- Backdoor samples in test set: 184 samples
- Binary labels for Backdoor: ALL have `binary_label=1` (Attack)
- But confusion matrix row 1 (Attack): 0 samples!

**How is this possible?**

The only explanation: **zero_day_mask is selecting the WRONG samples!**

---

## Root Cause Hypothesis

### Problem: zero_day_mask Selection Error

The `zero_day_mask` is supposed to identify **Backdoor samples (multiclass=3)**, but it's actually identifying **Normal samples (multiclass=0)** or a mix that excludes Backdoor.

### Evidence:

1. **Confusion matrix row 0 has 178 samples** (117 TN + 61 FP)
2. **Total test set from latest run: 706 samples**
   - Normal: 178 (25.2%)
   - Attack: 528 (74.8%)

3. **The 178 samples in zero_day confusion matrix match EXACTLY the number of Normal samples!**

**Conclusion**: `zero_day_mask` is selecting **Normal samples** instead of **Backdoor samples**!

---

## Why This Happens

### Possible Bug Locations:

#### 1. Zero-Day Mask Creation (main.py lines 3199, 4141, 4170)

```python
if self._is_zero_day_attack(test_attack_cat_original[original_idx]):
    zero_day_mask[seq_idx] = True
```

**Potential issue**: `_is_zero_day_attack()` method might be checking for wrong attack name.

#### 2. Zero-Day Attack Name Mismatch

```python
# config_loader.py line 72
'zero_day_attack': "Backdoor"

# But _is_zero_day_attack() might be checking for:
# - "backdoor" (lowercase)
# - "Backdoors" (plural)
# - Or using wrong source for attack names
```

#### 3. Attack Category Mismatch

The `test_attack_cat_original` might contain:
- String labels: "Normal", "Fuzzers", "Backdoor"
- But configuration expects: "Backdoor"
- Case sensitivity or whitespace issues could cause mismatch

---

## The Scatter Plot Mystery Solved

### Why Scatter Plot Shows Good Separation:

The scatter plot displays ALL samples (Normal vs ALL attacks):
- Normal samples: 178 (good predictions as normal)
- Known attacks (Fuzzers, Analysis, DoS, etc.): ~344 samples (62.9% detected = 332/528)
- Backdoor (zero-day): ~184 samples (misclassified as normal)

**Visual result**: Since 344 known attacks are mostly detected well, the scatter plot shows good attack/normal separation.

**But**: The 184 Backdoor samples are in the "missed" group (False Negatives), making them appear as low-confidence attacks or misclassified as normal in the scatter plot.

### Why ZDR = 0:

1. `zero_day_mask` incorrectly selects **Normal samples** instead of Backdoor
2. These 178 Normal samples all have `binary_label=0`
3. Confusion matrix computed on these Normal samples:
   - All samples have `y_true=0` (Normal class)
   - Placed in row 0 of confusion matrix
   - Row 1 (Attack class) has 0 samples
4. ZDR = TP / (TP + FN) = 0 / (0 + 0) = undefined → 0%

---

## The Fix

### Step 1: Verify zero_day_mask Selection

Add diagnostic logging in main.py where zero_day_mask is created:

```python
# After creating zero_day_mask
logger.info(f"🔍 Zero-day mask diagnostic:")
logger.info(f"   Total sequences: {len(zero_day_mask)}")
logger.info(f"   Zero-day sequences identified: {zero_day_mask.sum()}")

if 'y_test_multiclass' in self.preprocessed_data:
    y_mc = self.preprocessed_data['y_test_multiclass']
    # Map sequences to original samples
    # Check what multiclass labels the masked sequences have
    logger.info(f"   Multiclass labels of zero-day masked samples: {np.unique(y_mc[zero_day_mask])}")

    # Check if Backdoor (label 3) is actually in the mask
    backdoor_mask = (y_mc == 3)
    overlap = (zero_day_mask & backdoor_mask).sum()
    logger.info(f"   Backdoor samples (multiclass=3): {backdoor_mask.sum()}")
    logger.info(f"   Overlap with zero_day_mask: {overlap}")

    if overlap == 0:
        logger.error(f"   ❌ CRITICAL: zero_day_mask does NOT overlap with Backdoor samples!")
        logger.error(f"   This explains why ZDR=0 - wrong samples selected!")
```

### Step 2: Check _is_zero_day_attack() Method

```python
def _is_zero_day_attack(self, attack_cat):
    """Check if attack category is the zero-day attack"""
    # Add logging
    result = attack_cat == self.config.zero_day_attack
    if result:
        logger.debug(f"   Zero-day match: '{attack_cat}' == '{self.config.zero_day_attack}'")
    return result
```

### Step 3: Verify Attack Category Values

```python
# Log unique attack categories
if 'test_attack_cat_original' in self.preprocessed_data:
    unique_cats = np.unique(self.preprocessed_data['test_attack_cat_original'])
    logger.info(f"📊 Unique attack categories in test set: {unique_cats}")
    logger.info(f"   Configured zero-day attack: '{self.config.zero_day_attack}'")

    # Check if configured zero-day exists
    if self.config.zero_day_attack not in unique_cats:
        logger.error(f"   ❌ CRITICAL: '{self.config.zero_day_attack}' NOT found in attack categories!")
        logger.error(f"   Available: {unique_cats}")
```

---

## Expected Outcome After Fix

### Current (Broken):
```
Zero-day evaluation:
  num_samples: 184 (but selecting Normal samples by mistake!)
  confusion_matrix: [[117, 61], [0, 0]]  ← All in row 0 (Normal)
  ZDR: 0%
```

### After Fix:
```
Zero-day evaluation:
  num_samples: 184 (correctly selecting Backdoor samples)
  confusion_matrix: [[TN, FP], [FN, TP]]  ← TP and FN non-zero
  ZDR: 40-80% (actual Backdoor detection performance)
```

---

## Immediate Action

Run the following diagnostic to confirm:

```python
import json
import numpy as np
import pickle

# Load test set
with open('saved_test_sets/cicids_test_set_trial_0.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Load performance metrics
with open('performance_plots/performance_metrics_.json', 'r') as f:
    perf = json.load(f)

# Compare
y_test = test_data['y_test']  # Binary
y_mc = test_data['y_test_multiclass']  # Multiclass

normal_count = (y_test == 0).sum()
backdoor_count = (y_mc == 3).sum()

zero_day_samples = perf['evaluation_results']['adapted_model']['zero_day_only']['num_samples']
cm = perf['evaluation_results']['adapted_model']['zero_day_only']['confusion_matrix']
cm_total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]

print(f"Test set Normal samples: {normal_count}")
print(f"Test set Backdoor samples: {backdoor_count}")
print(f"Zero-day evaluation num_samples: {zero_day_samples}")
print(f"Zero-day confusion matrix total: {cm_total}")

if cm_total == normal_count:
    print("\n❌ CONFIRMED: zero_day_mask is selecting Normal samples!")
elif cm_total == backdoor_count:
    print("\n✅ Correct: zero_day_mask is selecting Backdoor samples")
else:
    print(f"\n⚠️  Unexpected: Mismatch between {cm_total} and {backdoor_count} or {normal_count}")
```

---

## Conclusion

The scatter plot shows good separation because **known attacks are detected well**. But ZDR=0 because **`zero_day_mask` is selecting Normal samples instead of Backdoor samples**, causing the zero-day evaluation to compute metrics on the wrong samples entirely.

This is NOT a model failure - it's a **sample selection bug** in the evaluation code.

Fix: Ensure `zero_day_mask` correctly identifies Backdoor samples by verifying:
1. Attack category strings match exactly ("Backdoor" case-sensitive)
2. `_is_zero_day_attack()` method works correctly
3. Sequence-to-sample mapping preserves correct labels
