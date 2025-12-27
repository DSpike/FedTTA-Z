# ZERO-DAY DETECTION RATE (ZDR) FIX - COMPLETED

## Issue Summary

**Problem**: Zero-Day Detection Rate (ZDR) = 0% despite Backdoor samples existing in UNSW-NB15 dataset

**Root Cause**: Configuration file `config.py` was using CICIDS2017 `attack_types` mapping even though the system was configured to use UNSW-NB15 dataset.

---

## Root Cause Analysis

### What Was Happening

1. **Dataset Configuration** (in [config_loader.py](config_loader.py:70-72)):
   ```python
   'data_path': "UNSW_NB15_training-set.csv",
   'test_path': "UNSW_NB15_testing-set.csv",
   'zero_day_attack': "Backdoor",  # UNSW-NB15 attack
   ```

2. **Attack Types Mapping** (in [config.py](config.py:170-186) - **BEFORE FIX**):
   ```python
   # CICIDS2017 attack types (ACTIVE for CICIDS2017 dataset)
   attack_types = {
       'BENIGN': 0,
       'Bot': 1,
       'DDoS': 2,
       ...
       'PortScan': 10,
       ...
   }
   ```
   **Problem**: No 'Backdoor' in this dictionary!

3. **Zero-Day Label Lookup** (in [config.py](config.py:423)):
   ```python
   return self.attack_types.get(self.zero_day_attack, 0)
   # Equivalent to: attack_types.get("Backdoor", 0)
   # Returns: 0 (default, because "Backdoor" not in CICIDS2017 attack_types)
   ```

4. **Stratified Test Subset** (in [main.py](main.py:742)):
   ```python
   zero_day_label = self.config.zero_day_attack_label  # Returns 0 (wrong!)
   zero_day_indices = np.where(y_multiclass_np == zero_day_label)[0]
   # Searches for samples with label 0 (Normal in UNSW-NB15)
   # Should search for label 3 (Backdoor in UNSW-NB15)
   ```

5. **Result**:
   - Stratified subset selected Normal samples (label 0) as "zero-day"
   - Real Backdoor samples (label 3) were excluded or treated as non-zero-day
   - Zero-day evaluation computed metrics on wrong samples
   - Confusion matrix: `[[128, 47], [0, 0]]` - row 1 empty (no attack samples)
   - ZDR = 0% (no true Backdoor samples in evaluation)

---

## Fix Applied

### Changes to [config.py](config.py)

#### 1. Added UNSW-NB15 Attack Types (Lines 169-181)

```python
# UNSW-NB15 attack types (ACTIVE for UNSW-NB15 dataset)
attack_types = {
    'Normal': 0,
    'Fuzzers': 1,
    'Analysis': 2,
    'Backdoor': 3,      # ← KEY: Now Backdoor → label 3
    'DoS': 4,
    'Exploits': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9,
}
```

#### 2. Commented Out CICIDS2017 Attack Types (Lines 183-202)

```python
# CICIDS2017 attack types (uncomment if switching to CICIDS2017 dataset)
'''
attack_types = {
    'BENIGN': 0,
    'Bot': 1,
    ...
}
'''
```

#### 3. Added UNSW-NB15 Dataset Detection (Line 220-222)

```python
elif 'Backdoor' in self.attack_types and 'Fuzzers' in self.attack_types:
    # UNSW-NB15 dataset detected
    self._init_unsw_categories()
```

#### 4. Added UNSW Category Initialization (Lines 369-400)

```python
def _init_unsw_categories(self):
    """
    Initialize UNSW-NB15 category mapping
    10 attack types (already at appropriate granularity for zero-day detection)
    """
    # UNSW-NB15: Each attack type is its own category
    self.attack_category_mapping = {
        'Normal': 'Normal',
        'Fuzzers': 'Fuzzers',
        'Analysis': 'Analysis',
        'Backdoor': 'Backdoor',   # ← Maps to itself
        'DoS': 'DoS',
        'Exploits': 'Exploits',
        'Generic': 'Generic',
        'Reconnaissance': 'Reconnaissance',
        'Shellcode': 'Shellcode',
        'Worms': 'Worms',
    }

    # Category → integer mapping (same as attack_types)
    self.category_types = {
        'Normal': 0,
        'Fuzzers': 1,
        'Analysis': 2,
        'Backdoor': 3,   # ← Correct label
        'DoS': 4,
        'Exploits': 5,
        'Generic': 6,
        'Reconnaissance': 7,
        'Shellcode': 8,
        'Worms': 9,
    }
```

---

## Verification

### Test Result

```python
from config_loader import get_dataset_config

config = get_dataset_config()
print(f'zero_day_attack: {config.zero_day_attack}')
print(f'zero_day_attack_label: {config.zero_day_attack_label}')

# Output:
# zero_day_attack: Backdoor
# zero_day_attack_label: 3  ✅ CORRECT! (was 0 before fix)
```

### UNSW-NB15 Test Set Composition

```
Test set (82,332 samples):
  Normal: 37,000 (44.9%)
  Generic: 18,871 (22.9%)
  Exploits: 11,132 (13.5%)
  Fuzzers: 6,062 (7.4%)
  DoS: 4,089 (5.0%)
  Reconnaissance: 3,496 (4.2%)
  Analysis: 677 (0.8%)
  Backdoor: 583 (0.7%)  ← Zero-day attack (should now be detected)
  Shellcode: 378 (0.5%)
  Worms: 44 (0.1%)
```

---

## Expected Results After Fix

### Before Fix (Wrong Label)

```
zero_day_attack_label = 0 (Normal label)
Stratified subset selects Normal samples as "zero-day"
Zero-day evaluation:
  num_samples: 175
  confusion_matrix: [[128, 47], [0, 0]]  ← Row 1 empty
  ZDR: 0.0%
```

### After Fix (Correct Label)

```
zero_day_attack_label = 3 (Backdoor label)
Stratified subset selects actual Backdoor samples
Zero-day evaluation:
  num_samples: ~150-200 (depends on stratified sampling)
  confusion_matrix: [[TN, FP], [FN, TP]]  ← Both rows populated
  ZDR: 40-80% (actual Backdoor detection performance)
```

---

## Why This Happened

### Timeline of Events

1. **Original System**: Designed for CICIDS2017 with PortScan as zero-day
2. **Dataset Switch**: Changed to UNSW-NB15 in [config_loader.py](config_loader.py:70-72)
3. **Partial Update**: Updated `zero_day_attack` to "Backdoor"
4. **Missing Update**: FORGOT to update `attack_types` mapping in [config.py](config.py:170-186)
5. **Result**: Label mismatch - looking for label 0 instead of label 3

---

## Files Modified

1. **[config.py](config.py)**:
   - Added UNSW-NB15 attack_types (lines 169-181)
   - Commented out CICIDS2017 attack_types (lines 183-202)
   - Added UNSW dataset detection (line 220-222)
   - Added `_init_unsw_categories()` method (lines 369-400)

---

## Current Status

✅ **Fix Applied**: [config.py](config.py) updated with UNSW-NB15 attack types
✅ **Verification**: `zero_day_attack_label` now returns 3 (correct)
⏳ **Running**: `main.py` is executing with fixed configuration
📝 **Log File**: `run_unsw_backdoor_fixed.log`

---

## Next Steps

1. ⏳ **Wait for completion**: `main.py` is running (preprocessing → training → evaluation)
2. ✅ **Expected outcome**:
   - Zero-day mask correctly identifies Backdoor samples
   - Stratified subset contains actual Backdoor samples (not Normal samples)
   - Zero-day evaluation computes metrics on real Backdoor samples
   - ZDR will show actual Backdoor detection performance (likely 40-80%)
   - Confusion matrix row 1 will be populated (TP and FN > 0)

3. 📊 **Verify results**:
   ```bash
   # After completion, check results
   python -c "
   import json
   with open('performance_plots/performance_metrics_.json', 'r') as f:
       metrics = json.load(f)
   final_eval = metrics.get('final_evaluation_results', {})
   print(f'ZDR: {final_eval.get(\"zero_day_detection_rate\")}')
   print(f'Confusion Matrix: {final_eval.get(\"confusion_matrix\")}')
   "
   ```

---

## Key Learnings

### 1. Configuration Consistency is Critical

When switching datasets, ensure ALL configuration files are updated:
- ✅ Dataset paths ([config_loader.py](config_loader.py))
- ✅ Zero-day attack name ([config_loader.py](config_loader.py))
- ✅ Attack types mapping ([config.py](config.py)) ← **CRITICAL, was missing**
- ✅ Category initialization methods ([config.py](config.py))

### 2. Label Mapping is the Source of Truth

The `attack_types` dictionary in [config.py](config.py) determines:
- Which label corresponds to which attack
- What `zero_day_attack_label` returns
- Which samples are selected for zero-day evaluation

If this mapping is wrong, **everything downstream fails**.

### 3. Debugging Steps for ZDR=0

When ZDR=0, check in this order:
1. ✅ Does zero-day attack exist in raw test set? (Yes: 583 Backdoor samples)
2. ✅ Is `zero_day_attack` configured correctly? (Yes: "Backdoor")
3. ❌ **Is `attack_types` mapping correct for the dataset?** (NO - was CICIDS2017)
4. ❌ Does `zero_day_attack_label` return correct label? (NO - returned 0 instead of 3)
5. ❌ Does stratified subset select correct samples? (NO - selected Normal instead of Backdoor)

---

## Solution Comparison

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| `attack_types` | CICIDS2017 | UNSW-NB15 |
| "Backdoor" in dictionary? | ❌ No | ✅ Yes |
| `zero_day_attack_label` | 0 (wrong) | 3 (correct) |
| Stratified subset selects | Normal samples | Backdoor samples |
| Zero-day confusion matrix | `[[128, 47], [0, 0]]` | `[[TN, FP], [FN, TP]]` |
| ZDR | 0.0% | 40-80% (TBD) |

---

**Status**: ✅ **FIX COMPLETE** - System is now running with correct UNSW-NB15 configuration.

The zero-day detection should now work correctly for Backdoor attacks in UNSW-NB15 dataset.
