# TTT Model Fixes - Implementation Guide
**Critical fixes required to make TTT adaptation work**

---

## 🔴 CRITICAL ISSUE IDENTIFIED

**Your TTT model is NOT adapting at all!**

The system crashes during TTT adaptation and falls back to the base model, which is why:
- TTT performance = Base model performance (or worse)
- Zero-day detection rate is 0.0000
- Predictions are identical between base and TTT models

---

## 📋 Root Causes Summary

| # | Issue | Impact | File | Line |
|---|-------|--------|------|------|
| 1 | GradScaler.unscale_() crash in CPU mode | TTT doesn't run | models/transductive_fewshot_model.py | 2680 |
| 2 | No zero-day samples in test set | Can't measure ZDR | Preprocessor/data loading | N/A |
| 3 | Zero-day attack mismatch | Looking for wrong attack | Config | N/A |
| 4 | TTT LR too low (0.0001113) | Minimal adaptation | config_loader.py | 81 |

---

## 🔧 Fix #1: GradScaler Crash (CRITICAL) ⚠️

### Problem
```python
# models/transductive_fewshot_model.py:2680
scaler.unscale_(meta_optimizer)  # ❌ Crashes in CPU mode
```

### Error
```
AttributeError: 'GradScaler' object has no attribute 'unscale_'
```

### Root Cause
`GradScaler.unscale_()` only exists when:
- Mixed precision is enabled (use_fp16=True)
- CUDA is available (GPU mode)

Your system runs in **CPU mode** with **mixed precision disabled**, so this method doesn't exist.

### Solution

**File**: [models/transductive_fewshot_model.py](models/transductive_fewshot_model.py:2680)

```python
# BEFORE (line 2677-2684):
scaled_loss.backward()

# Gradient clipping
scaler.unscale_(meta_optimizer)  # ❌ CRASHES
torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

scaler.step(meta_optimizer)
scaler.update()

# AFTER (FIXED):
scaled_loss.backward()

# Gradient clipping (with CPU mode support)
if scaler.is_enabled():  # ✅ Check if mixed precision is enabled
    scaler.unscale_(meta_optimizer)
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    scaler.step(meta_optimizer)
    scaler.update()
else:  # ✅ CPU mode - no scaling needed
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    meta_optimizer.step()
```

### Impact
- ✅ TTT will actually run instead of crashing
- ✅ Parameters will update (not stay at 0.000000)
- ✅ Predictions will differ from base model

---

## 🔧 Fix #2: Zero-Day Samples Missing (CRITICAL) ⚠️

### Problem
```
Zero-Day Attacks Only (0 samples, 0.0% of test set)
Available labels in test data: [0, 1]
```

Your test set has **NO zero-day attack samples!**

### Root Cause Analysis

**Config says**:
```python
# config_loader.py line 72
'zero_day_attack': "PortScan",
```

**System looks for**:
```
Line 802: Zero-day attack: 'Exploits', label: 5
```

**Test set contains**:
```
Labels: [0, 1] only (Normal and Attack)
```

**Result**: Zero-day attack doesn't exist in test set!

### Investigation Needed

1. **Check what attack types are in CICIDS2017_test.csv**:
```python
import pandas as pd
df = pd.read_csv('CICIDS2017_test.csv')
print(df[' Label'].value_counts())  # Note: may have space in column name
```

2. **Verify attack_types dictionary matches dataset**:
```python
# config.py should have CICIDS2017 attack_types uncommented
# Lines 167-186 (currently commented out)
```

3. **Check preprocessor**:
```python
# preprocessing/blockchain_federated_cicids_preprocessor.py
# Verify it correctly encodes CICIDS2017 attack types
```

### Possible Solutions

**Option A**: Update zero_day_attack to match what's in test set
```python
# If test set has "Bot", "DDoS", etc., update:
'zero_day_attack': "Bot",  # or "DDoS", "SSH-Patator", etc.
```

**Option B**: Regenerate test set with PortScan included

**Option C**: Check if using wrong dataset
- Config says CICIDS2017
- But test set might be from different dataset

### Impact
- ✅ Zero-day samples will exist in test set
- ✅ ZDR will be measurable (not 0.0000)
- ✅ Can evaluate TTT improvement on zero-day detection

---

## 🔧 Fix #3: TTT Learning Rate Too Low 🟡

### Problem
```
Configuration:
  - Configured LR: 0.002
  - Actual LR used: 0.00011136839897653453 (100x smaller!)
```

### Root Cause
The TTT learning rate is being divided or scaled down somewhere in the code.

### Solution

**File**: [config_loader.py](config_loader.py:81)

```python
# BEFORE:
'ttt_lr': 0.002,

# AFTER (increase by 5-10x):
'ttt_lr': 0.01,  # or 0.02
```

**Alternative**: Find where LR is being scaled down and remove that scaling.

### Impact
- ✅ More meaningful parameter updates during TTT
- ✅ Stronger adaptation to test distribution
- ✅ Better zero-day detection improvement

---

## 🔧 Fix #4: Uncomment CICIDS2017 Attack Types 🟠

### Problem
```python
# config.py lines 167-186
# CICIDS2017 attack types are COMMENTED OUT
'''
attack_types = {
    'BENIGN': 0,
    'Bot': 1,
    ...
}
'''
```

But you're using KDD attack_types instead!

### Solution

**File**: [config.py](config.py:68-186)

```python
# BEFORE (lines 68-125): UNCOMMENT KDD attack_types
attack_types = {
    'normal': 0,
    'back': 1,
    'neptune': 3,
    ...  # KDD attacks
}

# BEFORE (lines 167-186): COMMENT OUT KDD, UNCOMMENT CICIDS2017
'''
attack_types = {
    'BENIGN': 0,
    'Bot': 1,
    ...  # CICIDS2017 attacks
}
'''

# AFTER:
# COMMENT OUT KDD (lines 68-125)
'''
attack_types = {
    'normal': 0,
    ...  # KDD attacks
}
'''

# UNCOMMENT CICIDS2017 (lines 167-186)
attack_types = {
    'BENIGN': 0,
    'Bot': 1,
    'DDoS': 2,
    'DoS GoldenEye': 3,
    'DoS Hulk': 4,
    'DoS Slowhttptest': 5,
    'DoS slowloris': 6,
    'FTP-Patator': 7,
    'Heartbleed': 8,
    'Infiltration': 9,
    'PortScan': 10,  # ✅ PortScan is here!
    'SSH-Patator': 11,
    'Web Attack  Brute Force': 12,
    'Web Attack  Sql Injection': 13,
    'Web Attack  XSS': 14,
}
```

### Impact
- ✅ Attack type labels match dataset
- ✅ Zero-day attack "PortScan" will map correctly
- ✅ Proper multiclass evaluation

---

## 📝 Implementation Checklist

### Step 1: Fix GradScaler Crash ⭐ **DO THIS FIRST**
- [ ] Open `models/transductive_fewshot_model.py`
- [ ] Go to line 2680
- [ ] Replace `scaler.unscale_()` with conditional check
- [ ] Add else branch for CPU mode
- [ ] Test: Run training and verify no crash

### Step 2: Fix Attack Type Dictionary
- [ ] Open `config.py`
- [ ] Comment out KDD attack_types (lines 68-125)
- [ ] Uncomment CICIDS2017 attack_types (lines 167-186)
- [ ] Save file

### Step 3: Verify Zero-Day Attack Exists
- [ ] Check CICIDS2017_test.csv contains "PortScan" samples
- [ ] If not, update `config_loader.py` line 72 to use attack that exists
- [ ] Verify preprocessor correctly encodes attack types

### Step 4: Increase TTT Learning Rate
- [ ] Open `config_loader.py`
- [ ] Line 81: Change `'ttt_lr': 0.002` to `'ttt_lr': 0.01`
- [ ] Save file

### Step 5: Run Test
- [ ] Run `python main.py`
- [ ] Check logs for:
   - [ ] No GradScaler error
   - [ ] Parameter change > 0.001 (not 0.000000)
   - [ ] Prediction difference > 10%
   - [ ] Zero-day samples > 0 in test set
   - [ ] ZDR > 0.0000

---

## 🎯 Expected Results After Fixes

### Before Fixes:
```
TTT Adaptation FAILED: AttributeError
Parameter change: 0.000000
Prediction difference: 0.0%
Zero-day samples: 0
ZDR: 0.0000
```

### After Fixes:
```
TTT Adaptation SUCCESSFUL ✅
Parameter change: 0.0123 ✅
Prediction difference: 23.5% ✅
Zero-day samples: 89 ✅
ZDR: 0.7234 ✅
TTT F1 > Base F1 ✅
```

---

## 🔬 Verification Commands

### Check if fix worked:
```bash
# Run training
python main.py

# Check for errors
grep "TTT Adaptation FAILED" run_log.txt
# Should be EMPTY after fix

# Check parameter changes
grep "Parameter change:" run_log.txt
# Should show > 0.001, not 0.000000

# Check zero-day samples
grep "Zero-day samples:" run_log.txt
# Should show > 0, not 0
```

### Quick test script:
```python
import torch
from torch.cuda.amp import GradScaler

scaler = GradScaler(enabled=False)  # CPU mode
print(f"Scaler enabled: {scaler.is_enabled()}")
print(f"Has unscale_: {hasattr(scaler, 'unscale_')}")

# Test the fix
if scaler.is_enabled():
    print("Would use scaler.unscale_()")
else:
    print("CPU mode - skipping scaler operations ✅")
```

---

## 📞 Need Help?

If after applying these fixes TTT still doesn't work:

1. Check the error logs for new error messages
2. Verify zero-day attack samples exist in test set
3. Check if LR is still being scaled down somewhere
4. Investigate TTT probability collapse issue (separate problem)

---

## ✅ Summary

**The #1 reason TTT underperforms**: It crashes and doesn't run at all!

**Fix priority**:
1. 🔴 **CRITICAL**: Fix GradScaler crash (line 2680)
2. 🔴 **CRITICAL**: Fix zero-day samples missing
3. 🟡 **IMPORTANT**: Increase TTT LR
4. 🟡 **IMPORTANT**: Fix attack_types dictionary

**After these fixes**, TTT should:
- Actually run without crashing
- Update model parameters
- Show different predictions from base model
- Improve zero-day detection

Good luck! 🚀
