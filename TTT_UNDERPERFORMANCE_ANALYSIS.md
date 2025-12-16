# TTT Model Underperformance - Root Cause Analysis
**Date**: 2025-12-16
**Investigation**: Why TTT model is not outperforming base model

---

## 🔴 **CRITICAL FINDINGS: TTT Adaptation is FAILING**

### **Summary**
The TTT (Test-Time Training) model is **not actually adapting** due to a critical runtime error. The model returns **identical predictions** to the base model, causing TTT to underperform.

---

## 📊 Performance Comparison (Most Recent Run)

### Base Model Performance
```
Overall Performance:
  - Accuracy:  66.46%
  - F1-Score:  71.79%
  - AUC-PR:    65.42%
  - ROC AUC:   66.60%
  - MCC:       0.3208

Zero-Day Detection:
  - Zero-day samples found: 0 (0.0% of test set)
  - ZDR: 0.0000 (NO ZERO-DAY SAMPLES IN TEST SET)
```

### TTT Enhanced Model Performance
```
Overall Performance:
  - Accuracy:  54.02% ⬇ -12.44%
  - F1-Score:  70.14% ⬇ -1.65%
  - AUC-PR:    61.79% ⬇ -3.63%
  - ROC AUC:   44.66% ⬇ -21.94%
  - MCC:      -0.0021 ⬇ -0.3229
  - FAR:      100.00% 🔴 CRITICAL (predicts everything as attack)

Zero-Day Detection:
  - Zero-day samples found: 0 (0.0% of test set)
  - ZDR: 0.0000 (NO ZERO-DAY SAMPLES IN TEST SET)
```

**Result**: TTT model performs **WORSE** than base model on all metrics!

---

## 🔍 ROOT CAUSE #1: TTT Adaptation CRASHES

### Error Location
**File**: `coordinators/simple_fedavg_coordinator.py:2277`

### Error Details
```python
ERROR - TTT Adaptation FAILED: 'GradScaler' object has no attribute 'unscale_'
Error type: AttributeError

Traceback:
  File "coordinators/simple_fedavg_coordinator.py", line 2277, in adapt
    self.scaler.unscale_(optimizer)
AttributeError: 'GradScaler' object has no attribute 'unscale_'
```

### Impact
```
WARNING - Returning base model without TTT adaptation due to error
Adaptation Verification:
  - Prediction difference: 0.0% (0/100 samples changed)
  - Parameter change: 0.000000
WARNING - Only 0.0% predictions changed - adaptation may not be effective!
WARNING - Parameter change is very small (0.000000e+00) - model may not have adapted!
```

### Consequence
```
WARNING - Adapted model predictions are IDENTICAL to base model!
WARNING - No TTT adaptation data found on adapted model
```

**CONCLUSION**: TTT adaptation **COMPLETELY FAILED**. The "adapted" model is just the base model with no changes!

---

## 🔍 ROOT CAUSE #2: NO ZERO-DAY SAMPLES IN TEST SET

### Test Set Composition Issue
```
Line 725: Zero-Day Attacks Only (0 samples, 0.0% of test set)
Line 733: Non-Zero-Day Samples (635 samples, 100.0% of test set)

Line 802: Zero-day attack: 'Exploits', label: 5
Line 803: Test label distribution: tensor([292, 343])
Line 804: Zero-day samples: 0, Non-zero-day samples: 635

WARNING - No zero-day samples found! Check if 'Exploits' (label 5) exists in test data.
WARNING - Available labels in test data: [0, 1]
```

### Configuration Mismatch
```
Config says:
  - zero_day_attack: "PortScan" (from config_loader.py)

System looks for:
  - zero_day_attack: "Exploits" (label 5)

Test set contains:
  - Labels: [0, 1] (only Normal and Attack classes)
  - NO "Exploits" label!
  - NO "PortScan" label!
```

**CONCLUSION**: The zero-day attack type configured doesn't exist in the test set!

---

## 🔍 ROOT CAUSE #3: TTT Overconfidence (Collapse to "Attack" Class)

Even though TTT didn't actually adapt, when the system tries to evaluate it:

### TTT Probability Distribution
```
Base Model:
  - Attack prob range: [0.0000, 1.0000]
  - Attack prob mean: 0.6444, std: 0.4710
  - Attack prob median: 1.0000
  - Samples with prob > 0.9: 399/635 (62.8%)

TTT Model (after "adaptation"):
  - Attack prob range: [0.9673, 0.9984] ⚠️ COLLAPSED RANGE
  - Attack prob mean: 0.9958, std: 0.0052 ⚠️ NO VARIANCE
  - Attack prob median: 0.9981
  - Samples with prob > 0.9: 635/635 (100.0%) ⚠️ EVERYTHING HIGH CONFIDENCE

TTT Predictions (threshold=0.90):
  - Predicted Normal: 0/635 (0.0%) 🔴
  - Predicted Attack: 635/635 (100.0%) 🔴
  - Actual distribution: Normal=False, Attack=343
```

**CONCLUSION**: Even without actual adaptation, the evaluation shows TTT would predict **EVERYTHING as Attack**, causing 100% False Alarm Rate!

---

## 🔍 ROOT CAUSE #4: Configuration Issues

### Issue 1: GradScaler.unscale_() Compatibility
```python
# coordinators/simple_fedavg_coordinator.py:2277
self.scaler.unscale_(optimizer)  # ❌ Method doesn't exist in CPU mode
```

**Problem**: `GradScaler.unscale_()` is only available when mixed precision is enabled and CUDA is available. System is running in CPU mode.

### Issue 2: TTT Learning Rate Too Low
```
Configuration:
  - Learning rate: 0.00011136839897653453 (extremely small)
  - Trainable parameters: 1856
  - Steps: 293
```

Even if TTT worked, the learning rate is **too small** to cause meaningful adaptation in 293 steps.

### Issue 3: Mixed Precision in CPU Mode
```
Line 768: Mixed precision: Disabled (CPU mode)
```

But the code still tries to use GradScaler operations that require GPU/mixed precision.

---

## 📋 Critical Issues Summary

| Issue # | Problem | Impact | Severity |
|---------|---------|--------|----------|
| **1** | TTT crashes with GradScaler AttributeError | TTT doesn't adapt at all (returns base model) | 🔴 **CRITICAL** |
| **2** | No zero-day samples in test set | Can't measure ZDR (0.0000) | 🔴 **CRITICAL** |
| **3** | Zero-day attack mismatch ("Exploits" vs "PortScan") | System looks for wrong attack type | 🔴 **CRITICAL** |
| **4** | GradScaler.unscale_() used in CPU mode | Crashes during TTT adaptation | 🟠 **HIGH** |
| **5** | TTT learning rate too low (0.0001113) | Even if TTT worked, changes would be minimal | 🟡 **MEDIUM** |
| **6** | TTT would predict everything as Attack | 100% FAR even if adaptation worked | 🟠 **HIGH** |

---

## 💡 Why TTT Underperforms Base Model

**Primary Reason**: TTT doesn't actually run due to crashes. The system falls back to the base model.

**Secondary Reason (if TTT worked)**: The evaluation code path has bugs that would cause TTT to:
1. Predict everything as "Attack" (100% FAR)
2. Have worse accuracy and MCC than base model
3. Show no ability to distinguish normal vs attack traffic

---

## 🔧 Required Fixes

### Fix #1: Remove GradScaler.unscale_() in CPU Mode ⭐ **CRITICAL**

**File**: `coordinators/simple_fedavg_coordinator.py:2277`

```python
# BEFORE (line 2277):
self.scaler.unscale_(optimizer)

# AFTER:
if self.scaler.is_enabled():  # Only unscale if mixed precision is enabled
    self.scaler.unscale_(optimizer)
```

### Fix #2: Fix Zero-Day Attack Configuration ⭐ **CRITICAL**

**Option A**: Update config_loader.py to use correct attack type
```python
# config_loader.py line 72
'zero_day_attack': "PortScan",  # Current value

# Check if PortScan exists in CICIDS2017 dataset
# If not, change to an attack that exists, e.g., "Bot", "DDoS", etc.
```

**Option B**: Ensure test set includes configured zero-day attack type
- Verify CICIDS2017_test.csv contains "PortScan" samples
- Check label encoding matches config.py attack_types dictionary

### Fix #3: Increase TTT Learning Rate 🟡 **MEDIUM PRIORITY**

**File**: `config_loader.py` line 81 (or config.py)

```python
# BEFORE:
'ttt_lr': 0.002,  # But actual LR used is 0.0001113 (very low)

# AFTER:
'ttt_lr': 0.01,  # Increase by 5x for meaningful adaptation
```

### Fix #4: Fix TTT Overconfidence Issue 🟠 **HIGH PRIORITY**

This requires investigating:
1. Why TTT probabilities collapse to [0.96, 0.99]
2. Why all samples get predicted as "Attack"
3. Temperature scaling or calibration issues

**Investigate**: `models/transductive_fewshot_model.py` TTT adaptation logic

### Fix #5: Add Proper Error Handling ⚠️

```python
# coordinators/simple_fedavg_coordinator.py
try:
    # TTT adaptation
    if self.scaler.is_enabled():
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        self.scaler.step(optimizer)
        self.scaler.update()
    else:
        # CPU mode - no mixed precision
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
except Exception as e:
    logger.error(f"TTT step failed: {e}")
    # Continue with next step instead of aborting entirely
```

---

## 📈 Expected Impact After Fixes

### Fix #1 (GradScaler): ✅ TTT will actually run
- Predictions will differ from base model
- Parameters will actually update
- Real adaptation will occur

### Fix #2 (Zero-day config): ✅ ZDR will be measurable
- Test set will include zero-day samples
- Can measure ZDR improvement from TTT

### Fix #3 (Learning rate): ✅ Stronger adaptation
- More meaningful parameter updates
- Better test-time learning

### Fix #4 (Overconfidence): ✅ Better calibration
- FAR won't be 100%
- Better precision/recall balance

---

## 🎯 Immediate Action Items

1. **CRITICAL**: Fix GradScaler.unscale_() crash
   - File: `coordinators/simple_fedavg_coordinator.py:2277`
   - Add `if self.scaler.is_enabled():` guard

2. **CRITICAL**: Fix zero-day attack configuration
   - Verify "PortScan" exists in test set
   - Or change to attack that exists

3. **HIGH**: Investigate TTT overconfidence
   - Check why probabilities collapse to 0.96-0.99
   - Review temperature scaling

4. **MEDIUM**: Increase TTT learning rate
   - Change from 0.002 to 0.01 or higher

5. **RUN TEST**: After fixes, verify TTT actually adapts
   - Check parameter changes > 0
   - Check predictions differ from base model

---

## ✅ Verification Checklist

After applying fixes, verify:

- [ ] TTT runs without crashing
- [ ] Parameter change > 0.001 (not 0.000000)
- [ ] Prediction difference > 10% (not 0.0%)
- [ ] Test set contains zero-day samples > 0
- [ ] ZDR is measurable (not 0.0000)
- [ ] FAR is < 100% (not predicting everything as attack)
- [ ] TTT outperforms base model on at least one metric

---

## 📚 Files to Modify

1. **coordinators/simple_fedavg_coordinator.py** (line 2277)
   - Fix GradScaler.unscale_() crash

2. **config_loader.py** (line 72)
   - Verify/fix zero_day_attack setting

3. **config_loader.py** (line 81)
   - Increase ttt_lr from 0.002 to 0.01

4. **models/transductive_fewshot_model.py**
   - Investigate TTT overconfidence issue

---

## 🔬 Additional Investigation Needed

1. Why is the actual LR (0.0001113) much lower than configured (0.002)?
2. Where is "Exploits" coming from if config says "PortScan"?
3. Why does TTT probability distribution collapse?
4. Is temperature scaling (T=1.54) calibrated correctly?

---

**Priority**: Fix #1 (GradScaler crash) first, then Fix #2 (zero-day config), then run test to verify TTT works.
