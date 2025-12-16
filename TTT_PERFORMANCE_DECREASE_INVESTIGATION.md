# TTT Performance Decrease Analysis

## 🔍 Problem Identified

**Current Run Results:**
- Base Model: **0.8645 ± 0.0358** (86.45%)
- TTT Model: **0.8555 ± 0.0433** (85.55%) ❌ **-0.9% DECREASE**

**Previous Successful Runs (from PROJECT_SUMMARY.md):**
- Base Model: **81.90% ± 4.45%**
- TTT Model: **88.83% ± 4.97%** ✅ **+6.93% IMPROVEMENT**

## 🚨 Root Causes

### **1. Configuration Mismatch**

**Current Config (Restored from GitHub):**
```python
num_rounds: int = 3          # Reduced for quick test
dirichlet_alpha: float = 100  # Near IID (very low heterogeneity)
use_pseudo_labels: bool = False  # Pure TENT only
```

**Previous Successful Config (from PROJECT_SUMMARY.md):**
```python
num_rounds: int = 3-15        # More rounds
dirichlet_alpha: float = 0.5  # Moderate non-IID (RECOMMENDED)
use_pseudo_labels: bool = True  # TENT + Pseudo-labels
```

### **2. Why TTT Performed Worse**

**Issue 1: Near-IID Data (`dirichlet_alpha=100`)**
- With `α=100`, data is nearly IID (homogeneous across clients)
- Base model already performs well (86.45%) because data is easy
- TTT has less room to improve when base is already strong
- **Previous runs used `α=0.5`** which creates realistic non-IID scenario where TTT helps more

**Issue 2: Pure TENT vs Pseudo-Labeling**
- Current run uses **pure TENT** (`use_pseudo_labels=False`)
- Pure TENT only provides **+2-5% improvement** (as documented)
- **Previous runs likely used pseudo-labeling** which provides **+8-12% improvement**
- Pure TENT might even hurt performance when base model is already strong

**Issue 3: Base Model Too Strong**
- Base model accuracy: **86.45%** (very high)
- With near-IID data and good base model, TTT adaptation can cause **overfitting**
- TTT adapts to query set but loses generalization
- This explains why TTT accuracy **decreased** (-0.9%)

### **3. Confidence Check May Skip Adaptation**

**Location:** `coordinators/simple_fedavg_coordinator.py` line 801
```python
if base_confidence > 0.92:
    logger.info("⏭️  Base model already very confident - skipping adaptation")
    return self.model  # Returns original model!
```

**Potential Issue:**
- If base confidence > 0.92, adaptation is skipped entirely
- This would return the base model as "TTT model"
- Need to verify if this check triggered in current run

## 🔧 Fixes Required

### **Fix 1: Restore Previous Configuration**

Update `config.py`:
```python
dirichlet_alpha: float = 0.5  # Change from 100 to 0.5 (moderate non-IID)
use_pseudo_labels: bool = True  # Change from False to True (enable pseudo-labeling)
num_rounds: int = 5  # Consider increasing from 3 (but 3 might be OK)
```

### **Fix 2: Verify Adaptation Actually Ran**

Check logs for:
- "⏭️ Base model already very confident - skipping adaptation" message
- If this appears, the confidence check is preventing TTT from running

### **Fix 3: Remove or Adjust Confidence Check**

The confidence check at line 801 might be too aggressive. Consider:
- Removing it entirely, OR
- Lowering threshold from 0.92 to 0.98, OR
- Making it configurable

## 📊 Expected Results After Fix

With correct configuration:
- **Base Model**: ~82-86% (depending on rounds)
- **TTT Model**: ~88-90% (with pseudo-labeling)
- **Improvement**: +6-8% ✅

## ✅ Next Steps

1. **Update config.py:**
   - `dirichlet_alpha: 100 → 0.5`
   - `use_pseudo_labels: False → True`

2. **Check if adaptation was skipped:**
   - Review logs for "skipping adaptation" message
   - Verify TTT actually ran

3. **Re-run with corrected config:**
   ```bash
   python main.py
   ```

4. **Compare results:**
   - Should see +6-8% improvement with pseudo-labeling
   - Should match previous successful runs

