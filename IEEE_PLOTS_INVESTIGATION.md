# IEEE Statistical Plots Investigation Report

## Summary

This document analyzes both the **statistical logic** and **visualization** code for IEEE plot generation. Several critical issues were identified that prevent accurate statistical reporting.

---

## File Structure

- **Main File**: `ieee_statistical_plots.py`
- **Called From**: `main.py` lines 5247-5283
- **Class**: `IEEEStatisticalVisualizer`

---

## 1. STATISTICAL LOGIC ISSUES

### Issue 1.1: Incorrect Key Names (Critical Bug)

**Location**: `plot_statistical_comparison()` lines 218, 237

**Problem**:

- Line 218: Uses `ttt_model` but should be `ttt_metatasks` (in k-fold format branch)
- Line 237: Looks for `ttt_model` but actual key is `adapted_model` (from `main.py:5223`)

**Current Code**:

```python
# Line 218 (inside k-fold branch):
ttt_means = [
    ttt_model.get('accuracy', 0.0),  # ❌ ttt_model is undefined!
    ...
]

# Line 237 (standard format):
ttt_model = real_results.get('ttt_model', {})  # ❌ Key doesn't exist!
```

**Actual Structure** (from `main.py:5221-5226`):

```python
evaluation_results = {
    'base_model': base_evaluation_results,
    'adapted_model': adapted_evaluation_results,  # ✅ Correct key name
    'comparison': {}
}
```

**Fix Required**:

- Line 218: Change `ttt_model` → `ttt_metatasks`
- Line 237: Change `ttt_model` → `adapted_model`

---

### Issue 1.2: Hardcoded Standard Deviations

**Location**: Multiple locations in `plot_statistical_comparison()`

**Problem**: Standard deviations are hardcoded to `0.01` instead of using real statistical values.

**Affected Lines**:

- Lines 224-230: TTT model stds (k-fold format) = all 0.01
- Lines 251-256: Base model stds (standard format) = all 0.01
- Lines 267-273: TTT model stds (standard format) = all 0.01

**Current Code**:

```python
base_stds = [
    0.01,  # ❌ Hardcoded - should extract from data
    0.01,  # ❌ Hardcoded
    0.01,  # ❌ Hardcoded
    0.01,  # ❌ Hardcoded
    0.01   # ❌ Hardcoded
]
```

**Impact**:

- **Statistical significance cannot be accurately calculated**
- **Confidence intervals are misleading** (all ±0.01 regardless of actual variance)
- **IEEE publication requirement violated**: Cannot claim statistical robustness without real std devs

**Fix Required**: Extract standard deviations from:

- K-fold results: `base_kfold.get('accuracy_std', ...)`
- Meta-tasks results: `ttt_metatasks.get('accuracy_std', ...)`
- Or calculate from multiple evaluation runs

---

### Issue 1.3: Missing K-Fold and Meta-Tasks Evaluation

**Location**: `main.py` - No actual k-fold/meta-tasks evaluation runs

**Problem**:

- `main.py` passes `evaluation_results` with only `base_model` and `adapted_model`
- **No `base_model_kfold` or `ttt_model_metatasks` keys exist**
- Code falls back to hardcoded dummy values (lines 277-282)

**Evidence**:
From run output:

```
📊 Using REAL standard evaluation results for IEEE statistical plots
  Base Model keys: ['model_type', 'accuracy', 'accuracy_sklearn', ...]
  TTT Model keys: []  # ❌ Empty!
```

**Root Cause**:

- `main.py` calls `evaluate_base_model_only()` and `evaluate_adapted_model()`
- These functions **do NOT** perform k-fold CV or meta-tasks evaluation
- The statistical robustness methods (`_evaluate_base_model_kfold`, `_evaluate_ttt_model_metatasks`) exist but are **never called** in the main flow

**Fix Required**:

1. Actually run k-fold CV and meta-tasks evaluation in `main.py`
2. Store results in `evaluation_results['base_model_kfold']` and `evaluation_results['ttt_model_metatasks']`
3. Or remove k-fold branch if not implementing statistical robustness

---

### Issue 1.4: Simulated Data in Other Plots

**Location**: `plot_kfold_cross_validation_results()`, `plot_meta_tasks_evaluation_results()`, `plot_effect_size_analysis()`

**Problem**: These plots use **hardcoded/simulated** data instead of real evaluation results.

**Affected Functions**:

1. **`plot_kfold_cross_validation_results()`** (lines 86-132):

   - Lines 92-93: Simulated fold results

   ```python
   accuracy_scores = [0.712, 0.718, 0.715, 0.720, 0.714]  # ❌ Simulated
   f1_scores = [0.708, 0.715, 0.712, 0.718, 0.713]  # ❌ Simulated
   ```

   - Lines 109-110: Simulated means/stds

   ```python
   means = [0.716, 0.713, 0.433]  # ❌ Simulated
   stds = [0.003, 0.004, 0.008]  # ❌ Simulated
   ```

2. **`plot_meta_tasks_evaluation_results()`** (lines 134-178):

   - Lines 140-141: Simulated meta-task results

   ```python
   task_accuracies = np.random.normal(0.724, 0.021, 100)  # ❌ Simulated
   task_f1_scores = np.random.normal(0.723, 0.021, 100)  # ❌ Simulated
   ```

3. **`plot_effect_size_analysis()`** (lines 357-403):
   - Lines 363, 385: Simulated effect sizes and power values
   ```python
   effect_sizes = [0.35, 0.33, 0.34, 0.35, 0.23]  # ❌ Simulated
   power_values = [0.85, 0.92, 0.96, 0.98, 0.99]  # ❌ Simulated
   ```

**Impact**:

- **Not publication-ready**: Cannot use simulated data in IEEE papers
- **Misleading results**: Plots show fake statistical robustness
- **Ethical issue**: Claiming statistical validity without real calculations

---

## 2. VISUALIZATION ISSUES

### Issue 2.1: Plot Style Inconsistency

**Location**: IEEE style setup (lines 18-37)

**Observation**:

- Good IEEE formatting (serif fonts, 300 DPI, etc.)
- But inconsistent with main visualization style (Times New Roman in `performance_visualization.py`)

**Impact**: Minor - visual consistency issue

---

### Issue 2.2: Missing Error Handling

**Location**: All plot functions

**Problem**: No try-except blocks for plot generation failures

**Impact**:

- Silent failures could occur
- No graceful degradation if matplotlib/seaborn errors

---

### Issue 2.3: Annotation Positioning Issues

**Location**: `plot_statistical_comparison()` lines 296-305, 322-338

**Issue**:

- Smart positioning logic tries to avoid overlaps
- But may fail with extreme values
- Line 319: `ax.set_ylim(-0.2, 1.0)` extends negative range to prevent overlap - unusual for metric plots

---

## 3. DATA FLOW ANALYSIS

### Current Flow:

```
main.py (line 5206)
  └─> evaluate_base_model_only()
       └─> Returns: base_evaluation_results
            └─> Single evaluation (no k-fold, no std dev)

  └─> evaluate_adapted_model()
       └─> Returns: adapted_evaluation_results
            └─> Single evaluation (no meta-tasks, no std dev)

  └─> evaluation_results = {
        'base_model': base_evaluation_results,      # ✅ Has accuracy, f1, etc.
        'adapted_model': adapted_evaluation_results, # ✅ Has accuracy, f1, etc.
        'comparison': {}                              # ✅ Has comparison
      }

  └─> ieee_visualizer.plot_statistical_comparison(real_results=evaluation_results)
       └─> Tries to find 'base_model_kfold' ❌ (doesn't exist)
       └─> Falls back to 'base_model' ✅ (exists)
       └─> Tries to find 'ttt_model' ❌ (doesn't exist - should be 'adapted_model')
       └─> Gets empty dict → all values = 0.0
       └─> Uses hardcoded stds = 0.01 for all metrics
```

### Expected Flow (for statistical robustness):

```
main.py
  └─> evaluate_base_model_kfold()  # ❌ NOT CALLED
       └─> Returns: {'accuracy_mean': X, 'accuracy_std': Y, ...}

  └─> evaluate_ttt_model_metatasks()  # ❌ NOT CALLED
       └─> Returns: {'accuracy_mean': X, 'accuracy_std': Y, ...}

  └─> evaluation_results = {
        'base_model': {...},              # Single evaluation
        'adapted_model': {...},           # Single evaluation
        'base_model_kfold': {...},       # ❌ MISSING - should have k-fold results
        'ttt_model_metatasks': {...}     # ❌ MISSING - should have meta-tasks results
      }
```

---

## 4. ROOT CAUSE SUMMARY

1. **Key Name Mismatch**: Code expects `ttt_model` but receives `adapted_model`
2. **No Statistical Robustness Evaluation**: K-fold and meta-tasks methods exist but are never called
3. **Hardcoded Standard Deviations**: All std devs = 0.01 instead of real values
4. **Simulated Data in Plots**: 3 out of 5 plots use fake data

---

## 5. RECOMMENDATIONS

### Priority 1 (Critical - Fix Immediately):

1. ✅ Fix key name: `ttt_model` → `adapted_model` (line 237)
2. ✅ Fix undefined variable: `ttt_model` → `ttt_metatasks` (line 218)
3. ✅ Extract real standard deviations instead of hardcoded 0.01

### Priority 2 (Important - Statistical Validity):

4. ⚠️ Either implement actual k-fold CV and meta-tasks evaluation, OR
5. ⚠️ Remove k-fold/meta-tasks branches and use single evaluation with calculated std devs
6. ⚠️ Replace simulated data in k-fold, meta-tasks, and effect size plots with real data

### Priority 3 (Nice to Have):

7. Add error handling to plot functions
8. Align font style with main visualization (Times New Roman)
9. Improve annotation positioning logic

---

## 6. IMPACT ASSESSMENT

### Current State:

- ✅ Plot generation works (no crashes)
- ❌ Statistical values are incorrect/misleading
- ❌ Cannot claim statistical robustness for publication
- ❌ Standard deviations are meaningless (all 0.01)

### After Fixes:

- ✅ Accurate statistical reporting
- ✅ Real confidence intervals
- ✅ Publication-ready plots
- ✅ Valid statistical significance claims

---

## Conclusion

The IEEE plotting code has **structural issues** that prevent accurate statistical reporting:

1. **Key name bugs** prevent data extraction
2. **Hardcoded standard deviations** make statistical claims invalid
3. **Simulated data** in 3 plots makes them unusable for publication
4. **Missing evaluation methods** (k-fold, meta-tasks) are never executed

**Recommendation**: Fix critical bugs immediately, then decide whether to implement full statistical robustness evaluation or use simpler single-evaluation approach with proper error bars.
