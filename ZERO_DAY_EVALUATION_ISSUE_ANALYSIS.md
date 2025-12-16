# Zero-Day Evaluation Issue Analysis

## Problem Statement

**User Concern**: 
1. Zero-Day Detection Rate (ZDR) is calculated as attack detection rate, not actual zero-day identification
2. No separate ROC/PR curves for zero-day vs known attacks

## Analysis

### ✅ Issue 1: ZDR Calculation - **CORRECT**

**Location**: `main.py:4776-4795`

**Current Implementation**:
```python
# ZDR = TP / (TP + FN) for zero-day samples = recall on zero-day samples
zero_day_tp = ((zero_day_predictions == 1) & (zero_day_actual == 1)).sum()
zero_day_fn = ((zero_day_predictions == 0) & (zero_day_actual == 1)).sum()
zero_day_detection_rate = zero_day_tp / (zero_day_tp + zero_day_fn) if (zero_day_tp + zero_day_fn) > 0 else 0.0
```

**Analysis**:
- ✅ **ZDR is correctly calculated** as recall on zero-day samples only
- ✅ It measures: "Of all zero-day attacks, how many were detected?"
- ✅ It is **NOT** just overall attack detection rate
- ✅ Zero-day mask is correctly created using `zero_day_attack_label` (line 2378)

**Conclusion**: ZDR calculation is **CORRECT** - it specifically measures zero-day attack detection, not overall attack detection.

---

### ❌ Issue 2: Separate ROC/PR Curves - **MISSING**

**Location**: 
- Calculation: `main.py:2629-2654` (zero-day PR curve calculated)
- Storage: `main.py:2721` (stored in results as `zero_day_pr_curve`)
- Visualization: `visualization/performance_visualization.py:1183-1341` (NOT plotted separately)

**Current Implementation**:

1. **Zero-day PR curve IS calculated**:
   ```python
   # Line 2632-2638: Zero-day-specific PR curve calculated
   zero_day_precision_curve, zero_day_recall_curve, zero_day_pr_thresholds = precision_recall_curve(
       zero_day_y_true_bin, zero_day_attack_probs
   )
   zero_day_pr_curve = {
       'precision': zero_day_precision_curve.tolist(),
       'recall': zero_day_recall_curve.tolist(),
       'thresholds': zero_day_pr_thresholds.tolist()
   }
   ```

2. **Zero-day PR curve IS stored**:
   ```python
   # Line 2721: Stored in results
   'pr_curve': zero_day_pr_curve,  # Zero-day-specific PR curve
   ```

3. **But NOT plotted separately**:
   - `plot_roc_curves()` only plots overall ROC curves (all samples)
   - `plot_pr_curves()` only plots overall PR curves (all samples)
   - Zero-day-specific curves are calculated but never visualized

**Problem**:
- ❌ Cannot visually compare zero-day vs known attack performance
- ❌ Overall curves mix zero-day and known attacks, making it hard to assess zero-day-specific performance
- ❌ Zero-day PR curve data exists but is unused

**Impact**:
- Hard to assess if model performs differently on zero-day vs known attacks
- Cannot identify if improvements are due to zero-day or known attack detection
- Missing critical visualization for zero-day detection evaluation

---

## Root Cause

1. **Zero-day PR curve is calculated** but stored in nested structure (`results['zero_day']['pr_curve']`)
2. **Visualization functions** (`plot_roc_curves`, `plot_pr_curves`) only access top-level `roc_curve` and `pr_curve`
3. **No separate plotting function** for zero-day vs known attack curves

---

## Solution

### Fix 1: Add Separate Zero-Day vs Known Attack ROC/PR Curves

**Implementation Plan**:

1. **Extract zero-day and known attack data** from results
2. **Calculate ROC/PR curves separately** for:
   - Zero-day attacks only
   - Known attacks only
   - Overall (for comparison)
3. **Create new visualization function** to plot all three curves together
4. **Add to main visualization pipeline**

**Code Changes Needed**:

1. **In `main.py`**: Ensure zero-day and known attack data are properly separated and stored
2. **In `visualization/performance_visualization.py`**: 
   - Add `plot_zero_day_vs_known_roc_curves()` function
   - Add `plot_zero_day_vs_known_pr_curves()` function
3. **In `main.py`**: Call new visualization functions in `generate_performance_visualizations()`

**Expected Output**:
- Separate ROC curves: Zero-Day, Known Attacks, Overall
- Separate PR curves: Zero-Day, Known Attacks, Overall
- Clear comparison showing if model performs differently on zero-day vs known attacks

---

## Verification

### Current State:
- ✅ ZDR correctly calculated (zero-day recall)
- ✅ Zero-day PR curve calculated
- ❌ Zero-day PR curve NOT visualized
- ❌ No separate ROC curves for zero-day vs known attacks

### After Fix:
- ✅ ZDR correctly calculated (zero-day recall)
- ✅ Zero-day PR curve calculated
- ✅ Zero-day PR curve visualized separately
- ✅ Separate ROC curves for zero-day vs known attacks
- ✅ Clear comparison of zero-day vs known attack performance

---

## Recommendation

**Priority**: HIGH

This is a critical issue for zero-day detection evaluation because:
1. Zero-day attacks are the primary focus of the research
2. Cannot assess if improvements are zero-day-specific or overall
3. Missing visualization makes it hard to understand model behavior
4. Publication requires clear zero-day vs known attack comparison

**Action Items**:
1. ✅ Verify ZDR calculation is correct (DONE - it is correct)
2. ❌ Add separate ROC/PR curves for zero-day vs known attacks (TODO)
3. ❌ Update visualization pipeline to include new curves (TODO)
4. ❌ Test with actual data to ensure curves are meaningful (TODO)

