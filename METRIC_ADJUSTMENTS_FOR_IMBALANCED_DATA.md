# Metric Adjustments for Imbalanced Zero-Day Detection

## Summary of Changes Implemented

### ✅ 1. AUC-PR Calculation Added

- **Base Model**: Now calculates AUC-PR (Precision-Recall AUC)
- **TTT Model**: Now calculates AUC-PR (Precision-Recall AUC)
- **Reporting**: AUC-PR marked as PRIMARY metric in logs
- **Comparison**: AUC-PR improvement included in performance comparison

### ✅ 2. Precision-Recall Curve Data

- PR curve data stored in results for visualization
- Can be used for threshold optimization

### ✅ 3. PR-Based Threshold Finder Function

- Added `find_optimal_threshold_pr()` function
- Optimizes threshold based on PR curve (F1-score, balanced, or precision)
- Better for imbalanced data than ROC-based threshold

---

## Recommended Additional Adjustments

### 1. **Threshold Optimization Method** (OPTIONAL but Recommended)

**Current**: TTT uses ROC-based threshold optimization
**Recommendation**: Use PR-based threshold for zero-day detection

**Why**: PR-based threshold focuses on precision/recall of attacks (rare class), while ROC-based is dominated by normal samples.

**Implementation**:

```python
# In evaluate_adapted_model(), replace ROC-based threshold:
# Current: find_optimal_threshold(y_test_binary, attack_probs, method='balanced')
# Recommended: find_optimal_threshold_pr(y_test_binary, attack_probs, method='f1')
```

**Trade-off**:

- ✅ Better reflects zero-day detection capability
- ⚠️ Might give different threshold than ROC-based (but more appropriate for rare class)

---

### 2. **Metrics Reporting Priority**

**Current Priority** (after changes):

1. ✅ AUC-PR (PRIMARY - emphasized in logs)
2. ROC AUC (secondary)
3. Accuracy, F1-Score (standard metrics)

**This is appropriate** - AUC-PR is now primary metric for your imbalanced zero-day detection case.

---

### 3. **Consider Weighted F1-Score**

**Current**: Using binary F1-score (appropriate)
**Alternative**: Weighted F1-score (already calculated but not emphasized)

**Recommendation**: Keep binary F1-score as primary. Your current approach is correct.

---

### 4. **Precision@High Recall Metrics**

**Optional Addition**: Report precision at specific recall thresholds

- Precision@Recall=0.8: "If we want to catch 80% of attacks, what's the precision?"
- Precision@Recall=0.9: "If we want to catch 90% of attacks, what's the precision?"

**Use Case**: Production deployment decisions ("We need to catch 90% of attacks, can we accept the false alarm rate?")

---

### 5. **Zero-Day Specific AUC-PR**

**Optional**: Calculate AUC-PR separately for:

- Zero-day attacks only
- Non-zero-day attacks only

**Benefit**: Shows TTT's impact specifically on zero-day detection

---

## Implementation Status

### ✅ Completed:

1. AUC-PR calculation for both models
2. PR curve data storage
3. AUC-PR in results dictionary
4. AUC-PR in logging (marked as PRIMARY)
5. AUC-PR improvement in comparison
6. PR-based threshold finder function (available but not used yet)

### ⚠️ Optional Enhancements:

1. Use PR-based threshold for TTT (recommended)
2. Add Precision@Recall metrics (optional)
3. Add zero-day specific AUC-PR (optional)

---

## Current Metrics Summary

### Base Model Metrics:

- ✅ Accuracy
- ✅ Precision, Recall, F1-Score
- ✅ AUC-ROC (secondary)
- ✅ **AUC-PR (PRIMARY)** ⭐ NEW
- ✅ MCC
- ✅ Zero-day specific metrics (separate reporting)

### TTT Model Metrics:

- ✅ Accuracy
- ✅ Precision, Recall, F1-Score
- ✅ AUC-ROC (secondary)
- ✅ **AUC-PR (PRIMARY)** ⭐ NEW
- ✅ MCC
- ✅ Zero-day specific metrics (separate reporting)

### Comparison Metrics:

- ✅ Accuracy improvement
- ✅ F1-Score improvement
- ✅ AUC-ROC improvement
- ✅ **AUC-PR improvement (PRIMARY)** ⭐ NEW
- ✅ Zero-day detection improvement

---

## Recommended Next Steps

### High Priority:

1. ✅ **DONE**: Add AUC-PR calculation
2. **Consider**: Switch TTT threshold optimization to PR-based

### Medium Priority:

3. Add Precision@Recall metrics for production planning
4. Add zero-day specific AUC-PR

### Low Priority:

5. Visualization: Add PR curve plots alongside ROC curves
6. Documentation: Explain metric choice in paper

---

## Threshold Optimization Recommendation

**For Zero-Day Detection, Use PR-Based Threshold:**

**Current (ROC-based):**

```python
ttt_optimal_threshold, _, _, _, _ = find_optimal_threshold(
    y_test_binary, attack_probs, method='balanced', band=(0.1, 0.9))
```

**Recommended (PR-based):**

```python
ttt_optimal_threshold, _, _, _, _ = find_optimal_threshold_pr(
    y_test_binary, attack_probs, method='f1', min_recall=0.2)
```

**Why PR-based is better:**

- Optimizes F1-score on PR curve (focuses on attacks)
- Not affected by normal sample dominance
- More aligned with zero-day detection goal

---

## Conclusion

**Current metrics are now appropriate** with AUC-PR added as primary metric.

**Optional improvement**: Switch threshold optimization to PR-based for better zero-day detection alignment.

**No other critical adjustments needed** - your metric suite now properly addresses imbalanced zero-day detection!
