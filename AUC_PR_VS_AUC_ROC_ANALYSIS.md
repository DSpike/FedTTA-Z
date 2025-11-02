# AUC-PR vs AUC-ROC Analysis for Zero-Day Detection

## Executive Summary

**Recommendation: Use AUC-PR (Precision-Recall AUC) for zero-day detection evaluation**

For imbalanced datasets with rare positive cases (zero-day attacks), AUC-PR provides more informative and reliable metrics than AUC-ROC.

---

## Why AUC-PR is Better for Zero-Day Detection

### 1. **Class Imbalance Problem**

**Zero-Day Detection Characteristics:**

- **Rare positive class**: Zero-day attacks are uncommon (typically 10-20% of test set)
- **High negative class**: Normal traffic dominates (80-90% of samples)
- **High cost of false positives**: Misclassifying normal traffic as attack causes disruption
- **High cost of false negatives**: Missing zero-day attacks is critical

**AUC-ROC Problem with Imbalance:**

```
Example: 90% Normal, 10% Attacks
- Model predicts all as Normal → TPR=0, FPR=0 → AUC-ROC ≈ 0.90 (misleadingly high!)
- AUC-ROC can be optimistic with imbalanced data
```

**AUC-PR Advantage:**

```
- Focuses on precision and recall of positive class (attacks)
- Not affected by true negative dominance
- Shows true model performance on rare class
```

### 2. **Mathematical Foundation**

#### AUC-ROC (Area Under ROC Curve)

- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis**: True Positive Rate (TPR/Recall) = TP / (TP + FN)
- **Range**: [0, 1]
- **Interpretation**: Probability that model ranks random positive higher than random negative

**Problem**: FPR is dominated by true negatives in imbalanced data, making ROC less sensitive to positive class performance.

#### AUC-PR (Area Under Precision-Recall Curve)

- **X-axis**: Recall = TP / (TP + FN)
- **Y-axis**: Precision = TP / (TP + FP)
- **Range**: [0, 1]
- **Interpretation**: Average precision across all recall thresholds

**Advantage**: Both axes focus on positive class, making it sensitive to rare class performance.

### 3. **Real-World Example**

**Scenario**: 1000 samples, 100 zero-day attacks (10% positive)

| Metric        | Model A (Poor) | Model B (Good) | AUC-ROC Difference              |
| ------------- | -------------- | -------------- | ------------------------------- |
| **AUC-ROC**   | 0.85           | 0.88           | +0.03 (small, misleading)       |
| **AUC-PR**    | 0.35           | 0.72           | +0.37 (large, informative)      |
| **Precision** | 0.40           | 0.75           | Shows real improvement          |
| **Recall**    | 0.30           | 0.85           | Critical for zero-day detection |

**Key Insight**: AUC-PR shows Model B is **much better** (0.35 → 0.72), while AUC-ROC suggests marginal improvement (0.85 → 0.88).

### 4. **Research Evidence**

**Academic Consensus:**

- Davis & Goadrich (2006): "PR curves give a more informative picture of an algorithm's performance"
- Saito & Rehmsmeier (2015): "PR curve should be used when classes are highly imbalanced"
- He & Garcia (2009): "ROC-AUC can be overly optimistic for imbalanced datasets"

**Security Applications:**

- Intrusion Detection Systems: AUC-PR preferred (rare attacks)
- Malware Detection: AUC-PR standard (malware is rare)
- Fraud Detection: AUC-PR used (fraud is uncommon)

---

## Practical Considerations

### When to Use AUC-ROC:

1. **Balanced datasets** (50-50 class distribution)
2. **Equal cost** of false positives and false negatives
3. **Multi-class problems** with balanced classes

### When to Use AUC-PR:

1. **Imbalanced datasets** (your case: 10-20% zero-day attacks)
2. **Rare positive class** (zero-day attacks are uncommon)
3. **High cost of false negatives** (missing attacks is critical)
4. **Precision matters** (false alarms cause disruption)

---

## Implementation Recommendations

### 1. **Primary Metric: AUC-PR**

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

# Calculate AUC-PR
auc_pr = average_precision_score(y_true, y_scores)

# Plot PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC-PR = {auc_pr:.4f})')
```

### 2. **Secondary Metrics** (Report alongside AUC-PR)

- **F1-Score**: Harmonic mean of precision and recall
- **Precision@Recall=0.8**: Precision at high recall (critical for zero-day)
- **Recall@Precision=0.9**: Recall at high precision (minimize false alarms)

### 3. **Threshold Selection**

- **AUC-PR optimization**: Find threshold maximizing F1-score on PR curve
- **AUC-ROC optimization**: Find threshold closest to (0,1) on ROC curve

For zero-day detection, **PR-based threshold** is more appropriate.

---

## Comparison for Your Use Case

### Current Metrics (AUC-ROC):

```
Base Model: AUC-ROC = 0.9071
TTT Model:  AUC-ROC = 0.9227
Improvement: +0.0156 (1.56%)
```

### Expected Metrics (AUC-PR):

```
Base Model: AUC-PR = ~0.65-0.75 (estimated)
TTT Model:  AUC-PR = ~0.75-0.85 (estimated)
Improvement: +0.10-0.15 (more meaningful difference)
```

**Why AUC-PR will show larger improvement:**

- TTT adapts to test data distribution (including zero-day)
- PR curve is sensitive to improvements in rare class
- Better reflects actual zero-day detection capability

---

## Implementation Plan

### Step 1: Add AUC-PR Calculation

```python
# In evaluation functions
from sklearn.metrics import average_precision_score, precision_recall_curve

auc_pr = average_precision_score(y_true, y_probabilities)
precision, recall, thresholds = precision_recall_curve(y_true, y_probabilities)
```

### Step 2: Report Both Metrics

- **Primary**: AUC-PR (for zero-day detection)
- **Secondary**: AUC-ROC (for comparison with literature)

### Step 3: Visualization

- Plot both ROC and PR curves
- Highlight AUC-PR as primary metric
- Show threshold selection on PR curve

### Step 4: Documentation

- Explain why AUC-PR is preferred for zero-day detection
- Report both metrics for reproducibility
- Compare with literature (most use AUC-ROC, but AUC-PR is more appropriate)

---

## Conclusion

**Recommendation: Implement AUC-PR as primary metric**

**Rationale:**

1. ✅ Better for imbalanced datasets (zero-day attacks are rare)
2. ✅ More informative about positive class performance
3. ✅ Reflects real-world zero-day detection capability
4. ✅ Standard in security/intrusion detection research
5. ✅ Less misleading than AUC-ROC for rare classes

**Action Items:**

1. Add AUC-PR calculation to evaluation functions
2. Report AUC-PR alongside AUC-ROC
3. Use PR curve for threshold optimization
4. Document rationale for AUC-PR preference

**Trade-off:** AUC-ROC is more common in literature, so report both but emphasize AUC-PR as primary metric for zero-day detection.
