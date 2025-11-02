# Zero-Day Detection Rate vs Overall AUC-PR: Why They Differ

## 🔍 The Issue

Your TTT model shows:

- **Zero-day detection rate: 1.0000** (perfect - 100% of zero-day attacks detected)
- **AUC-PR: 0.9629** (high, but not perfect)
- **Overall Accuracy: 0.8886** (not perfect)

**Question:** Why is AUC-PR not 1.0 if zero-day detection is perfect?

## 📊 Root Cause Analysis

### What Each Metric Measures:

1. **Zero-Day Detection Rate** (1.0000):

   - Calculated ONLY on **zero-day samples** (37 samples)
   - Formula: `(zero_day_predictions == 1).mean()`
   - Result: All 37 zero-day samples correctly predicted as attacks ✅
   - **Scope:** Only zero-day samples

2. **AUC-PR** (0.9629):

   - Calculated on **ENTIRE test set** (332 samples)
   - Includes:
     - 37 zero-day samples (perfect detection ✅)
     - 295 non-zero-day samples (some errors ⚠️)
   - **Scope:** All test samples (zero-day + non-zero-day)

3. **Overall Accuracy** (0.8886):
   - Calculated on **ENTIRE test set** (332 samples)
   - Includes errors from both zero-day and non-zero-day samples
   - **Scope:** All test samples

## 🔢 Breakdown of Your Results

From your latest run:

```
Test Set Composition:
├─ Zero-day samples: 37 (11.1%)
└─ Non-zero-day samples: 295 (88.9%)
   Total: 332 samples

Zero-Day Only Performance:
├─ Accuracy: 1.0000 ✅ (perfect)
├─ Precision: 1.0000 ✅ (perfect)
├─ Recall: 1.0000 ✅ (perfect)
└─ Detection Rate: 1.0000 ✅ (perfect)

Non-Zero-Day Performance:
├─ Accuracy: 0.8746 ⚠️ (some errors)
├─ Precision: 0.8713 ⚠️ (some errors)
├─ Recall: 0.9412 ⚠️ (some errors)
└─ F1-Score: 0.9049 ⚠️ (some errors)

Overall Performance (Combined):
├─ Accuracy: 0.8886 (weighted average)
├─ AUC-PR: 0.9629 (includes non-zero-day errors)
└─ ROC AUC: 0.9311 (includes non-zero-day errors)
```

## 💡 Why AUC-PR is Lower Than Zero-Day Detection Rate

### Mathematical Explanation:

1. **Zero-Day Detection Rate (1.0)**:

   - Only considers 37 zero-day samples
   - All 37 correctly predicted as attacks
   - Perfect score: 1.0

2. **AUC-PR (0.9629)**:
   - Considers ALL 332 samples
   - Zero-day: 37 samples, all correct (contributes ~1.0)
   - Non-zero-day: 295 samples, some errors (contributes ~0.87)
   - **Weighted average:** (37×1.0 + 295×0.87) / 332 ≈ 0.89
   - AUC-PR also considers precision at different recall thresholds
   - False positives on non-zero-day samples reduce precision
   - False negatives on non-zero-day samples reduce recall
   - **Result:** AUC-PR = 0.9629 (high, but not perfect due to non-zero-day errors)

### Visual Example:

```
Zero-Day Samples (37):
[✅ ✅ ✅ ✅ ✅ ... ✅ ✅ ✅]  ← All 37 correct = 100%

Non-Zero-Day Samples (295):
[✅ ✅ ❌ ✅ ✅ ❌ ✅ ✅ ❌ ... ✅ ✅]  ← Some errors = 87.46%

Combined (332):
[✅ ✅ ❌ ✅ ✅ ❌ ✅ ✅ ✅ ... ✅ ✅]  ← Overall = 88.86%
                                    AUC-PR = 96.29%
```

## 🎯 Key Insight

**AUC-PR is NOT measuring zero-day detection specifically** - it's measuring **overall binary classification performance** (attack vs normal) across ALL test samples.

### Two Different Questions:

1. **"How well does the model detect zero-day attacks?"**

   - Answer: **1.0000** (zero-day detection rate)
   - This is perfect! ✅

2. **"How well does the model classify attacks vs normal overall?"**
   - Answer: **0.9629** (AUC-PR)
   - This includes errors on non-zero-day samples ⚠️

## 📈 Why This Makes Sense

### Why Zero-Day Detection Can Be Perfect While Overall AUC-PR is Lower:

1. **Perfect Zero-Day Detection**:

   - TTT model successfully adapts to zero-day patterns
   - All 37 zero-day samples correctly identified as attacks
   - Zero-day specific performance: 1.0 ✅

2. **Imperfect Overall Performance**:

   - Model still makes errors on non-zero-day samples
   - Some normal samples misclassified as attacks (false positives)
   - Some non-zero-day attacks misclassified as normal (false negatives)
   - These errors reduce overall AUC-PR from 1.0 to 0.9629

3. **Why AUC-PR Still High (0.9629)**:
   - Zero-day samples (perfect) contribute positively
   - Non-zero-day samples (87% accuracy) still perform reasonably well
   - Overall: Good but not perfect

## 🔬 Scientific Interpretation

### For Your Publication:

**This is actually GOOD news!** It shows:

1. ✅ **TTT successfully adapts to zero-day attacks** (100% detection)
2. ✅ **Overall performance remains high** (96.29% AUC-PR)
3. ⚠️ **Room for improvement on non-zero-day samples** (87.46% accuracy)

### Recommendation:

Report **both metrics** separately:

```
Zero-Day Detection Performance:
├─ Detection Rate: 1.0000 (100% of zero-day attacks detected)
└─ Zero-Day Specific Metrics: Perfect across all metrics

Overall Attack Detection Performance:
├─ AUC-PR: 0.9629 (excellent overall performance)
├─ ROC AUC: 0.9311 (good discrimination)
└─ Overall Accuracy: 0.8886 (weighted by sample distribution)
```

## 🎯 Conclusion

**Zero-day detection rate = 1.0** means the TTT model perfectly detects zero-day attacks.

**AUC-PR = 0.9629** means the overall binary classification performance is very good (96.29%), but not perfect due to errors on non-zero-day samples.

**These metrics are measuring different things:**

- Zero-day detection rate: Performance on zero-day samples ONLY
- AUC-PR: Performance on ALL samples (zero-day + non-zero-day)

**Both metrics are correct and valid** - they answer different questions about model performance!
