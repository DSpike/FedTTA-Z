# Base Model Performance Paradox - Investigation

## 🚨 Critical Finding

You've identified a major issue: **Base model performs BETTER on zero-day attacks (unseen) than on known attacks (trained on)!**

---

## 📊 Performance Summary (From Your Last Run)

### **Performance on Different Test Sets:**

| Test Set | Accuracy | F1-Score | Details |
|----------|----------|----------|---------|
| **Overall (All samples)** | 84.48% | 87.44% | Including zero-day |
| **Known Attacks Only** (EXCLUDING zero-day) | 77.84% | 82.35% | Normal + Known attacks |
| **Zero-Day Only** | 100.0% ✅ | 100.0% ✅ | 4/4 samples detected! |

### **The Paradox:**
```
Zero-Day Performance (100%) > Overall Performance (84%) > Known Attacks (78%)
```

**This is backwards!** The model should perform BEST on known attacks it was trained on!

---

## 🔍 Root Cause Analysis

### **Issue #1: Only 4 Zero-Day Samples** 🚨

From logs:
```
Zero-day samples: 4, Non-zero-day samples: 176
Zero-day attack: 'Generic', label: 1
```

**Problem:** With only 4 samples, the metrics are unreliable:
- 4/4 correct = 100% (looks perfect!)
- 3/4 correct = 75% (would look bad)
- **ONE misclassification changes ZDR by 25%!**

**Statistical Reliability:**
- Sample size: 4 is **WAY too small**
- Standard error: ~25% per sample
- 95% confidence interval: ±50%
- **Results are statistically meaningless**

### **Issue #2: Test Set Composition Severely Imbalanced**

```
Total test set: 180 samples
- Zero-day (Generic): 4 samples (2.2%)
- Known attacks: ??? samples
- Normal: ??? samples
```

**Problems:**
1. **Zero-day proportion too low** (2.2% vs typical 10-30%)
2. **Class imbalance** causing metric distortion
3. **Evaluation not representative** of real-world scenarios

### **Issue #3: "Generic" Attack Classification Issue**

From logs:
```
Zero-day attack: 'Generic', label: 1
```

**Question:** What is "Generic" attack?
- Is it a catch-all category?
- Or a specific attack type that happens to be rare?

**Hypothesis:** "Generic" might be:
1. **Misclassified samples** during preprocessing
2. **Outliers** that don't fit other categories
3. **Actual attack** but poorly represented in training

### **Issue #4: Possible Data Leakage**

**Suspicion:** The model might have seen these "zero-day" samples during training

**Evidence to check:**
1. Are "Generic" attacks truly excluded from training?
2. Is there overlap between train/test sets?
3. Are sequence boundaries properly maintained?

### **Issue #5: Evaluation Methodology Issue**

From logs:
```
Base model evaluation mode: INCLUDING all test samples
Base model evaluation mode: EXCLUDING zero-day samples
```

**Problem:** Different evaluation modes giving confusing results:
- WITH zero-day: 84.48% accuracy (better)
- WITHOUT zero-day: 77.84% accuracy (worse)

**This suggests:** The 4 zero-day samples are **easier** to classify than known attacks!

---

## 🎯 Detailed Performance Breakdown

### **From K-Fold Cross-Validation:**

```
Base Model (K-Fold on known attacks):
  Accuracy: 67.78% ± 6.2%
  F1-Score: 64-78% (varies by fold)
  Fold accuracies: [61%, 78%, 64%, 72%, 64%]
```

**Observations:**
1. **High variance** (61% to 78%)
2. **Unstable performance** across folds
3. **Average performance mediocre** (67.8%)

### **On Filtered Test Set (Excluding Zero-Day):**

```
Known Attacks + Normal:
  Accuracy: 77.84%
  F1-Score: 82.35%
  Precision: 87.50%
  Recall: 77.78%
```

**Better than K-fold but still not great**

### **On Zero-Day (Only 4 Samples):**

```
Zero-Day Detection:
  Accuracy: 100% (4/4 correct)
  F1-Score: 100%
  AUC-PR: 1.0000
```

**Suspiciously perfect - too good to be true!**

---

## 🔬 Possible Explanations

### **Explanation #1: Lucky Guess (Most Likely)** ⭐

With only 4 samples, getting all 4 correct is not statistically significant:
- Random chance of 4/4 at 50% baseline: (0.5)^4 = 6.25%
- With a 70% classifier: (0.7)^4 = 24%
- **NOT that unlikely!**

### **Explanation #2: "Generic" is Actually Easy**

The "Generic" attack might have characteristics that are:
- Very different from normal traffic (easy to detect)
- Similar to other attacks the model saw (partial overlap)
- Outliers that are obviously anomalous

### **Explanation #3: Test Set Contamination**

The 4 "Generic" samples might have:
- Similar patterns to training data
- Been inadvertently included in training
- Sequential overlap with training samples

### **Explanation #4: Evaluation Bug**

The evaluation code might be:
- Incorrectly filtering samples
- Miscalculating metrics
- Using wrong labels

---

## 🚨 Critical Issues to Fix

### **Issue 1: Insufficient Zero-Day Samples** (CRITICAL)

**Current:** 4 samples (2.2% of test set)
**Needed:** At least 30-50 samples (10-20% of test set)

**Why:** Statistical reliability requires minimum sample size

**Fix:**
1. Check test set composition
2. Increase zero-day proportion to 10-30%
3. Ensure at least 50+ zero-day samples

### **Issue 2: Verify Zero-Day Selection**

**Questions to answer:**
1. How is "Generic" defined in the dataset?
2. Is it truly unseen during training?
3. Are there other attack types that should be zero-day?

**Fix:**
1. Verify zero_day_attack configuration
2. Check if "Generic" samples are in training data
3. Consider using a more common attack as zero-day (e.g., "PortScan")

### **Issue 3: Class Imbalance**

**Current distribution unknown** - need to check:
```
- How many Normal samples?
- How many Known attack samples (by type)?
- How many Zero-day samples?
```

**Fix:** Balance test set or use stratified sampling

### **Issue 4: Evaluation Consistency**

**Current:** Multiple evaluation modes causing confusion

**Fix:** Use single, clear evaluation protocol:
1. Evaluate on ALL test samples (get overall metrics)
2. Separately report zero-day-specific metrics
3. Separately report known-attack metrics
4. Compare fairly

---

## 🔍 Diagnostic Steps

### **Step 1: Check Test Set Composition**

```python
# Add this diagnostic code
import pandas as pd

# Load test set
test_df = pd.read_csv('CICIDS2017_test.csv')

# Check class distribution
print("Test Set Class Distribution:")
print(test_df['Label'].value_counts())
print(f"\nTotal samples: {len(test_df)}")

# Check zero-day attack
zero_day_attack = 'Generic'  # From config
zero_day_count = (test_df['Label'] == zero_day_attack).sum()
print(f"\nZero-day attack ({zero_day_attack}): {zero_day_count} samples ({zero_day_count/len(test_df)*100:.2f}%)")
```

### **Step 2: Verify Train/Test Split**

```python
# Check for data leakage
train_df = pd.read_csv('CICIDS2017_train.csv')
test_df = pd.read_csv('CICIDS2017_test.csv')

# Check if zero-day attack is in training data
zero_day_in_train = (train_df['Label'] == 'Generic').sum()
print(f"Zero-day attack in training: {zero_day_in_train} samples")

if zero_day_in_train > 0:
    print("⚠️ WARNING: Data leakage detected! Zero-day attack found in training data!")
```

### **Step 3: Re-run with Different Zero-Day Attack**

Try a more common attack with more samples:
```python
# config_loader.py line 72
'zero_day_attack': "PortScan",  # Has 31,761 samples in dataset
```

### **Step 4: Check Sequence Boundaries**

Verify that sequences don't span train/test boundary

---

## 📝 Recommendations

### **Immediate Actions:**

1. **Check "Generic" attack distribution**
   ```bash
   grep -c "Generic" CICIDS2017_train.csv
   grep -c "Generic" CICIDS2017_test.csv
   ```

2. **Switch to "PortScan" as zero-day** (more samples)
   ```python
   'zero_day_attack': "PortScan"  # 31,761 samples available
   ```

3. **Increase zero-day proportion in test set**
   - Target: 10-30% of test set
   - Minimum: 50+ samples for statistical reliability

4. **Verify no data leakage**
   - Ensure zero-day attack NOT in training data
   - Check sequence boundaries

### **Long-term Solutions:**

1. **Use stratified sampling** for train/test split
2. **Ensure minimum samples per class** (50+ for rare classes)
3. **Report confidence intervals** for small sample sizes
4. **Use multiple zero-day scenarios** (not just one attack type)

---

## 🎯 Expected Results After Fixes

### **With "PortScan" as Zero-Day (31,761 samples):**

```
Expected Zero-Day Detection: 70-85%
Expected Known Attacks: 80-90%
Expected Overall: 75-88%
```

**This would be normal:** Known > Overall > Zero-Day

### **With Proper Test Set (10-30% zero-day):**

- Statistically reliable metrics
- Proper confidence intervals
- Meaningful performance comparison

---

## Summary

**The Paradox:**
- Zero-Day: 100% (4/4 samples) ✅ Suspiciously perfect
- Known: 77.84% ⚠️ Should be higher
- Overall: 84.48%

**Root Causes:**
1. ✅ Only 4 zero-day samples (statistically unreliable)
2. ⚠️ "Generic" attack might be unusual/easy
3. ⚠️ Possible data leakage
4. ⚠️ Test set severely imbalanced (2.2% zero-day)

**Immediate Fix:**
Switch to "PortScan" as zero-day attack → more samples, reliable metrics

**Next Steps:**
1. Verify "Generic" is NOT in training data
2. Check test set class distribution
3. Switch to PortScan or ensure 50+ Generic samples
4. Re-run evaluation with proper test set composition
