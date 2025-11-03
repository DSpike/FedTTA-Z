# K-Fold CV Impact Analysis on TTT Performance

## Current Situation

### **Current Approach (Single Evaluation):**

- **Base Model**: Evaluates on **ALL test samples** (332 samples)
- **TTT Model**:
  - Adapts on **ALL test samples** (332 samples)
  - Evaluates on **ALL test samples** (332 samples)

### **Proposed K-Fold CV (k=3):**

- **Base Model**: Evaluates on **111 samples per fold** (332/3)
- **TTT Model**:
  - Adapts on **221 samples per fold** ((2/3)\*332)
  - Evaluates on **111 samples per fold** (1/3\*332)

---

## Concerns

### **1. Smaller Evaluation Set Per Fold**

- **Current**: 332 samples per evaluation
- **k-fold (k=3)**: 111 samples per fold
- **Impact**: ⚠️ Smaller per-fold evaluation might have higher variance

### **2. Reduced TTT Adaptation Data**

- **Current**: 332 samples for adaptation
- **k-fold (k=3)**: 221 samples per fold (↓ 33% reduction)
- **Impact**: ⚠️ TTT might adapt less effectively with less data

---

## Solutions

### **Solution 1: Increase k (More Folds)**

**Approach**: Use k=5 or k=10 instead of k=3

**k=5:**

- **Evaluation per fold**: 332/5 ≈ **66 samples**
- **TTT Adaptation per fold**: (4/5)\*332 ≈ **266 samples** ✅ (only 20% reduction)

**k=10:**

- **Evaluation per fold**: 332/10 ≈ **33 samples** (too small!)
- **TTT Adaptation per fold**: (9/10)\*332 ≈ **299 samples** ✅ (only 10% reduction)

**Trade-off**: More folds = better statistics BUT smaller evaluation sets per fold

---

### **Solution 2: Modified K-Fold (Asymmetric Split)**

**Approach**: Use larger portion for TTT adaptation, smaller for evaluation

**Example:**

- **TTT Adaptation**: Use 80% of test data (266 samples) ✅
- **Evaluation**: Use 20% of test data (66 samples)
- **Result**: TTT gets more adaptation data, evaluation still statistically valid

**Implementation:**

```python
# For TTT k-fold CV:
# - Use 80-20 split: 80% for adaptation, 20% for evaluation
# - Repeat k=5 times with different 20% hold-outs
```

---

### **Solution 3: Nested Approach (Best of Both)**

**Approach**: Combine both methods

1. **For Point Estimates** (Single Evaluation):

   - Keep current approach: Use ALL test data
   - This gives maximum TTT performance

2. **For Statistical Robustness** (k-fold CV):
   - Run k-fold CV for variance estimation
   - Use larger k (k=5 or k=10)
   - Report BOTH: point estimate (full data) + k-fold statistics

**Result**:

- ✅ Maximum TTT performance (point estimate)
- ✅ Statistical robustness (k-fold variance)
- ✅ Best of both worlds

---

### **Solution 4: Bootstrap Instead of K-Fold**

**Approach**: Bootstrap resampling with replacement

**Advantages**:

- ✅ Can use full test set for each bootstrap iteration
- ✅ TTT always gets full adaptation data
- ✅ More samples = better statistics

**Implementation**:

- Run 20-50 bootstrap iterations
- Each iteration: Resample 332 samples WITH replacement
- TTT adapts on full resampled set, evaluates on it
- Calculate mean and std across bootstrap iterations

---

## Recommendation

### **Best Approach: Solution 3 (Nested) + Solution 1 (Larger k)**

**Implementation:**

1. **Keep Single Evaluation** for point estimates:

   - Use ALL 332 test samples
   - TTT adapts on ALL samples
   - This gives maximum performance

2. **Add k-fold CV** for statistical robustness:

   - Use **k=5** (instead of k=3)
   - TTT adapts on **266 samples per fold** (only 20% reduction)
   - Evaluates on **66 samples per fold**
   - Calculate variance across 5 folds

3. **Report Both**:
   - Point estimate: Full dataset evaluation (maximum performance)
   - Statistical variance: k-fold CV (k=5) for confidence intervals

**Benefits:**

- ✅ TTT gets 80% of data for adaptation (266/332) - minimal performance loss
- ✅ Statistical robustness from 5-fold CV
- ✅ Fair comparison (both models use same folds)
- ✅ IEEE publication-ready (real variance, not estimates)

---

## Expected Impact

### **TTT Performance with k=5:**

- **Adaptation data**: 266 samples (vs 332) = **↓20% reduction**
- **Expected impact**: Minor (TTT entropy minimization doesn't need much data)
- **Evaluation per fold**: 66 samples (vs 332) = **↓80% reduction**
- **Impact on metrics**: Higher variance per fold, but mean should be similar

### **Statistical Robustness:**

- **Before**: Estimated std dev (1.5% of mean)
- **After**: Real std dev from 5-fold CV ✅
- **Confidence**: Can claim statistical significance ✅

---

## Final Recommendation

✅ **Use k=5 fold CV** with:

- TTT adapts on (4/5) = 80% of test data per fold
- Both models evaluated on same (1/5) hold-out per fold
- Report both: full-dataset point estimate + k-fold statistics

This minimizes TTT performance loss while providing real statistical variance.
