# Base Model High Zero-Day Performance Investigation

## 🔍 **Investigation Summary**

The base model is showing high performance on zero-day detection, but this is misleading due to **test set composition** and **insufficient zero-day samples** after sequence creation.

---

## 📊 **Current Situation**

From the latest run:

- **Zero-day samples identified**: 2 out of 332 sequences (0.6%)
- **Overall accuracy**: 90.06%
- **Zero-day-only accuracy**: 100% (but only 2 samples - not statistically meaningful)
- **Non-zero-day accuracy**: 90.00% (330 samples)

---

## ⚠️ **Root Causes**

### **1. Insufficient Zero-Day Samples After Sequence Creation**

**Problem**: Only 2 zero-day sequences are created from the test subset:

- Test subset: 5,000 samples (simple slicing `[:5000]` - **doesn't preserve class distribution**)
- After sequence creation: 332 sequences total
- Zero-day sequences: Only 2 sequences (0.6%)

**Why This Happens**:

1. Simple slicing takes the **first 5000 samples** from test data
2. Zero-day attacks might be distributed throughout the test set, not just at the beginning
3. Sequence creation further reduces the sample count
4. The mapping logic correctly identifies sequences, but there are too few to begin with

**Impact**:

- Zero-day metrics (100% accuracy) are **not statistically meaningful** with only 2 samples
- Overall performance (90%) is **inflated** because 330/332 samples (99.4%) are non-zero-day attacks the model HAS seen

---

### **2. Test Set Composition Bias**

**Current Test Set**:

- Normal samples: 93 (28%)
- Non-zero-day attacks: 237 (71.4%) - Model HAS seen these
- Zero-day attacks: 2 (0.6%) - Model has NOT seen these

**Why This Matters**:

- The model performs well (90%) because **99.4% of test samples are familiar**
- Only 0.6% are truly zero-day (unseen)
- High performance is expected since the model is mostly tested on seen samples

---

### **3. Zero-Day Exclusion is Working Correctly** ✅

**Verified**:

- Training data correctly excludes zero-day attacks (preprocessor lines 741-742)
- Meta-learning correctly excludes zero-day attacks (transductive_fewshot_model.py lines 979-980)
- **No data leakage detected**

---

## ✅ **Fixes Applied**

### **1. Stratified Sampling for Test Subset** (Just Added)

**File**: `main.py` - Method: `_stratified_test_subset()`

**What It Does**:

- Uses stratified sampling based on multiclass labels to preserve zero-day distribution
- Ensures zero-day attacks are proportionally represented in the test subset

**Expected Impact**:

- Should increase zero-day samples from 2 to ~50-100 sequences (depending on original distribution)
- Provides statistically meaningful zero-day metrics

---

## 🎯 **Expected Results After Fix**

### **Before (Current)**:

- Zero-day samples: 2 (0.6%)
- Zero-day accuracy: 100% (not meaningful)
- Overall accuracy: 90.06% (inflated)

### **After (With Stratified Sampling)**:

- Zero-day samples: ~50-100 (15-30%)
- Zero-day accuracy: ~50-70% (realistic for unseen attacks)
- Overall accuracy: ~75-85% (more realistic)

---

## 📋 **Next Steps**

1. **✅ Run system with stratified sampling** - Verify zero-day samples increase
2. **Analyze zero-day-only metrics** - Should show lower, more realistic performance
3. **Compare base vs TTT** on zero-day samples only (fair comparison)
4. **Verify no data leakage** - Confirm zero-day attacks are not in training

---

## 🔍 **Why Base Model Shows High Performance**

The base model's **90% accuracy is NOT due to zero-day detection success**. Instead:

1. **99.4% of test samples are familiar** (Normal + non-zero-day attacks)
2. **Only 0.6% are truly zero-day** (too few for meaningful metrics)
3. **The model correctly classifies familiar samples** (expected behavior)
4. **Zero-day metrics are unreliable** with only 2 samples

---

## 📈 **Conclusion**

The high performance is **misleading** because:

- The test set is heavily biased toward familiar samples
- Insufficient zero-day samples (2) make metrics unreliable
- Overall accuracy reflects performance on seen samples, not zero-day detection

**After the stratified sampling fix**:

- More zero-day samples will be available
- Metrics will be statistically meaningful
- Performance will likely drop to realistic levels (~50-70% on zero-day attacks)

