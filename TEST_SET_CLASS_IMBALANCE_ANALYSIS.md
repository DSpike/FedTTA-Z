# Test Set Class Imbalance Analysis

## 🚨 **YES - Test Set Has Multiple Class Imbalance Issues**

---

## 📊 **Current Test Set Composition (After Sequence Creation)**

### **From Latest Run Logs:**

```
Test sequences: 587 sequences
├── Zero-day sequences: 0/587 (0.0%)  ← EXTREME IMBALANCE
└── Non-zero-day sequences: 587/587 (100.0%)
    ├── Normal (label 0): 508 samples (86.5%)
    └── Attack (label 1): 79 samples (13.5%)  ← IMBALANCE
```

**OR** (from evaluation logs):

```
Test label distribution: tensor([587])
├── Label 0 (Normal): 587 samples (100%)
└── Label 1 (Attack): 0 samples (0%)  ← EXTREME IMBALANCE
```

---

## 🔴 **Multiple Imbalance Issues Identified**

### **Issue 1: Zero-Day vs Non-Zero-Day Imbalance (CRITICAL)**

**Current Status:**

- Zero-day sequences: **0%** (0/587)
- Non-zero-day sequences: **100%** (587/587)

**Severity:** 🔴 **EXTREME** (100:0 ratio)

**Impact:**

- Cannot evaluate zero-day detection
- Optimization never sees zero-day samples
- Model cannot learn zero-day patterns
- Zero-day detection rate always 0% or undefined

---

### **Issue 2: Normal vs Attack Class Imbalance**

**Current Status (Within Non-Zero-Day):**

- Normal (label 0): **508 samples (86.5%)**
- Attack (label 1): **79 samples (13.5%)**

**Severity:** 🔴 **SEVERE** (6.4:1 ratio)

**Impact:**

- Model biased toward predicting Normal
- Low recall on attack detection
- High accuracy but low F1-score on attacks
- False negatives on attack samples

---

### **Issue 3: Binary Classification Imbalance**

**If Only Normal Present:**

- Normal (label 0): **587 samples (100%)**
- Attack (label 1): **0 samples (0%)**

**Severity:** 🔴 **EXTREME** (Binary classification impossible)

**Impact:**

- Cannot calculate ROC/PR curves (need both classes)
- Cannot evaluate attack detection
- Model only learns to predict Normal
- All metrics become meaningless

---

## 📈 **Intended vs Actual Distribution**

### **Intended Composition (Before Sequences):**

```
Test Set (253,749 samples):
├── 30% Normal/BENIGN (~76,125 samples)
└── 70% Attacks (~177,624 samples)
    ├── ~35-50% Zero-day (~89,000-133,500 samples)
    └── ~20-35% Other attacks (~35,000-62,000 samples)
```

**Binary Distribution:**

- Normal: 30%
- Attack: 70%
- **Ratio: 2.3:1 (Acceptable imbalance)**

**Zero-Day Distribution:**

- Zero-day: ~35-50%
- Non-zero-day: ~50-65%
- **Ratio: ~0.7-1.0:1 (Balanced)**

---

### **Actual Composition (After Sequence Creation):**

```
Test Sequences (587 sequences):
├── Zero-day: 0% (0 sequences)  ← LOST DURING SEQUENCE CREATION
└── Non-zero-day: 100% (587 sequences)
    ├── Normal: 86.5% (508 sequences)
    └── Attack: 13.5% (79 sequences)  ← OR 0% if only Normal
```

**Binary Distribution:**

- Normal: 86.5% (or 100%)
- Attack: 13.5% (or 0%)
- **Ratio: 6.4:1 (SEVERE) or ∞:1 (EXTREME)**

**Zero-Day Distribution:**

- Zero-day: 0%
- Non-zero-day: 100%
- **Ratio: 0:1 (EXTREME)**

---

## 🔍 **Root Causes of Imbalance**

### **Root Cause 1: Zero-Day Samples Lost During Sequence Creation**

**Problem:**

- Stratified sampling before sequence creation targets 35% zero-day
- But sequence creation with stride/length mapping loses zero-day samples
- Post-sequence filtering cannot fix what doesn't exist

**Why This Happens:**

1. Simple slicing `[:10000]` doesn't preserve class distribution
2. Sequence creation doesn't maintain multiclass label mapping correctly
3. Zero-day samples might be at the end of test data, not beginning

**From Code (`main.py` lines 1000-1023):**

```python
# Fallback to simple slicing if multiclass labels not available
X_test_subset = self.preprocessed_data['X_test'][:test_subset_size]
y_test_subset = self.preprocessed_data['y_test'][:test_subset_size]
```

This simple slicing loses class distribution!

---

### **Root Cause 2: Normal vs Attack Imbalance After Sampling**

**Problem:**

- Test subset sampling doesn't preserve 30% Normal / 70% Attack ratio
- Sequence creation further distorts distribution
- No rebalancing after sequence creation

**Why This Happens:**

1. Test data might be sorted (all Normal first, then attacks)
2. Simple slicing `[:10000]` takes first samples → mostly Normal
3. Sequence creation doesn't rebalance classes

---

### **Root Cause 3: Post-Sequence Filtering Fails**

**Problem:**

- Code tries to filter to 20% zero-day after sequences
- But if zero-day sequences = 0, filtering cannot create them

**From Code (`main.py` lines 1065-1110):**

```python
target_zero_day_percentage = 0.20
# ...
available_zero_day = len(zero_day_indices)  # = 0!
# Cannot achieve 20% if available_zero_day = 0
```

---

## 🎯 **Impact on Performance**

### **Why Low Overall Performance:**

1. **Zero-Day Imbalance (0% zero-day):**

   - Model never sees zero-day during optimization
   - Cannot learn zero-day patterns
   - Zero-day detection rate = 0% or undefined

2. **Normal vs Attack Imbalance (86.5% vs 13.5%):**

   - Model predicts Normal most of the time
   - Low attack recall
   - High accuracy but low F1-score
   - Many false negatives on attacks

3. **Binary Class Imbalance:**
   - If only Normal present: binary classification impossible
   - Cannot calculate proper metrics
   - Model overfits to Normal class

---

## ✅ **Solutions to Fix Imbalance**

### **Solution 1: Fix Stratified Sampling Before Sequence Creation**

**Change:** Use proper stratified sampling that preserves class distribution

```python
# Instead of: X_test_subset = X_test[:test_subset_size]
# Use stratified sampling:
from sklearn.model_selection import train_test_split
X_test_subset, _, y_test_subset, _, y_multiclass_subset, _ = train_test_split(
    X_test, y_test, y_test_multiclass,
    test_size=1 - (test_subset_size / len(X_test)),
    stratify=y_test_multiclass,  # Preserve multiclass distribution
    random_state=42
)
```

**Benefit:**

- Preserves zero-day percentage before sequences
- Maintains Normal/Attack ratio

---

### **Solution 2: Ensure Zero-Day Samples Are Included**

**Change:** Explicitly sample zero-day samples in subset

```python
# First, ensure zero-day samples are included in subset
zero_day_mask = (y_test_multiclass == zero_day_attack_label)
zero_day_indices = np.where(zero_day_mask)[0]
normal_indices = np.where(y_test == 0)[0]
attack_indices = np.where((y_test == 1) & ~zero_day_mask)[0]

# Sample with desired ratios
n_zero_day = min(int(test_subset_size * 0.20), len(zero_day_indices))
n_normal = int(test_subset_size * 0.30)
n_attack = test_subset_size - n_zero_day - n_normal

selected_indices = (
    np.random.choice(zero_day_indices, n_zero_day, replace=False) +
    np.random.choice(normal_indices, n_normal, replace=False) +
    np.random.choice(attack_indices, n_attack, replace=False)
)
```

**Benefit:**

- Guarantees zero-day samples in subset
- Controls exact distribution ratios

---

### **Solution 3: Rebalance After Sequence Creation**

**Change:** Rebalance sequences after creation to target distribution

**Current Code:**

- Already tries to do this (lines 1065-1110)
- But fails if no zero-day sequences exist

**Fix:**

- Ensure zero-day sequences exist BEFORE filtering
- Use stratified sampling in solution 1 and 2 above

---

### **Solution 4: Use Validation Set for Optimization**

**Alternative Approach:**

- Validation set has no zero-day (by design)
- But has balanced Normal/Attack distribution
- Better for optimizing overall performance

**Trade-off:**

- ✅ Better overall performance
- ✅ Balanced class distribution
- ❌ No zero-day detection optimization

---

## 📊 **Expected Distribution After Fix**

### **Target Distribution (After Sequences):**

```
Test Sequences (target: 587 sequences):
├── Zero-day: ~20% (~117 sequences)
└── Non-zero-day: ~80% (~470 sequences)
    ├── Normal: ~30% (~176 sequences)
    └── Attack: ~50% (~294 sequences)
```

**Binary Distribution:**

- Normal: 30% (176 sequences)
- Attack: 70% (411 sequences = 117 zero-day + 294 other)
- **Ratio: 2.3:1 (Acceptable)**

**Zero-Day Distribution:**

- Zero-day: 20% (117 sequences)
- Non-zero-day: 80% (470 sequences)
- **Ratio: 0.25:1 (Controlled imbalance)**

---

## 🎯 **Recommended Fix Priority**

1. **Priority 1: Fix Stratified Sampling** (Solution 1)

   - Use proper stratified sampling before sequence creation
   - Preserves class distribution

2. **Priority 2: Ensure Zero-Day Inclusion** (Solution 2)

   - Explicitly sample zero-day samples
   - Guarantees zero-day presence

3. **Priority 3: Verify After Sequence Creation** (Solution 3)
   - Check distribution after sequences
   - Apply post-sequence filtering if needed

---

## 📝 **Summary**

**Yes, test set has SEVERE class imbalance issues:**

1. 🔴 **Zero-Day vs Non-Zero-Day: 0% vs 100%** (EXTREME)
2. 🔴 **Normal vs Attack: 86.5% vs 13.5%** (SEVERE)
3. 🔴 **Binary Classification: Possibly 100% vs 0%** (EXTREME)

**Root Cause:**

- Zero-day samples lost during sequence creation
- Stratified sampling not used before sequences
- Simple slicing `[:10000]` doesn't preserve distribution

**Fix:**

- Use stratified sampling before sequence creation
- Explicitly include zero-day samples in subset
- Rebalance after sequence creation









