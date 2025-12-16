# Zero-Day Performance vs Overall Performance Discrepancy Analysis

## 🎯 **Your Observation**

You're seeing that:

- **TTT performance on zero-day-only samples**: Higher (e.g., 82%)
- **TTT performance on full test set**: Lower (e.g., 77%)
- **Base model performance on full test set**: Even lower (e.g., 60%)

This seems **counterintuitive** because zero-day attacks are supposed to be harder (unseen), so you'd expect lower performance on them, not higher!

---

## 🔍 **Root Cause Analysis**

### **1. Different Evaluation Contexts (CRITICAL)**

The two evaluations use the **same TTT-adapted model**, but in **different contexts**:

#### **Full Test Set Evaluation** (`performance_comparison_annotated`):

```
Test Set Composition:
├── 40% Normal samples
├── 35% Non-zero-day attacks (seen during training)
└── 25% Zero-day attacks (unseen)

Evaluation Process:
1. TTT adapts on FULL query set (Normal + Attacks mixed)
2. Prototypes computed from validation data (NO zero-day)
3. Evaluates on ALL test samples (mixed classes)
```

#### **Zero-Day Only Evaluation** (`zero_day_performance_comparison`):

```
Test Set Composition:
└── 100% Zero-day attacks (unseen)

Evaluation Process:
1. Uses SAME TTT-adapted model (already adapted)
2. Prototypes computed from validation data (NO zero-day) - SAME as full test
3. Evaluates ONLY on zero-day samples (filtered subset)
```

**Key Insight**: The **same adapted model** is used, but evaluated on **different sample distributions**.

---

### **2. Prototype Mismatch Effect**

**Problem**: Prototypes are computed from **validation data** (which has NO zero-day attacks):

```python
# Line 2940-2951 in main.py
X_val_tensor = torch.FloatTensor(self.preprocessed_data['X_val']).to(self.device)
support_x = X_val_tensor[support_indices]  # Random 200 samples from validation
support_y = y_val_binary[support_indices]  # Normal + known attacks only

prototypes, unique_labels = global_model.compute_prototypes(support_x, support_y)
```

**Impact**:

- ✅ **Zero-day samples**: Model learned to detect "outliers" via TTT → Works well with outlier-aware prototypes
- ⚠️ **Normal samples**: May be misclassified because prototypes are from validation data (different distribution)
- ⚠️ **Non-zero-day attacks**: May be misclassified if TTT adapted too much toward zero-day patterns

---

### **3. TTT Adaptation Bias Toward Outliers**

**How TTT Works** (Lines 5112-5122):

```python
# TTT adapts using entropy minimization + pseudo-labeling
adapted_model = self.coordinator.adapt_to_test_data(
    query_x=query_x,  # FULL test set (Normal + Attacks + Zero-day)
    query_y=None,     # Unsupervised adaptation
    method='tent_pseudo'
)
```

**What Happens**:

1. TTT sees **mixed query set** (Normal + Attacks + Zero-day)
2. Entropy minimization pushes the model to be **confident on in-distribution samples**
3. **Zero-day attacks are outliers** → Model learns to distinguish them from known patterns
4. This helps zero-day detection, but may **hurt normal sample classification**

**Result**:

- ✅ TTT becomes **expert at detecting zero-day** (outlier detection)
- ⚠️ TTT may **overfit to zero-day patterns**, causing:
  - Normal samples misclassified as attacks (false positives)
  - Non-zero-day attacks confused with zero-day patterns

---

### **4. Class Imbalance in Full Test Set**

**Full Test Set Composition**:

```
40% Normal samples        ← Model may struggle (if TTT adapted too aggressively)
35% Non-zero-day attacks  ← Should be easy (seen during training)
25% Zero-day attacks      ← TTT adapted specifically for these
```

**Why Overall Performance is Lower**:

| Sample Type              | Expected Performance        | Why Lower in Full Test?                      |
| ------------------------ | --------------------------- | -------------------------------------------- |
| **Normal**               | Should be high (95%+)       | ⚠️ TTT overfitting may cause false positives |
| **Non-zero-day Attacks** | Should be high (90%+)       | ⚠️ Prototype mismatch or TTT bias            |
| **Zero-day Attacks**     | Should be moderate (70-85%) | ✅ TTT adapted specifically for these        |

**When you filter to zero-day only**:

- ✅ Removes Normal samples (which TTT struggles with)
- ✅ Removes Non-zero-day attacks (which may be confused)
- ✅ Only evaluates on samples TTT adapted for → **Higher performance**

---

### **5. Different Support Sets (Actually Same, But Context Differs)**

**Both evaluations use the SAME prototypes from validation data**, but:

**Full Test Set**:

- Prototypes represent: Normal + Known Attacks
- Query samples: Normal + Known Attacks + Zero-day
- **Mismatch**: Zero-day doesn't fit prototypes well, but TTT compensates
- **Problem**: Normal samples may be pushed toward attack prototypes

**Zero-Day Only**:

- Prototypes represent: Normal + Known Attacks (same)
- Query samples: Only Zero-day (filtered)
- **Match**: TTT adapted specifically for zero-day outliers
- **Advantage**: No Normal samples to misclassify

---

## 🎯 **Why This Is Actually EXPECTED Behavior**

### **1. TTT Is Optimized for Outlier Detection**

TTT (Test-Time Training) is designed to:

- ✅ Adapt to **distribution shifts** (e.g., zero-day attacks)
- ✅ Minimize **entropy** on test data
- ✅ Handle **outliers** better than base model

**Result**: TTT performs better on **hard samples** (zero-day) than **easy samples** (normal traffic).

---

### **2. Prototype-Based Evaluation Penalizes Normal Samples**

**Prototype-Based Classification**:

```python
# Distance-based classification
distances = torch.cdist(query_embeddings, prototypes, p=2)
predictions = unique_labels[torch.argmin(distances, dim=1)]
```

**Problem**:

- Normal samples should cluster around Normal prototype
- If TTT pushes Normal samples away (due to entropy minimization), they may:
  - Get closer to Attack prototypes
  - Be misclassified as attacks

**Zero-Day Samples**:

- Already outliers → Don't cluster with Normal or known Attack prototypes
- TTT helps identify them as "something different"
- Less affected by prototype mismatch

---

### **3. TTT Adaptation May Cause Overfitting**

**What TTT Does**:

1. Adapts model parameters on test data
2. Minimizes entropy (increases confidence)
3. Uses pseudo-labeling

**Risk**:

- ✅ Helps with zero-day (outlier detection)
- ⚠️ May overfit to test distribution
- ⚠️ May hurt performance on normal samples (not represented in adaptation)

---

## 📊 **Expected Performance Patterns**

### **Scenario 1: Healthy TTT Adaptation**

```
Full Test Set:
├── Normal: 85-90% (slight drop from base 95%)
├── Non-zero-day Attacks: 88-92% (good)
└── Zero-day Attacks: 78-85% (good improvement from base 60%)

Overall: 82-87% (good balance)

Zero-Day Only:
└── Zero-day: 78-85% (same as full test set)
```

### **Scenario 2: TTT Overfitting (Your Case)**

```
Full Test Set:
├── Normal: 70-75% (dropped from base 85%) ⚠️
├── Non-zero-day Attacks: 75-80% (dropped from base 90%) ⚠️
└── Zero-day Attacks: 80-85% (good) ✅

Overall: 75-78% (lower due to Normal/non-zero-day drop)

Zero-Day Only:
└── Zero-day: 80-85% (higher because Normal/non-zero-day removed)
```

**This matches your observation!**

---

## ✅ **Solutions & Recommendations**

### **1. Check Normal Sample Performance**

Add separate metrics for Normal vs Attack in full test set:

```python
# In evaluate_adapted_model(), add:
normal_mask = (y_test_binary == 0)
normal_accuracy = (adapted_predictions[normal_mask] == y_test_binary[normal_mask]).float().mean()
normal_fp_rate = (adapted_predictions[normal_mask] == 1).float().mean()  # False positive rate

logger.info(f"Normal sample accuracy: {normal_accuracy:.2%}")
logger.info(f"Normal sample false positive rate: {normal_fp_rate:.2%}")
```

**If Normal accuracy is low** → TTT is overfitting to zero-day.

---

### **2. Reduce TTT Adaptation Intensity**

**Current TTT Config** (check `config.py`):

```python
ttt_base_steps=258  # Number of adaptation steps
ttt_lr=0.0001518747922672249  # Learning rate
entropy_weight=0.6705241236872915  # Entropy minimization weight
```

**Recommendation**:

- Reduce `ttt_base_steps` (e.g., 100-150 instead of 258)
- Reduce `entropy_weight` (e.g., 0.3-0.5 instead of 0.67)
- This will make TTT **less aggressive**, preserving Normal sample performance

---

### **3. Use Separate Prototypes for Zero-Day Detection**

**Idea**: Use **different prototypes** for different sample types:

```python
# Normal samples: Use Normal prototype (from validation)
# Attack samples: Use Attack prototype (from validation)
# Zero-day samples: Use outlier detection (distance threshold)

if confidence < threshold:
    # Low confidence = likely zero-day
    prediction = detect_as_zero_day(embedding, prototypes)
else:
    # High confidence = use prototype-based classification
    prediction = nearest_prototype(embedding, prototypes)
```

---

### **4. Balanced TTT Adaptation**

**Current**: TTT adapts on **full query set** (mixed Normal + Attacks)

**Alternative**: Adapt **separately** on:

- Normal samples (preserve Normal detection)
- Attack samples (improve attack detection)

```python
# Separate adaptation
normal_adapted_model = adapt_to_subset(query_x[normal_mask], ...)
attack_adapted_model = adapt_to_subset(query_x[attack_mask], ...)
```

---

### **5. Monitor False Positive Rate**

**Key Metric**: False Positive Rate (FAR) on Normal samples

```python
# Normal samples incorrectly classified as attacks
normal_fp = (predictions[normal_mask] == 1).sum()
normal_tn = (predictions[normal_mask] == 0).sum()
far = normal_fp / (normal_fp + normal_tn)
```

**If FAR > 5%** → TTT is causing too many false alarms on Normal traffic.

---

## 📝 **How to Present This in Your Paper**

### **✅ Acceptable Explanation**:

```
"The TTT-adapted model achieves 82% detection rate on zero-day attacks,
compared to 77% overall accuracy on the full test set. This discrepancy
arises because TTT adapts specifically to outlier patterns (zero-day attacks),
which may reduce performance on in-distribution samples (normal traffic).
When evaluated on zero-day samples only, the performance reflects the model's
specialized capability for novel attack detection. The overall performance
includes both normal traffic and known attacks, where the model must balance
between zero-day detection and false positive reduction."
```

### **⚠️ Avoid This**:

```
"Our model achieves 82% accuracy on zero-day attacks, demonstrating
excellent performance."
```

**Why**: Doesn't explain the discrepancy with overall performance.

---

## 🎯 **Bottom Line**

**Your observation is EXPECTED and EXPLAINABLE**:

1. ✅ **TTT is optimized for outlier detection** (zero-day attacks)
2. ✅ **Prototype mismatch** favors zero-day detection over Normal classification
3. ✅ **Filtering to zero-day only** removes samples TTT struggles with (Normal)
4. ✅ **Full test set performance** reflects the trade-off: Better zero-day detection at cost of Normal accuracy

**This is NOT a bug** - it's a **characteristic of TTT adaptation** when dealing with imbalanced, outlier-heavy test sets.

**Recommendation**: Report both metrics (overall + zero-day-only) with clear explanation of the trade-off.








