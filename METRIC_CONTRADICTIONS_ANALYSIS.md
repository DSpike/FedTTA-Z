# Metric Contradictions Analysis

## ⚠️ **YES - There ARE Potential Contradictions Between Metrics**

Different optimization metrics can optimize for conflicting goals, leading to trade-offs.

---

## 🔴 **Identified Contradictions**

### **1. Zero-Day Detection vs Overall Performance**

**Contradiction:**

- Optimizing for **`ttt_zero_day_detection_rate`** may increase **false positives** (FAR)
- Model becomes more sensitive → detects zero-day attacks better
- But also flags more normal samples as attacks (higher false alarm rate)

**Example Scenario:**

```python
# Optimizing for zero-day detection only:
ttt_zero_day_detection_rate = 1.0  # Perfect! ✅
# But:
ttt_far = 0.85  # Very high false alarm rate! ⚠️
ttt_f1_score = 0.45  # Poor overall performance! ⚠️
```

**Why this happens:**

- To maximize zero-day detection, model becomes more aggressive
- Lowers threshold → catches zero-day attacks → also catches more normal samples
- **Trade-off**: High zero-day detection ↔ High false alarms

---

### **2. Base Model vs TTT Performance**

**Contradiction:**

- Optimizing **only for TTT metrics** (`ttt_zero_day_detection_rate`, `ttt_f1_score`) may make the **base model weak**
- Hyperparameters that make TTT effective might not be optimal for base model
- Result: Base model performs poorly, but TTT compensates

**Example Scenario:**

```python
# Optimizing for TTT only:
base_f1_score = 0.35  # Poor base model! ⚠️
ttt_f1_score = 0.75   # Good TTT! ✅

# vs. Balanced optimization:
base_f1_score = 0.60  # Good base model! ✅
ttt_f1_score = 0.68   # Good TTT! ✅
```

**Why this happens:**

- Base model needs strong few-shot learning capability
- TTT needs ability to adapt quickly
- These may require different hyperparameters (e.g., learning rate, epochs)

**This is why `balanced_base_ttt` exists!** ✅

---

### **3. Accuracy vs Zero-Day Detection (Imbalanced Data)**

**Contradiction:**

- Optimizing for **`ttt_accuracy`** can be misleading with imbalanced data
- If 90% of test set is Normal, optimizing accuracy → predict "Normal" for everything
- **High accuracy** (90%) but **zero-day detection = 0%** ❌

**Example Scenario:**

```python
# Optimizing for accuracy with 90% Normal samples:
ttt_accuracy = 0.92        # High accuracy! ✅
ttt_zero_day_detection_rate = 0.0  # Zero detection! ❌
# Model just predicts "Normal" for everything
```

**Why this happens:**

- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- With 90% Normal samples, always predicting Normal gives 90% accuracy
- But misses ALL zero-day attacks

**This is why `ttt_auc_pr` or `ttt_f1_score` are better for imbalanced data!**

---

### **4. Zero-Day Detection vs Non-Zero-Day Performance**

**Contradiction:**

- Optimizing for **zero-day detection** may hurt **known attack detection**
- Model becomes too specialized for zero-day patterns
- May misclassify known attacks

**Example Scenario:**

```python
# Optimizing for zero-day only:
ttt_zero_day_detection_rate = 1.0      # Perfect! ✅
ttt_non_zero_day_f1 = 0.40             # Poor! ⚠️
# Model is great at zero-day but bad at known attacks
```

**Why this happens:**

- Zero-day and known attacks have different patterns
- Optimizing hyperparameters for zero-day may not help known attacks
- Model becomes too specialized

**This is why `multi_objective` balances both!**

---

## 📊 **Metric Comparison Table**

| Metric                            | Zero-Day   | Overall  | Base Model | Known Attacks | False Alarms | Contradiction Risk                       |
| --------------------------------- | ---------- | -------- | ---------- | ------------- | ------------ | ---------------------------------------- |
| **`ttt_zero_day_detection_rate`** | ⭐⭐⭐⭐⭐ | ⭐⭐     | ❌ Ignored | ⭐⭐          | ⚠️ High risk | **HIGH** - May increase FAR              |
| **`ttt_f1_score`**                | ⭐⭐⭐     | ⭐⭐⭐⭐ | ❌ Ignored | ⭐⭐⭐⭐      | ⭐⭐⭐       | **MODERATE** - Balanced but ignores base |
| **`ttt_accuracy`**                | ⭐⭐       | ⭐⭐⭐   | ❌ Ignored | ⭐⭐⭐        | ⭐⭐⭐⭐     | **HIGH** - Misleading with imbalance     |
| **`ttt_auc_pr`**                  | ⭐⭐⭐⭐   | ⭐⭐⭐⭐ | ❌ Ignored | ⭐⭐⭐⭐      | ⭐⭐⭐       | **LOW** - Good for imbalance             |
| **`balanced_base_ttt`**           | ⭐⭐⭐⭐   | ⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐      | ⭐⭐⭐       | **LOW** - Most balanced ✅               |
| **`multi_objective`**             | ⭐⭐⭐⭐   | ⭐⭐⭐⭐ | ❌ Ignored | ⭐⭐⭐⭐      | ⭐⭐⭐       | **LOW** - Good balance (TTT-only)        |

---

## 🎯 **Recommended Metrics (Ranked by Balance)**

### **1. `balanced_base_ttt` ⭐ RECOMMENDED**

**Formula:**

```
0.40 × base_f1 + 0.30 × ttt_zdr + 0.30 × ttt_f1
```

**Pros:**

- ✅ Optimizes BOTH base model AND TTT
- ✅ Balanced zero-day and overall performance
- ✅ Fair comparison (base model not artificially weak)
- ✅ Best for comprehensive evaluation

**Contradictions:**

- ⚠️ **MINOR**: Might slightly favor base model (40% weight)
- ⚠️ **MINOR**: Slight trade-off between base and TTT

**When to use:** **Default choice** - Best for fair, comprehensive optimization

---

### **2. `multi_objective` ⭐ GOOD FOR TTT-ONLY**

**Formula:**

```
0.30 × zero_day_zdr + 0.35 × non_zero_day_f1 + 0.35 × overall_f1
```

**Pros:**

- ✅ Balances zero-day and known attacks
- ✅ Good overall TTT performance
- ✅ Reduces false alarm risk

**Contradictions:**

- ⚠️ **MODERATE**: Ignores base model completely
- ⚠️ May find configs where base model is weak but TTT compensates

**When to use:** When you only care about TTT performance (not base model)

---

### **3. `ttt_auc_pr` ⭐ GOOD FOR IMBALANCED DATA**

**Pros:**

- ✅ Best metric for imbalanced datasets (your case)
- ✅ Handles false positives well
- ✅ Standard in IDS research

**Contradictions:**

- ⚠️ **MODERATE**: May not prioritize zero-day detection enough
- ⚠️ Ignores base model

**When to use:** When data is imbalanced and you want balanced precision/recall

---

### **4. `ttt_zero_day_detection_rate` ⚠️ HIGH RISK**

**Pros:**

- ✅ Maximizes zero-day detection
- ✅ Good for research focused on zero-day

**Contradictions:**

- 🔴 **HIGH**: May dramatically increase false alarms
- 🔴 May hurt overall performance
- 🔴 Ignores base model

**When to use:** Only if zero-day detection is THE ONLY priority (research-only)

---

### **5. `ttt_accuracy` ⚠️ NOT RECOMMENDED**

**Pros:**

- ✅ Simple to understand

**Contradictions:**

- 🔴 **VERY HIGH**: Misleading with imbalanced data
- 🔴 Can optimize for "always predict Normal"
- 🔴 Ignores zero-day completely

**When to use:** Never with imbalanced data (your case)

---

## 🔍 **Specific Contradiction Scenarios**

### **Scenario 1: Zero-Day Only Optimization**

```python
metric = "ttt_zero_day_detection_rate"
# Possible result:
ttt_zero_day_detection_rate = 0.95  # Excellent! ✅
ttt_far = 0.80                       # Very high! ⚠️
base_f1 = 0.30                       # Poor! ⚠️
ttt_f1 = 0.45                        # Poor! ⚠️
```

**Trade-off:** High zero-day detection ↔ High false alarms + Poor base model

---

### **Scenario 2: TTT-Only Optimization**

```python
metric = "ttt_f1_score"  # or any TTT-only metric
# Possible result:
base_f1 = 0.35           # Weak base model! ⚠️
ttt_f1 = 0.75            # Strong TTT! ✅
# Makes base model look artificially weak
```

**Trade-off:** Strong TTT ↔ Weak base model (unfair comparison)

---

### **Scenario 3: Accuracy Optimization (Imbalanced)**

```python
metric = "ttt_accuracy"
# Test set: 90% Normal, 10% Attacks
# Possible result:
ttt_accuracy = 0.92              # High! ✅
ttt_zero_day_detection_rate = 0.0  # Zero! ❌
# Model just predicts "Normal" for everything
```

**Trade-off:** High accuracy ↔ Zero zero-day detection (model becomes useless)

---

### **Scenario 4: Balanced Optimization** ✅

```python
metric = "balanced_base_ttt"
# Possible result:
base_f1 = 0.60              # Good! ✅
ttt_zero_day_detection_rate = 0.85  # Good! ✅
ttt_f1 = 0.70               # Good! ✅
ttt_far = 0.35              # Acceptable! ✅
```

**Trade-off:** Balanced - All metrics reasonable (no extreme contradictions)

---

## 💡 **Recommendations**

### **For Your Use Case (Zero-Day Detection IDS):**

**Option 1: `balanced_base_ttt` (RECOMMENDED)** ⭐

- ✅ Optimizes base model AND TTT
- ✅ Balanced zero-day and overall performance
- ✅ Fair comparison (best for paper)
- ✅ Low contradiction risk

**Option 2: `multi_objective`**

- ✅ Good TTT balance
- ✅ Balances zero-day and known attacks
- ⚠️ Ignores base model

**Option 3: `ttt_auc_pr`**

- ✅ Best for imbalanced data
- ✅ Good precision/recall balance
- ⚠️ May not prioritize zero-day enough

---

## ⚠️ **Metrics to AVOID**

1. **`ttt_accuracy`** - Misleading with imbalanced data
2. **`ttt_zero_day_detection_rate`** alone - Too aggressive, high false alarms
3. **TTT-only metrics** without base model consideration - Unfair comparison

---

## ✅ **Solution: Use `balanced_base_ttt`**

This metric is specifically designed to **minimize contradictions**:

```
balanced_score = 0.40 × base_f1 + 0.30 × ttt_zdr + 0.30 × ttt_f1
```

**Why it's better:**

- ✅ Forces optimization to consider base model (40% weight)
- ✅ Still prioritizes zero-day detection (30% weight)
- ✅ Maintains overall TTT performance (30% weight)
- ✅ **Reduces contradictions** by balancing all aspects

---

## 🔬 **How to Detect Contradictions**

After optimization, check these in the results:

1. **High zero-day detection + High FAR:**

   ```python
   if ttt_zdr > 0.9 and ttt_far > 0.6:
       print("⚠️ Contradiction: High zero-day but high false alarms")
   ```

2. **Good TTT + Poor base:**

   ```python
   if ttt_f1 > 0.7 and base_f1 < 0.4:
       print("⚠️ Contradiction: TTT compensates for weak base model")
   ```

3. **High accuracy + Zero zero-day:**
   ```python
   if ttt_accuracy > 0.85 and ttt_zdr < 0.1:
       print("⚠️ Contradiction: Model just predicts Normal")
   ```

---

## 📋 **Summary**

| Metric                            | Contradiction Risk | Recommendation                          |
| --------------------------------- | ------------------ | --------------------------------------- |
| **`balanced_base_ttt`**           | **LOW**            | ✅ **USE THIS** (Default)               |
| **`multi_objective`**             | **LOW**            | ✅ Good alternative (TTT-only)          |
| **`ttt_auc_pr`**                  | **LOW**            | ✅ Good for imbalanced data             |
| **`ttt_f1_score`**                | **MODERATE**       | ⚠️ Use if TTT-only acceptable           |
| **`ttt_zero_day_detection_rate`** | **HIGH**           | ⚠️ Research-only, high false alarm risk |
| **`ttt_accuracy`**                | **VERY HIGH**      | ❌ **AVOID** with imbalanced data       |

**Bottom Line:** Use `balanced_base_ttt` (default) to minimize contradictions and get fair, comprehensive optimization! ✅
