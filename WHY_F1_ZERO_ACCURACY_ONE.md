# Why F1-Score is Zero While Accuracy is 1.0

## 🔍 The Problem

You're seeing:
- **Accuracy = 1.0** (100% correct predictions)
- **F1-Score = 0.0** (zero F1)

This seems contradictory, but it's actually a **common edge case** in binary classification.

---

## 📊 What's Happening

### **Scenario: Model Predicts Only One Class**

This occurs when:

1. **The model predicts ALL samples as one class** (e.g., all Normal or all Attack)
2. **The test set contains ONLY that class** (e.g., all Normal samples)
3. **All predictions are correct** → Accuracy = 1.0 ✅
4. **But F1 for the other class = 0** → F1-Score = 0.0 ❌

---

## 🧮 Mathematical Explanation

### **Example: All Normal, Model Predicts All Normal**

**Test Set:**
- 100 Normal samples (class 0)
- 0 Attack samples (class 1)

**Model Predictions:**
- 100 Normal predictions (class 0)
- 0 Attack predictions (class 1)

**Confusion Matrix (Binary: Normal=0, Attack=1):**
```
                Predicted
              Normal  Attack
Actual Normal   100      0
       Attack     0      0
```

**Metrics:**
- **TP (True Positives for Attack)** = 0 (no attacks predicted correctly)
- **FP (False Positives for Attack)** = 0 (no false attack predictions)
- **FN (False Negatives for Attack)** = 0 (no attacks to miss)
- **TN (True Negatives for Normal)** = 100 (all normals correct)

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (0 + 100) / (0 + 100 + 0 + 0)
         = 100 / 100
         = 1.0 ✅
```

**F1-Score (for Attack class):**
```
Precision = TP / (TP + FP) = 0 / (0 + 0) = 0/0 → 0 (zero_division=0)
Recall    = TP / (TP + FN) = 0 / (0 + 0) = 0/0 → 0 (zero_division=0)
F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
          = 2 * (0 * 0) / (0 + 0)
          = 0/0 → 0 (zero_division=0) ❌
```

---

## 🎯 Why This Happens in Your System

### **Possible Causes:**

1. **Test Set Imbalance:**
   - Test set has only Normal samples (or only Attack samples)
   - Model correctly predicts all as that class
   - Accuracy = 1.0, but F1 for missing class = 0

2. **Model Overfitting:**
   - Model learned to always predict one class
   - Works perfectly if test set matches training distribution
   - Fails to detect the other class

3. **Zero-Day Exclusion:**
   - If zero-day attacks are excluded from test set
   - And model predicts all as Normal
   - Accuracy = 1.0 (all correct), but F1 for Attack = 0

4. **Confidence-Based Rejection:**
   - All samples rejected (low confidence)
   - Only high-confidence samples remain (all one class)
   - Accuracy = 1.0, but F1 = 0

---

## 🔍 How to Diagnose

### **Check Your Test Set Distribution:**

```python
# Check class distribution in test set
unique, counts = np.unique(y_test, return_counts=True)
print(f"Test set class distribution: {dict(zip(unique, counts))}")
```

**If you see:**
- Only one class → This explains F1=0, Accuracy=1
- Severe imbalance (e.g., 1000 Normal, 1 Attack) → Similar issue

### **Check Model Predictions:**

```python
# Check prediction distribution
unique_pred, counts_pred = np.unique(predictions, return_counts=True)
print(f"Prediction distribution: {dict(zip(unique_pred, counts_pred))}")
```

**If you see:**
- All predictions are one class → Model is not learning properly
- Matches test set distribution → Accuracy=1, but F1=0

### **Check Confusion Matrix:**

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_binary, y_pred_binary)
print("Confusion Matrix:")
print(cm)
```

**If you see:**
```
[[100   0]
 [  0   0]]  # All Normal, no Attacks
```
→ This confirms the issue

---

## ✅ Solutions

### **1. Fix Test Set Distribution**

Ensure your test set has both classes:

```python
# Stratified sampling to ensure both classes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### **2. Check Zero-Day Exclusion**

If you're excluding zero-day attacks, make sure test set still has known attacks:

```python
# Don't exclude all attacks, only zero-day
test_set_attacks = y_test[y_test != 0]  # Should have some attacks
print(f"Number of attacks in test set: {len(test_set_attacks)}")
```

### **3. Adjust Confidence Threshold**

If using confidence-based rejection, lower the threshold:

```python
# Lower confidence threshold to include more samples
confidence_threshold = 0.5  # Instead of 0.7 or 0.9
```

### **4. Use Macro F1 Instead**

For imbalanced data, use macro-averaged F1:

```python
from sklearn.metrics import f1_score
f1_macro = f1_score(y_true, y_pred, average='macro')  # Better for imbalanced data
```

### **5. Check Model Training**

If model always predicts one class, check:
- Class weights in loss function
- Training data distribution
- Model capacity (too simple?)

---

## 📈 Expected Behavior

### **Normal Scenario:**
- Test set: 80 Normal, 20 Attack
- Model predictions: 75 Normal, 25 Attack
- Accuracy: ~0.85-0.95
- F1-Score: ~0.60-0.80

### **Your Current Scenario:**
- Test set: 100 Normal, 0 Attack
- Model predictions: 100 Normal, 0 Attack
- Accuracy: 1.0 ✅
- F1-Score: 0.0 ❌ (no attacks to evaluate)

---

## 🎯 Action Items

1. **Check test set distribution** - Ensure both classes are present
2. **Check prediction distribution** - Model should predict both classes
3. **Review zero-day exclusion logic** - Don't exclude all attacks
4. **Use stratified sampling** - For balanced test sets
5. **Consider macro F1** - Better for imbalanced scenarios

---

## 💡 Key Takeaway

**F1-Score = 0 with Accuracy = 1.0** indicates:
- ✅ Model is making correct predictions
- ❌ But only for one class (the dominant class)
- ⚠️ **This is a data/class imbalance issue, not a model issue**

The model is working correctly, but the test set doesn't have enough diversity to evaluate F1 properly.







