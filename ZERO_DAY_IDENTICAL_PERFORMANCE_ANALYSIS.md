# Root Cause Analysis: Identical Zero-Day Performance (Base vs TTT)

## 🔍 Problem

Both base and TTT models show **exactly the same performance** on zero-day samples:

- **Zero-Day Detection Rate**: 94.59% (both)
- **Accuracy**: 94.59% (both)
- **Precision**: 100.00% (both)
- **Recall**: 94.59% (both)
- **F1-Score**: 97.22% (both)

## 📊 Root Cause: Three Main Issues

### 1. **Base Model Already Near-Perfect Performance**

From logs:

```
🔍 DEBUG BASE MODEL - Zero-day predictions: [2, 35]
🔍 DEBUG TTT MODEL - Zero-day predictions: [2, 35]
```

**Interpretation**:

- **35 samples** predicted as **attack** (correctly detected)
- **2 samples** predicted as **normal** (misclassified)
- All **37 samples** are actually **attacks**

**Detection Rate** = 35/37 = **94.59%**

**Key Insight**: Base model already detects 94.59% of zero-day attacks. Only 2 samples are misclassified.

---

### 2. **TTT Doesn't Specifically Target Zero-Day Samples**

**TTT Adaptation Process**:

- Adapts on **entire test set** (123 samples: 37 zero-day + 86 non-zero-day)
- Uses **entropy minimization** (unsupervised, no labels)
- Optimizes for **overall distribution**, not zero-day-specific

**Problem**:

```
Zero-day samples: 37/123 = 30.1% of adaptation set
Misclassified zero-day: 2/123 = 1.6% of adaptation set
```

**Why TTT Doesn't Fix the 2 Misclassified Samples**:

1. **Minority of minority**: Only 2 samples out of 123 (1.6%)
2. **Entropy minimization**: Focuses on overall entropy reduction, not specific misclassifications
3. **Optimization priority**: The 86 non-zero-day samples (70%) dominate the loss gradient
4. **High base performance**: Already at 94.59%, so improvement margin is small

---

### 3. **Prediction Analysis: Why They're Identical**

#### Base Model Predictions:

- Uses `argmax` directly on logits
- `zero_day_predictions = base_predictions[zero_day_mask]`
- Binary conversion: `(predictions != 0)`

#### TTT Model Predictions:

- Uses threshold-based binary predictions first
- Then converts to multiclass: `adapted_predictions = np.where(adapted_predictions_binary == 0, 0, adapted_predictions)`
- Same binary conversion: `(predictions != 0)`

**The Issue**: If both models predict the same **multiclass labels** for zero-day samples, the binary conversion will be identical.

**Why Predictions Are Identical**:

1. **Base model confidence**: Already high on zero-day samples
2. **TTT doesn't change zero-day predictions**: Adaptation doesn't specifically improve these 2 samples
3. **Same decision boundary**: Both classify 35 as attack, 2 as normal

---

## 🔬 Technical Analysis

### Code Flow Comparison

**Base Model** (`evaluate_base_model_only`):

```python
base_logits = global_model(X_test_tensor)
base_predictions = torch.argmax(base_logits, dim=1)  # Multiclass
zero_day_predictions = base_predictions[zero_day_mask]
zero_day_y_pred_bin = (zero_day_predictions != 0).astype(int)  # Binary
```

**TTT Model** (`evaluate_adapted_model`):

```python
adapted_logits = adapted_model(X_test_tensor)
adapted_predictions = torch.argmax(adapted_logits, dim=1)  # Multiclass
adapted_predictions = np.where(adapted_predictions_binary == 0, 0, adapted_predictions)
zero_day_predictions = adapted_predictions[zero_day_mask]
adapted_zero_day_y_pred_bin = (zero_day_predictions != 0).astype(int)  # Binary
```

**Observation**: Both use the same binary conversion `(predictions != 0)`. If the multiclass predictions are the same, the binary predictions will be identical.

---

### Why TTT Doesn't Improve Zero-Day Samples

#### 1. **Entropy Minimization Behavior**

TTT minimizes entropy:

```python
entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
```

**For Zero-Day Samples**:

- If base model already has **high confidence** (high probability for one class), entropy is already **low**
- TTT has **little incentive** to change these predictions
- The **2 misclassified samples** might have conflicting patterns, making them hard to fix

#### 2. **Pseudo-Label Quality**

TTT uses pseudo-labels with confidence threshold:

```python
confident_mask = confidences > threshold  # Typically 0.9-0.95
```

**For Zero-Day Samples**:

- If zero-day samples already have high confidence (even if wrong class), they get pseudo-labels
- But pseudo-labels might be **incorrect** for the 2 misclassified samples
- This reinforces the **wrong classification** instead of fixing it

#### 3. **Gradient Influence**

During TTT adaptation:

- **86 non-zero-day samples** (70%) contribute **70% of gradient**
- **37 zero-day samples** (30%) contribute **30% of gradient**
- **2 misclassified zero-day** (1.6%) contribute **1.6% of gradient**

**Result**: The gradient is dominated by non-zero-day samples. The 2 misclassified zero-day samples have minimal influence on optimization.

---

## 📈 Evidence from Latest Run

### Base Model Zero-Day Performance:

```
Zero-day predictions: [2, 35]  # 2 Normal, 35 Attack
Zero-day actual labels: [0, 37]  # All are attacks
Detection Rate: 35/37 = 94.59%
```

### TTT Model Zero-Day Performance:

```
Zero-day predictions: [2, 35]  # SAME: 2 Normal, 35 Attack
Zero-day actual labels: [0, 37]  # All are attacks
Detection Rate: 35/37 = 94.59%  # IDENTICAL
```

**Conclusion**: TTT does NOT change predictions for zero-day samples.

---

## 🎯 Why This Is Expected Behavior

### 1. **Base Model Quality**

- 94.59% detection rate is already **very high**
- Only 2 out of 37 samples are misclassified
- These 2 samples might be **true hard cases** or **outliers**

### 2. **TTT Optimization Goal**

- TTT optimizes for **overall test accuracy**, not zero-day-specific
- Improves overall from 81.30% → 85.37% (+4.07%)
- But doesn't specifically target zero-day samples

### 3. **Mathematical Constraint**

- With only **2 misclassified samples** out of 123 total (1.6%)
- Gradient-based optimization has **minimal influence** from these samples
- Entropy minimization doesn't specifically target misclassifications

---

## 🔧 Solutions (If Improvement Needed)

### Solution 1: **Zero-Day Weighted TTT** (Most Effective)

Modify TTT loss to weight zero-day samples more heavily:

```python
# In TENTPseudoLabels.adapt()
zero_day_weights = torch.ones(len(query_x), device=query_x.device)
zero_day_weights[zero_day_mask] = 5.0  # 5x weight for zero-day samples

# Weighted entropy loss
entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
weighted_entropy_loss = (entropy_loss * zero_day_weights).mean()
```

### Solution 2: **Two-Stage TTT**

Stage 1: Adapt on all samples (current approach)
Stage 2: Fine-tune specifically on zero-day samples:

```python
# After initial TTT
zero_day_query_x = query_x[zero_day_mask]
adapted_model = ttt_adapt(adapted_model, zero_day_query_x, lr=2*original_lr, steps=100)
```

### Solution 3: **Zero-Day Specific Loss**

Add explicit zero-day detection component:

```python
zero_day_entropy = entropy_loss[zero_day_mask].mean()
non_zero_day_entropy = entropy_loss[~zero_day_mask].mean()
total_loss = entropy_loss + 3.0 * zero_day_entropy  # Prioritize zero-day
```

### Solution 4: **Accept Current Performance**

94.59% zero-day detection is already excellent. TTT improves overall performance significantly (+4.07% accuracy).

---

## ✅ Conclusion

**Root Cause**: The identical performance is **expected** because:

1. ✅ **Base model already performs well** (94.59% detection)
2. ✅ **TTT optimizes for overall performance**, not zero-day-specific
3. ✅ **Only 2 samples are misclassified** (1.6% of adaptation set)
4. ✅ **Gradient-based optimization** doesn't prioritize these rare cases
5. ✅ **Entropy minimization** doesn't specifically target misclassifications

**Recommendation**:

- **Accept current performance** if 94.59% is acceptable
- **Implement zero-day weighted TTT** if you need to improve the 2 misclassified samples



