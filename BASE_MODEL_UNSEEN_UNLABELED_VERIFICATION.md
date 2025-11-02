# Base Model Unseen & Unlabeled Sample Verification

## Critical Question

**How can the base model outperform TTT model if it's truly tested on unseen and unlabeled samples?**

Let me verify if the base model evaluation truly uses **unseen** and **unlabeled** samples.

---

## Verification 1: Are Samples Truly Unseen?

### ✅ CONFIRMED: Zero-Day Samples Are Unseen

**Evidence from Preprocessing**:

- **Training data**: 193,570 samples
- **Zero-day attack "Analysis" (label=2)**: **EXCLUDED from training**
- **Test data**: 1,517 zero-day samples (Analysis attack) **INCLUDED**

**Code Evidence** (lines 741-746 in `blockchain_federated_unsw_preprocessor.py`):

```python
train_attacks_filtered = train_attacks[train_attacks['attack_cat'] != zero_day_attack]
train_data = pd.concat([train_normal, train_attacks_filtered], ignore_index=True)
```

**Conclusion**: ✅ Zero-day samples are **truly unseen** during training.

---

## Verification 2: Are Samples Truly Unlabeled During Prediction?

### ✅ CONFIRMED: Labels Are NOT Used During Prediction

**Code Evidence** (lines 2199-2204 in `main.py`):

```python
# Evaluate base model performance
with torch.no_grad():  # ✅ No gradient computation (no training)
    global_model.eval()  # ✅ Evaluation mode (no dropout, no batch norm updates)
    base_logits = global_model(X_test_tensor)  # ✅ Model ONLY sees features (X_test_tensor)
    base_predictions = torch.argmax(base_logits, dim=1)  # ✅ Predictions from features only
    base_probabilities = torch.softmax(base_logits, dim=1)
```

**Critical Points**:

1. **`X_test_tensor`** contains ONLY features (no labels)
2. **`y_test_tensor`** (labels) is NOT passed to the model
3. **`torch.no_grad()`** ensures no gradients are computed
4. **`model.eval()`** puts model in evaluation mode

**When Are Labels Used?**

- **After prediction**: Labels are used ONLY for calculating metrics (lines 2206-2290)
- **For zero-day mask**: Labels are used to identify which samples are zero-day (for separate metrics)

**Conclusion**: ✅ Samples are **truly unlabeled** during prediction. Labels are used only **post-prediction** for evaluation.

---

## Verification 3: Zero-Day Mask Creation

**Code Evidence** (lines 2142-2153 in `main.py`):

```python
# Create zero-day mask using multiclass labels
y_test_multiclass_seq = self.preprocessed_data['y_test_multiclass']
zero_day_mask = (y_test_multiclass_seq == zero_day_attack_label)
```

**Important**: The zero-day mask is created using labels, BUT:

- It's used **AFTER** predictions are made
- It's only used to **filter predictions** for calculating zero-day-specific metrics
- The model never sees this mask during prediction

**Conclusion**: ✅ Zero-day mask doesn't leak information to the model.

---

## Why Base Model Outperforms TTT: Analysis

### Performance Comparison (Latest Run)

| Metric       | Base Model | TTT Model | Difference              |
| ------------ | ---------- | --------- | ----------------------- |
| **Accuracy** | 0.8916     | 0.8223    | **+0.0693** (Base wins) |
| **F1-Score** | 0.9227     | 0.8543    | **+0.0684** (Base wins) |
| **ROC AUC**  | 0.8547     | 0.9346    | **-0.0799** (TTT wins)  |

### Possible Explanations

#### 1. **Threshold Selection Issue** ⚠️

**Base Model**:

- Uses **fixed threshold = 0.5** (line 3087)
- Appropriate for "as-is" evaluation

**TTT Model**:

- Uses **optimal threshold** found on test set (line 2599)
- Threshold optimization uses test labels: `find_optimal_threshold(y_test_binary, attack_probs, ...)`
- Found threshold: 0.1000 (extreme, at edge of band)

**Problem**: Optimal threshold is found using **test labels**, which could lead to:

- Overfitting to test distribution
- Suboptimal threshold if optimization fails
- Current threshold (0.1000) is too low, causing accuracy degradation

#### 2. **TTT Over-Adaptation** ⚠️

**From Logs**:

- TTT adapts on 332 sequences (full test set)
- TTT reduces overconfidence (probability mean: 0.7289 → 0.5452)
- But this might be **over-adapting** to the query set

**TTT Loss**:

- Final loss: 0.0115 (very low)
- Entropy minimized: 0.3540 → 0.0104
- Diversity maintained but model might be too specialized

#### 3. **Distribution Mismatch Between Adaptation and Evaluation**

**Current Status**: ✅ Fixed (both use stride=15)

But TTT adapts on:

- Random sample from test (includes zero-day)
- Unlabeled (unsupervised)

**Issue**: TTT might adapt to **mixed distribution** (zero-day + non-zero-day), while zero-day requires different adaptation.

---

## The Paradox Explained

### Why Base Model Can Outperform on Zero-Day?

**Key Insight**: The base model achieves **94.59% accuracy on zero-day attacks** even though:

1. ✅ Zero-day samples are unseen during training
2. ✅ Zero-day samples are unlabeled during prediction

**Possible Reasons**:

1. **Meta-Learning Generalization**:

   - Transductive meta-learning learns to generalize to unseen classes
   - Prototypical networks can classify unseen attack types by learning feature representations
   - The model learns "what makes an attack" rather than specific attack patterns

2. **Zero-Day Samples Are Similar to Seen Attacks**:

   - "Analysis" attack might share features with seen attacks (DoS, Reconnaissance, etc.)
   - The model generalizes attack patterns rather than memorizing specific types
   - Binary classification (Normal vs Attack) is easier than multiclass

3. **Base Model's Robust Decision Boundary**:

   - Fixed threshold (0.5) is more robust
   - Less affected by probability distribution shifts
   - Better calibrated for zero-day samples

4. **TTT Adaptation Hurts Performance**:
   - TTT adapts on mixed distribution (zero-day + non-zero-day)
   - Over-adapts to query set (entropy too low: 0.0104)
   - Optimal threshold (0.1000) is suboptimal for zero-day detection

---

## Conclusion

### ✅ Base Model IS Tested on Unseen & Unlabeled Samples

**Verification Results**:

1. ✅ Zero-day samples excluded from training
2. ✅ Labels NOT used during prediction (only features passed to model)
3. ✅ Labels used only post-prediction for metrics calculation
4. ✅ Zero-day mask doesn't leak information

### Why Base Model Outperforms?

**Most Likely Causes**:

1. **TTT Threshold Issue**: Optimal threshold (0.1000) is too low/extreme
2. **TTT Over-Adaptation**: Model over-adapts to query set, losing generalization
3. **Meta-Learning Strength**: Base model's meta-learning provides strong generalization to unseen attacks
4. **Distribution Mismatch**: TTT adapts to mixed distribution, hurting zero-day performance

**Recommendations**:

1. Fix TTT threshold selection (use calibration instead of optimal threshold on test set)
2. Reduce TTT adaptation intensity (higher entropy target)
3. Adapt TTT separately on zero-day vs non-zero-day samples
4. Use temperature scaling before threshold selection

---

## Final Answer

**The base model IS truly tested on unseen and unlabeled samples.** The higher performance is likely due to:

- Strong meta-learning generalization
- Robust fixed threshold (0.5)
- TTT's suboptimal threshold (0.1000) and over-adaptation

**This is actually realistic** - meta-learning models CAN generalize to unseen classes, especially for binary classification tasks (Normal vs Attack).

