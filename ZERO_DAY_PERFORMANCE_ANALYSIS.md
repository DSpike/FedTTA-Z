# Zero-Day Performance Analysis: Why Base and TTT Models Show Same Performance

## 🔍 Problem Statement

Both the base model and TTT model show **identical performance** on zero-day samples only:

- **Base Model**: Zero-Day Detection Rate = 94.59%
- **TTT Model**: Zero-Day Detection Rate = 94.59%
- **Accuracy**: Both show 94.59%
- **Precision**: Both show 100.00%
- **Recall**: Both show 94.59%

## 📊 Root Cause Analysis

### 1. **Detection Rate Calculation**

Both models use the **same formula** for zero-day detection rate:

**Base Model** (line 2763):

```python
zero_day_detection_rate = (zero_day_predictions != 0).float().mean().item()
```

**TTT Model** (line 3505):

```python
zero_day_detection_rate = (zero_day_predictions != 0).float().mean().item()
```

This calculation counts **how many zero-day samples are predicted as "attack" (non-zero)**.

### 2. **Key Issue: TTT Adaptation Strategy**

TTT adapts on the **entire test set** (123 samples):

- 37 zero-day samples (30.1%)
- 86 non-zero-day samples (69.9%)

**Problem**: TTT optimization focuses on **overall test distribution**, not specifically zero-day samples.

### 3. **Why Predictions Might Be Identical**

#### Hypothesis 1: Base Model Already Near-Perfect

- Base model detects 35/37 zero-day samples (94.59%)
- Only 2 samples are misclassified
- TTT might not improve these 2 samples because:
  - They might be true outliers/hard cases
  - Overall optimization doesn't prioritize these rare misclassifications

#### Hypothesis 2: TTT Changes Don't Affect Zero-Day Samples

- TTT might change predictions for **non-zero-day samples** more than zero-day samples
- Zero-day samples might already have high confidence, so TTT doesn't need to change them
- The 2 misclassified samples might remain misclassified

#### Hypothesis 3: Same Threshold/Decision Boundary

- Both models use the same **binary conversion**: `(predictions != 0)`
- If TTT changes probabilities but not the **multiclass class** (still predicts attack class), the binary prediction stays the same
- Zero-day samples might already be correctly classified, so TTT doesn't change them

### 4. **Evidence from Logs**

From the latest run:

```
🔍 DEBUG TTT MODEL - Zero-day predictions: [2, 35]
🔍 DEBUG TTT MODEL - Zero-day actual labels: [0, 37]
```

This means:

- 35 zero-day samples predicted as **attack** (class 1)
- 2 zero-day samples predicted as **normal** (class 0)
- All 37 are actually **attacks** (label 1)

The **detection rate = 35/37 = 94.59%**

### 5. **Why This Happens**

1. **Base Model Performance**: Already very good (94.59% detection)
2. **TTT Optimization**: Optimizes for **overall accuracy**, not zero-day-specific
3. **Class Imbalance**: Zero-day samples are only 30% of test set
4. **Loss Function**: TTT uses entropy minimization + pseudo-labels, which may not specifically target the 2 misclassified zero-day samples

## 🔧 Potential Solutions

### Option 1: **Zero-Day Aware TTT** (Recommended)

Modify TTT to give **higher weight** to zero-day samples during adaptation:

```python
# Weight zero-day samples more heavily
zero_day_weights = torch.ones(len(query_x))
zero_day_weights[zero_day_mask] = 5.0  # 5x weight for zero-day samples

# Use weighted loss
loss = weighted_loss(entropy_loss, zero_day_weights)
```

### Option 2: **Separate TTT for Zero-Day Samples**

Run TTT adaptation **separately** on zero-day samples with different hyperparameters:

```python
# Adapt on zero-day samples with higher learning rate
zero_day_query_x = query_x[zero_day_mask]
adapted_model = ttt_adapt(adapted_model, zero_day_query_x, lr=higher_lr)
```

### Option 3: **Zero-Day Loss Component**

Add explicit zero-day detection loss to TTT:

```python
zero_day_entropy_loss = entropy_loss[zero_day_mask].mean()
total_loss = entropy_loss + 2.0 * zero_day_entropy_loss
```

### Option 4: **Accept Current Performance**

If base model is already at 94.59%, this might be acceptable. TTT still improves **overall performance** (+4.07% accuracy).

## 📈 Current Performance Summary

### Overall Performance (All Test Samples):

- **Base**: Accuracy = 81.30%, F1 = 84.14%
- **TTT**: Accuracy = 85.37% (+4.07%), F1 = 86.36% (+2.22%)
- **✅ TTT improves overall performance**

### Zero-Day Only (37 samples):

- **Base**: ZDR = 94.59%, Accuracy = 94.59%
- **TTT**: ZDR = 94.59%, Accuracy = 94.59%
- **❌ No improvement on zero-day samples**

## 🎯 Conclusion

The identical performance is likely because:

1. **Base model is already excellent** at zero-day detection (94.59%)
2. **TTT optimizes for overall performance**, not zero-day-specific
3. **Only 2 samples are misclassified** - TTT doesn't specifically target these
4. **Zero-day samples are minority** (30% of test set), so overall optimization doesn't prioritize them

**Recommendation**: Implement zero-day-aware TTT weighting if you want to improve zero-day detection specifically.



