# TTT Loss vs Performance Mismatch Analysis

## Problem: Why Loss Decreases But Performance Drops

### Root Cause

The TTT loss is **decreasing** (0.519 → 0.104, -79.98%) but **performance drops** (accuracy: 0.7380 → 0.7289, F1: 0.7507 → 0.6959) because:

### 1. **Self-Reinforcing Incorrect Pseudo-Labels** ⚠️

**Location**: `coordinators/simple_fedavg_coordinator.py:1311`

```python
# Model learns to predict its own predictions!
pseudo_loss = F.cross_entropy(logits[confident_mask], pred_labels[confident_mask])
```

**Problem**:
- The model generates pseudo-labels from its own predictions
- If initial predictions are **wrong** (especially for zero-day attacks), the model **reinforces those errors**
- No validation to check if pseudo-labels are actually correct
- High confidence ≠ correct prediction (model can be confidently wrong)

**Example**:
- Initial prediction: Zero-day attack → "Normal" with 0.92 confidence
- Model treats this as correct pseudo-label
- Model learns: "This zero-day attack IS normal"
- Result: Loss decreases (model becomes confident) but accuracy drops

### 2. **Loss/Metric Mismatch** ⚠️

**Current Loss Components**:
```python
total_loss = (
    pseudo_label_weight * pseudo_loss +      # Encourages confident predictions
    repulsion_weight * repulsion_loss +     # Pushes ambiguous away from normal
    balance_weight * balance_loss            # Balances attack probabilities
)
```

**Problem**:
- **Pseudo-label loss**: Optimizes for confidence, not correctness
- **Repulsion loss**: Optimizes for feature separation, not classification accuracy
- **Balance loss**: Optimizes for probability distribution, not accuracy
- **None of these directly optimize accuracy or F1-score**

**Result**: Model can reduce loss by:
- Making confident predictions (even if wrong) → reduces pseudo_loss
- Pushing ambiguous samples away (even if they're actually normal) → reduces repulsion_loss
- Balancing probabilities (even if predictions are wrong) → reduces balance_loss

### 3. **Overfitting to Adaptation Data** ⚠️

**Problem**:
- Model optimizes on **adaptation query set** (265-266 samples)
- Test set may have **different distribution**
- With **300 steps**, model can overfit to adaptation data
- Adaptation data might not be representative of actual test distribution

**Evidence**:
- Loss decreases smoothly (good convergence on adaptation data)
- But performance on test set drops (poor generalization)

### 4. **Confidence Threshold Issues** ⚠️

**Current Thresholds**:
- Normal anchor: 0.90 (very strict)
- Attack confidence: 0.80 (less strict)

**Problem**:
- High thresholds select only very confident predictions
- But model can be **overconfident on wrong predictions**
- Zero-day attacks might be predicted as "Normal" with high confidence
- These incorrect high-confidence predictions become pseudo-labels

## Solutions

### Solution 1: Add Pseudo-Label Validation (Recommended)

**Add consistency checks** to filter out potentially incorrect pseudo-labels:

```python
# After generating pseudo-labels, check consistency
# 1. Check if prediction is consistent across multiple forward passes (with noise)
# 2. Check if prediction is consistent with feature similarity
# 3. Only use pseudo-labels that pass validation

# Example: Only use pseudo-labels that are consistent across 3 forward passes
consistent_mask = torch.ones_like(confident_mask, dtype=torch.bool)
for _ in range(3):
    noise = torch.randn_like(query_x) * 0.05
    outputs_noisy = adapted_model(query_x + noise)
    preds_noisy = outputs_noisy.argmax(dim=1)
    consistent_mask &= (preds_noisy == pred_labels)

# Only use validated pseudo-labels
validated_confident_mask = confident_mask & consistent_mask
```

### Solution 2: Reduce Adaptation Steps (Quick Fix)

**Current**: 300 steps (may cause overfitting)
**Recommended**: 50-100 steps with early stopping

```python
# Add early stopping based on validation loss
# Stop if loss doesn't improve for N steps
best_loss = float('inf')
patience = 10
no_improve_count = 0

for step in range(num_steps):
    # ... adaptation code ...
    
    if total_loss < best_loss:
        best_loss = total_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            logger.info(f"Early stopping at step {step}")
            break
```

### Solution 3: Add Regularization to Prevent Overfitting

**Add weight decay and dropout** during adaptation:

```python
# Increase weight decay
optimizer = torch.optim.AdamW(
    params,
    lr=lr,
    weight_decay=1e-3  # Increase from 1e-4
)

# Add dropout during adaptation
if hasattr(adapted_model, 'dropout'):
    adapted_model.dropout.p = 0.2  # Add dropout
```

### Solution 4: Use Teacher-Student with EMA (Already Implemented)

**Current**: Teacher model exists but may not be used effectively
**Improvement**: Use teacher predictions more conservatively

```python
# Only use teacher predictions if they're more confident than student
teacher_outputs = self.teacher_model(query_x)
teacher_probs = torch.softmax(teacher_outputs, dim=1)
teacher_conf, teacher_preds = teacher_probs.max(dim=1)

# Only use teacher pseudo-labels if teacher is more confident
teacher_mask = teacher_conf > max_probs
validated_mask = confident_mask & teacher_mask
```

### Solution 5: Lower Confidence Thresholds

**Current**: Normal=0.90, Attack=0.80
**Recommended**: Normal=0.75, Attack=0.65

**Rationale**: Lower thresholds allow more samples to be pseudo-labeled, reducing overfitting to high-confidence (potentially wrong) predictions.

### Solution 6: Add Entropy Regularization

**Add entropy penalty** to prevent overconfident wrong predictions:

```python
# Add entropy regularization to prevent overconfidence
entropy_penalty = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
total_loss = (
    self.pseudo_label_weight * pseudo_loss +
    self.repulsion_weight * repulsion_loss +
    self.balance_weight * balance_loss +
    0.1 * entropy_penalty  # Penalize low entropy (overconfidence)
)
```

## Recommended Immediate Actions

1. **Reduce TTT steps** from 300 to 100-150
2. **Lower confidence thresholds** (Normal: 0.90 → 0.75, Attack: 0.80 → 0.65)
3. **Add early stopping** based on validation loss
4. **Add pseudo-label validation** (consistency checks)
5. **Increase weight decay** to prevent overfitting

## Expected Impact

- **Loss may decrease more slowly** (but that's OK if performance improves)
- **Performance should improve** (accuracy/F1 should increase)
- **Better generalization** to test set
- **More stable adaptation** (less overfitting)

