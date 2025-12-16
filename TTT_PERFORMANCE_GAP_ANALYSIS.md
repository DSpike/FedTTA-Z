# TTT Performance Gap Analysis

## Problem Statement

**TTT Total Loss Improvement: 44%** but **Performance Improvement: ~0-1%**

The TTT adaptation shows significant loss reduction (entropy, pseudo-label, consistency losses all decreasing), but this doesn't translate to meaningful performance improvements compared to the base model.

## Root Causes

### 1. Threshold Mismatch ⚠️ **PRIMARY ISSUE**

**During TTT Adaptation:**
- Uses `ClassSpecificThresholds` with:
  - Normal class: 0.85 (high precision)
  - Attack class: 0.70 (lower for recall)
- Optimizes model to work well at these thresholds

**During Evaluation:**
- Uses PR-based threshold optimization (`find_optimal_threshold_pr`)
- Finds single optimal threshold (~0.5-0.6) based on F1-score
- This threshold is different from what TTT optimized for

**Impact:** TTT improves predictions at thresholds (0.85/0.70) that aren't used in evaluation (~0.5). The improvements don't translate because evaluation uses a different threshold.

### 2. Ensemble Voting Dampens Individual Improvements

**Problem:**
- Ensemble uses uncertainty-weighted averaging
- Individual models show 2% prediction change
- Ensemble shows 0.0% prediction change
- Individual improvements cancel out in the ensemble average

**Impact:** Even if individual TTT variants improve significantly, the ensemble average might stay similar.

### 3. Soft vs Hard Prediction Mismatch

**Problem:**
- TTT optimizes **soft predictions** (probabilities), entropy, consistency
- A 44% loss reduction can occur without changing **hard predictions** (argmax)
- Probabilities become more extreme (confident) but don't cross decision boundaries (0.5)

**Example:**
- Before TTT: P(attack) = 0.45 → predicted Normal
- After TTT: P(attack) = 0.48 → still predicted Normal (no change in argmax)
- Loss decreases because entropy decreases, but prediction stays the same

**Impact:** Loss improvements don't translate to accuracy improvements because hard predictions don't change.

### 4. Calibration vs Accuracy

**Problem:**
- TTT might improve **calibration** (confidence matches correctness)
- Without improving **accuracy** (correct predictions)
- Model becomes more confident about the same predictions

**Impact:** Model is better calibrated but makes the same mistakes.

## Recommended Fixes

### Fix 1: Align Threshold Strategy (HIGH PRIORITY)

**Option A: Use PR-Optimized Threshold During TTT Adaptation**
```python
# In TENTPseudoLabels._generate_pseudo_labels():
# Instead of class-specific thresholds, use PR-optimized threshold
optimal_threshold = find_optimal_threshold_pr(...)
confident_mask = confidences > optimal_threshold
```

**Option B: Use Class-Specific Thresholds During Evaluation**
```python
# In _evaluate_adapted_model():
# Apply class-specific thresholds instead of single PR threshold
for class_id in [0, 1]:
    class_threshold = threshold_manager.get_threshold(class_id)
    class_predictions = (probs[:, class_id] >= class_threshold).long()
```

**Recommendation:** Option A is better - use PR-optimized threshold during adaptation so TTT optimizes for what will be used in evaluation.

### Fix 2: Track Soft Prediction Changes

**Current:** Only track hard prediction changes (argmax)
**Fix:** Track soft prediction changes (probability shifts)

```python
# Track probability shifts, not just class changes
base_probs = torch.softmax(base_logits, dim=1)
adapted_probs = torch.softmax(adapted_logits, dim=1)
prob_shift = (adapted_probs - base_probs).abs().mean()
logger.info(f"Average probability shift: {prob_shift:.4f}")
```

### Fix 3: Evaluate Individual Models Separately

**Current:** Only evaluate ensemble average
**Fix:** Evaluate each TTT variant separately and report individual improvements

```python
# In _perform_ensemble_ttt_adaptation():
for name, model in [("Pseudo-label", pseudo_model), ...]:
    metrics = evaluate_model(model, test_data)
    logger.info(f"{name} TTT - Accuracy: {metrics['accuracy']:.4f}")
```

### Fix 4: Optimize TTT for Evaluation Metric

**Current:** TTT optimizes unsupervised losses (entropy, pseudo-label, consistency)
**Fix:** Add evaluation metric terms to TTT loss

```python
# Add F1/ZDR terms to TTT loss
def compute_ttt_loss_with_metrics(predictions, labels, entropy_loss, pseudo_loss):
    # Unsupervised losses
    unsupervised_loss = entropy_loss + pseudo_loss
    
    # If labels available (validation set), add supervised loss
    if labels is not None:
        f1_loss = 1.0 - f1_score(predictions, labels)  # Maximize F1
        total_loss = unsupervised_loss + 0.1 * f1_loss
    else:
        total_loss = unsupervised_loss
    
    return total_loss
```

### Fix 5: Use Best Individual Model Instead of Ensemble

**Current:** Always use ensemble (might cancel improvements)
**Fix:** Select best individual model based on validation performance

```python
# Evaluate each variant on validation set
best_model = None
best_accuracy = -1
for name, model in variants:
    val_accuracy = evaluate(model, val_data)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
logger.info(f"Using best individual model: {best_model_name} (accuracy: {best_accuracy:.4f})")
```

## Implementation Priority

1. **Fix 1 (Threshold Alignment)** - HIGH PRIORITY
   - Most likely to fix the performance gap
   - Ensures TTT optimizes for evaluation conditions

2. **Fix 3 (Individual Model Evaluation)** - MEDIUM PRIORITY
   - Helps diagnose which variants improve
   - Might reveal ensemble is the problem

3. **Fix 4 (Metric-Optimized TTT)** - MEDIUM PRIORITY
   - Directly optimizes for evaluation metrics
   - Requires validation set labels

4. **Fix 2 (Soft Prediction Tracking)** - LOW PRIORITY
   - Diagnostic only, doesn't fix performance

5. **Fix 5 (Best Individual Model)** - LOW PRIORITY
   - Alternative to ensemble, requires validation set

## Expected Impact

After implementing **Fix 1** (Threshold Alignment):
- TTT will optimize for the same threshold used in evaluation
- Expected improvement: +3-5% accuracy, +5-10% ZDR

After implementing **Fix 3** (Individual Model Evaluation):
- Will identify which TTT variants actually improve
- Might reveal that ensemble is canceling improvements

## Conclusion

The **44% loss improvement** is real, but it doesn't translate to performance because:
1. TTT optimizes for different thresholds than evaluation uses
2. Ensemble voting cancels individual improvements
3. Loss improvements are in soft predictions (calibration), not hard predictions (accuracy)

**Primary fix:** Align threshold strategy between adaptation and evaluation.



