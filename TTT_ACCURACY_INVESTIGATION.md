# TTT Accuracy Investigation

## Problem Summary

**Current Performance:**
- Base Model: Accuracy=0.8916, F1=0.9227, ROC AUC=0.8547
- TTT Model: Accuracy=0.8223, F1=0.8543, ROC AUC=0.9346

**Key Issue:** TTT shows **higher ROC AUC** but **lower accuracy/F1**, suggesting overconfidence or threshold issues.

## Hypothesis: Overconfidence from Entropy Minimization

### Root Cause Analysis

1. **TTT Entropy Minimization** (Line 416-417 in `simple_fedavg_coordinator.py`):
   ```python
   entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
   entropy_loss = entropy.mean()
   ```
   - This encourages **highly confident predictions** (probabilities near 0 or 1)
   - After 150 steps, model becomes very confident → extreme probabilities

2. **Extreme Threshold Selection**:
   - TTT adaptation makes probabilities extreme (e.g., most > 0.9 or < 0.1)
   - Optimal threshold optimization finds threshold at edge (0.0100)
   - Edge threshold causes **over-aggressive predictions** → lower accuracy

3. **Diversity Loss Working but Insufficient**:
   - Diversity loss maintains class balance (entropy 0.96 → 0.99)
   - But doesn't prevent probability extremes (mean confidence can still be very high/low)

### Evidence from Logs

- **Diversity Metrics:** Class entropy increased (✓), but probabilities still become extreme
- **Threshold:** 0.0100 (too low) → suggests probabilities are mostly high (> 0.01)
- **ROC AUC Improved:** Better ranking/calibration, but threshold selection is suboptimal

## Fixes Implemented

### 1. Threshold Band Constraint (✅ DONE)
- Changed from `(0.01, 0.99)` to `(0.1, 0.9)` to prevent extreme thresholds
- Added safety checks: if optimal still extreme, use median probability
- Updated final clamp to `[0.1, 0.9]` range

### 2. Analysis Logging (✅ DONE)
- Added probability distribution analysis for both base and TTT models
- Logs: min/max/mean/std/median probabilities, % with prob > 0.9
- Logs prediction distribution after threshold application

## Next Steps to Investigate

### Option A: Temperature Scaling (Recommended)
Apply temperature scaling to soften probabilities before threshold selection:
```python
temperature = 2.0  # Soften predictions
scaled_probs = torch.softmax(logits / temperature, dim=1)
```

### Option B: Entropy Weight Reduction
Reduce entropy minimization weight to prevent overconfidence:
```python
entropy_weight = 0.5  # Instead of 1.0
combined_loss = entropy_weight * entropy_loss + diversity_weight * diversity_loss
```

### Option C: Early Stopping Based on Validation
Stop TTT adaptation when validation metrics start degrading (overfitting detection).

### Option D: Calibrated Threshold Selection
Use calibration-aware threshold selection (Platt scaling or isotonic regression).

## Expected Results After Threshold Fix

- **Threshold:** Should be in reasonable range (0.1-0.9) instead of 0.01
- **Accuracy:** Should improve if threshold was the main issue
- **Probability Distribution:** Will reveal if overconfidence is the root cause


