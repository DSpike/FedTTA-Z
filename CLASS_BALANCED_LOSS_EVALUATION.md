# Class-Balanced Loss Evaluation Report

## Implementation Summary

**Date:** 2025-11-12  
**Change:** Added class-balanced entropy loss weighting to TTT adaptation

### What Was Changed

1. **Pure TENT Path** (`_perform_advanced_ttt_adaptation`):
   - Added inverse frequency weighting for entropy loss
   - Minority class (Attack) receives higher weight
   - Location: `coordinators/simple_fedavg_coordinator.py` lines 506-520

2. **TENT + Pseudo-Labels Path** (`TENTPseudoLabels.adapt`):
   - Added same class-balanced weighting to entropy minimization
   - Location: `coordinators/simple_fedavg_coordinator.py` lines 1121-1133

### How It Works

```python
# Compute class distribution from predictions
class_distribution = probs.mean(dim=0)

# Calculate inverse frequency weights
class_weights = 1.0 / (class_distribution + 1e-8)
class_weights = class_weights / class_weights.sum() * len(class_weights)

# Apply weights based on predicted class
predicted_classes = torch.argmax(probs, dim=1)
sample_weights = class_weights[predicted_classes]
weighted_entropy = entropy * sample_weights
entropy_loss = weighted_entropy.mean()
```

**Example:**
- If model predicts Normal=80%, Attack=20%:
  - Normal weight = 1/0.8 = 1.25
  - Attack weight = 1/0.2 = 5.0
- Attack predictions get 4x higher weight in loss

## Expected Effects

### Theoretical Benefits

1. **Better Minority Class Handling:**
   - Attack class (minority) gets higher weight
   - Model focuses more on correctly classifying attacks
   - Reduces bias toward majority Normal class

2. **Improved Zero-Day Detection:**
   - Zero-day attacks are minority samples
   - Higher weight encourages better adaptation to attack patterns
   - Should improve zero-day detection rate

3. **Balanced Precision/Recall:**
   - Prevents model from collapsing to "always predict Normal"
   - Encourages balanced predictions across classes

### Potential Risks

1. **Over-weighting:**
   - If Attack class is very rare, weights could be too high
   - Might cause instability in training

2. **Dynamic Weights:**
   - Weights change every batch based on predictions
   - Could cause oscillations if predictions are unstable

## Evaluation Metrics to Monitor

1. **Zero-Day Detection Rate (ZDR):**
   - Primary metric for minority class performance
   - Expected: Increase from ~56% to >60%

2. **Overall Accuracy:**
   - Should maintain or improve current ~87%
   - Monitor for any degradation

3. **F1-Score:**
   - Macro F1 should improve (better minority class recall)
   - Weighted F1 should remain stable

4. **Class Balance:**
   - Check confusion matrix for balanced predictions
   - Monitor false positive/negative rates

## Comparison: Before vs After

### Before Class-Balanced Loss (Previous Run - with pseudo-labeling)
- Base Model: 67.17% ± 5.51%
- TTT Model: 86.74% ± 4.23%
- Zero-Day Detection Rate: 56.76% → 86.49% (+29.73%)
- **Note:** Class-balanced loss was NOT active in this run

### After Class-Balanced Loss (Implementation Complete)
- **Status:** ✅ Implemented in both TTT paths
- **Expected Impact:**
  - Improved zero-day detection (minority class focus)
  - More balanced precision/recall
  - Better handling of class imbalance during adaptation

### Key Implementation Details

**Both paths now use class-balanced entropy loss:**
1. Pure TENT: Lines 506-520
2. TENT+Pseudo: Lines 1121-1133

**Weight Calculation:**
- Computes class distribution from batch predictions
- Applies inverse frequency weighting
- Attack class (minority) gets 4-5x higher weight than Normal

## Next Steps

1. ✅ Wait for current run to complete
2. Compare metrics:
   - Zero-day detection rate
   - Overall accuracy
   - F1-scores (macro vs weighted)
   - Confusion matrix patterns
3. Analyze:
   - Is minority class performance improved?
   - Are predictions more balanced?
   - Any training instability?

## Code Locations

- Pure TENT: `coordinators/simple_fedavg_coordinator.py:506-520`
- TENT+Pseudo: `coordinators/simple_fedavg_coordinator.py:1121-1133`

