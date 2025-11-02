# TTT Performance Decrease Analysis

## Actual Performance Metrics from Last Run

### Base Model (Transductive Meta-Learning):

- **Accuracy**: 0.8614 (86.14%)
- **F1-Score**: 0.8945 (89.45%)
- **ROC AUC**: 0.8550 (85.50%)
- **Zero-Day Detection Rate**: 0.8649 (86.49%)

### TTT Adapted Model:

- **Accuracy**: 0.7470 (74.70%) ❌ **-11.44% decrease**
- **F1-Score**: 0.7778 (77.78%) ❌ **-11.67% decrease**
- **ROC AUC**: 0.7749 (77.49%) ❌ **-8.01% decrease**
- **Optimal Threshold**: 0.1 (extreme threshold)
- **Zero-Day Detection Rate**: 0.7838 (78.38%)

## The Problem

**TTT loss decreased significantly** (0.3127 → 0.0226, 93% decrease), but **performance DECREASED** by ~11% across all metrics.

This confirms the **overfitting hypothesis**:

1. TTT is minimizing loss (entropy + diversity) on the query set
2. But this adaptation is hurting generalization to the actual test set
3. The extreme optimal threshold (0.1) suggests the model is poorly calibrated

## Root Cause

From the TTT adaptation logs:

- **Entropy Loss**: 0.3085 → 0.0105 (decreasing - model becoming confident)
- **Diversity Loss**: 0.0416 → 0.1212 (increasing 191% - model losing diversity)
- **Class Entropy**: 0.9640 → 0.8937 (decreasing - less diverse predictions)
- **Max Class Probability**: 0.6112 → 0.6895 (increasing - more concentrated)

**The model is:**

1. Becoming overconfident on the query set
2. Losing class diversity (collapsing towards dominant class)
3. Overfitting to the query set distribution
4. Performing worse on the actual test set

## Proposed Fixes

### 1. Adaptive Diversity Weight (IMPLEMENTED)

- Increases diversity regularization as diversity decreases
- Prevents collapse while allowing entropy minimization

### 2. Early Stopping on Diversity (IMPLEMENTED)

- Stops adaptation if diversity drops below 0.80 threshold
- Prevents overfitting to query set

### 3. Monitor Performance During Adaptation

- Track validation performance during TTT
- Stop if validation performance starts decreasing

### 4. Adjust Learning Rate

- TTT learning rate might be too high (0.00025)
- Consider reducing or using cosine annealing

### 5. Increase Query Set Size

- Current query set: 332 samples
- Larger query set reduces overfitting risk

## Expected Outcome After Fixes

- **Loss decreases** (as currently)
- **Diversity maintains** (class entropy > 0.90)
- **Performance improves** (or at least maintains base model performance)
- **Prevents overfitting** (better generalization)

