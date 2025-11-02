# TTT Loss vs Performance Analysis

## Problem Statement

TTT loss decreased significantly (93%), but performance metrics may have decreased. This investigation analyzes why loss reduction doesn't correlate with performance improvement.

## Root Cause Analysis

### From Latest Run Logs

**Loss Evolution (Step 0 → Step 149):**

- **Total Loss**: 0.3127 → 0.0226 (↓ 93% decrease)
- **Entropy Loss**: 0.3085 → 0.0105 (↓ 97% decrease)
- **Diversity Loss**: 0.0416 → 0.1212 (↑ 191% increase!)

**Diversity Metrics:**

- **Class Entropy**: 0.9640 → 0.8937 (↓ 7.3% - **LESS DIVERSE**)
- **Max Class Probability**: 0.6112 → 0.6895 (↑ 12.8% - **MORE CONCENTRATED**)
- **Unique Classes**: 2/2 → 2/2 (maintained)

**Performance Metrics (Current Run - FROM JSON FILE):**

- Base Model: Accuracy=0.8614, F1=0.8945, ROC AUC=0.8550
- TTT Model: Accuracy=0.7470, F1=0.7778, ROC AUC=0.7749
- **TTT is performing WORSE**: -11.44% accuracy, -11.67% F1, -8.01% ROC AUC

### The Problem: Overfitting to Query Set

**What's Happening:**

1. ✅ Entropy minimization is working → Model becomes more confident on individual samples
2. ❌ Diversity is decreasing → Model predictions are becoming less balanced
3. ❌ Diversity loss is increasing → Model is losing class diversity faster than entropy is decreasing

**Why This Happens:**

- **Entropy Loss** (0.3085 → 0.0105): Minimizing individual sample uncertainty
- **Diversity Loss** (0.0416 → 0.1212): Penalty for low class distribution entropy
- **Diversity Weight** (0.1): Too weak to counteract entropy minimization
- **Net Effect**: Model becomes confident but collapses towards dominant class

**Mathematical Explanation:**

```
Total Loss = Entropy Loss + (0.1 × Diversity Loss)

Step 0:  0.3127 = 0.3085 + (0.1 × 0.0416) = 0.3085 + 0.00416
Step 149: 0.0226 = 0.0105 + (0.1 × 0.1212) = 0.0105 + 0.01212
```

**Problem**: Even though total loss decreased, the diversity component (0.00416 → 0.01212) **increased 191%**, meaning the model is losing diversity.

### Why Performance Might Still Be Good (Currently)

1. **Model hasn't fully collapsed yet** (still predicting both classes)
2. **Zero-day detection improved** (+5.4% from 0.7297 to 0.7838)
3. **ROC AUC improved** (+1.8% from 0.8228 to 0.8407)
4. **But trend is concerning** - if diversity continues to decrease, model will collapse

## Proposed Solutions

### Solution 1: Adaptive Diversity Weight (RECOMMENDED)

**Principle**: Increase diversity weight as diversity decreases

```python
# Adaptive diversity weight based on current diversity
current_diversity = normalized_class_entropy  # [0, 1]
target_diversity = 0.85  # Target normalized entropy

# If diversity is below target, increase weight
if current_diversity < target_diversity:
    diversity_weight = 0.1 + (target_diversity - current_diversity) * 0.5
    # Example: if current_diversity=0.7, weight = 0.1 + 0.15*0.5 = 0.175
else:
    diversity_weight = 0.1  # Default
```

**Expected Impact**: Prevents collapse, maintains diversity

### Solution 2: Early Stopping Based on Diversity

**Principle**: Stop adaptation if diversity drops below threshold

```python
diversity_threshold = 0.85  # Minimum acceptable normalized class entropy

if normalized_class_entropy < diversity_threshold:
    logger.warning("Diversity below threshold - stopping adaptation")
    break
```

**Expected Impact**: Prevents overfitting to query set

### Solution 3: Increase Base Diversity Weight

**Principle**: Simply increase from 0.1 to 0.2-0.3

```python
ttt_diversity_weight: float = 0.2  # Increased from 0.1
```

**Expected Impact**: Stronger diversity regularization, but might slow convergence

### Solution 4: Balanced Loss Function

**Principle**: Use ratio-based loss instead of weighted sum

```python
# Instead of: total_loss = entropy_loss + weight * diversity_loss
# Use: total_loss = entropy_loss * (1 + diversity_loss)
# This ensures diversity has proportional impact
```

**Expected Impact**: More balanced optimization

## Recommended Implementation

**Priority 1: Adaptive Diversity Weight** (prevents collapse)
**Priority 2: Early Stopping** (safety net)
**Priority 3: Monitoring** (track diversity metrics)

## Expected Outcome

After implementing adaptive diversity weight:

- ✅ Loss decreases (as currently)
- ✅ Diversity maintains (class entropy stays > 0.90)
- ✅ Performance improves further (due to better generalization)
- ✅ Prevents model collapse (safety mechanism)
