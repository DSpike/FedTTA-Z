# Anchor Issue Diagnosis - Is It Fixed?

## TL;DR: **NO, THE ANCHOR ISSUE IS NOT FIXED**

Despite TTT running successfully, the anchor/prototype updates are making predictions **worse**, not better.

## Current Status

### What's Working ✅
1. TTT runs all 200 steps without errors
2. L2_Reg = 0.0 (disabled)
3. BatchNorm parameters update (896 parameters)
4. Prototypes update every 10 steps via k-means
5. Prototypes are stored and used for evaluation
6. Loss decreases: 0.0084 → 0.0050
7. **Some predictions DO change** (3/10 in first 10 samples)

### What's NOT Working ❌
1. **Base Model: 81.75%**
2. **TTT Model: 21.01%** ← Catastrophic degradation!
3. Changes make predictions **worse**, not better
4. On a subset of 100 samples, predictions are 100% identical (but this is coincidental)

## The Real Problem

**The anchors ARE moving, but they're moving in the WRONG direction!**

### Evidence

**Prediction Changes**:
```
Base model  (first 10): [225, 225, 225, 225, 225, 180, 225, 225, 225, 225]
Adapted     (first 10): [225, 225, 171, 225, 171, 225, 225, 225, 225, 225]
                                   ↑         ↑
                              Changed!  Changed!
```

**Performance**:
```
Base:  81.75% ← Correct predictions
TTT:   21.01% ← Mostly WRONG predictions
```

**Conclusion**: TTT is changing predictions, but changing correct predictions to INCORRECT ones!

## Why Are Anchors Moving Wrong?

There are several possible reasons:

### 1. **K-means on Noisy Embeddings**

During TTT, you use k-means clustering to generate pseudo-labels:

```python
# Step 1: Get embeddings
embeddings = model(support_set)  # [100, 256]

# Step 2: Cluster into 2 groups (Normal vs Attack)
kmeans = KMeans(n_clusters=2)
pseudo_labels = kmeans.fit_predict(embeddings)  # [0, 0, 1, 1, 0, ...]

# Step 3: Compute anchors from pseudo-labeled data
normal_anchor = mean(embeddings[pseudo_labels == 0])
attack_anchor = mean(embeddings[pseudo_labels == 1])
```

**The Problem**: If k-means assigns labels incorrectly, the anchors move to wrong positions!

**Example**:
```
True labels:  [N, N, N, N, A, A, A, A]  ← Ground truth
K-means says: [0, 0, 1, 0, 1, 1, 0, 1]  ← Pseudo-labels (some wrong!)
                     ↑           ↑
                  Wrong!      Wrong!

Normal anchor = mean([N, N, X, A])  ← Includes attack sample!
Attack anchor = mean([N, A, A, A])  ← Includes normal sample!

Result: Anchors are polluted and move to wrong positions!
```

### 2. **Distribution Mismatch in Support Set**

Your support set for TTT uses **100 samples** from the test set:

```python
support_size = 100
support_x_ttt = query_x[:support_size]  # First 100 test samples
```

**The Problem**: If these 100 samples don't represent the test distribution well:
- Anchors move to match these 100 samples
- But the other 656 samples have different distribution
- Result: Anchors are optimized for 100 samples, harm the other 656

**Example**:
```
Support set (100 samples):  90 Normal, 10 Attack (90/10 split)
Test set (756 samples):     361 Normal, 395 Attack (48/52 split)
                            ↑ Very different distribution!

Anchors move to match 90/10 split → Fail on 48/52 test set
```

### 3. **BatchNorm Updates Without Proper Model Adaptation**

TENT only updates BatchNorm parameters, NOT the main network weights:

```
Frozen (NOT updated):
├─ TCN layers
├─ Projection layers
└─ Classification head

Updated (TENT):
└─ BatchNorm scale & shift (896 params)
```

**The Problem**: BatchNorm changes affect the embedding space, but:
- TCN weights stay fixed
- Prototypes are recomputed based on shifted embeddings
- But the TCN can't adapt to match the new prototypes
- Result: Mismatch between embeddings and prototypes

**Analogy**:
```
Training: TCN creates embeddings → Prototypes computed → Match well ✓

TTT:      BatchNorm shifts embeddings → Prototypes recomputed → TCN can't adapt
          ↓                             ↓                        ↓
       Shifted space              New prototypes          Old TCN weights
                                                               ↓
                                              MISMATCH → Poor performance ❌
```

### 4. **In-Place Adaptation Side Effects**

You're adapting the model in-place:

```python
adapted_model = self.model  # Same object!
```

**Potential Problem**: If the base model evaluation happens AFTER TTT adaptation (or shares any state), the "base" model is actually already adapted.

But your logs show base model is evaluated BEFORE TTT, so this is probably not the issue.

## Proposed Fixes

### Fix #1: Use Ground-Truth Support Set (If Available) ⭐ RECOMMENDED

Instead of k-means pseudo-labels, use actual labeled samples:

```python
# Current (k-means pseudo-labels):
pseudo_labels = kmeans.fit_predict(embeddings)  # Noisy!

# Better (use known labels if available):
support_labels = actual_labels[:100]  # True labels
```

**But wait**: In true test-time training, you don't have labels! That's the whole point.

**Alternative**: Use **confidence-based filtering**:
```python
# Only use high-confidence predictions for anchor computation
predictions = model(support_x)
confidence = softmax(predictions).max(dim=1)

# Keep only confident samples (>90% confidence)
confident_mask = confidence > 0.9
confident_samples = support_x[confident_mask]
confident_labels = predictions[confident_mask].argmax(dim=1)

# Compute anchors from confident samples only
anchors = compute_prototypes(confident_samples, confident_labels)
```

### Fix #2: Increase Support Set Size

Use more samples for anchor computation:

```python
# Current:
support_size = 100

# Better:
support_size = 300  # or even 500
```

More samples → Better representation → More accurate k-means → Better anchors

### Fix #3: Use Transductive Inference

Instead of computing anchors once and using them for all test samples:

```python
# For each test batch:
1. Include test batch in anchor computation
2. Compute anchors from [support_set + current_test_batch]
3. Classify current_test_batch using these anchors
4. Repeat for next batch
```

This is **truly transductive** - uses test samples in their own classification.

### Fix #4: Disable Prototype Updates During TTT

Maybe the issue is that updating prototypes every 10 steps is causing instability:

```python
# Current:
prototype_update_interval = 10  # Update every 10 steps

# Try:
prototype_update_interval = 999999  # Never update (use initial prototypes)
```

Let BatchNorm adapt, but keep prototypes fixed.

### Fix #5: Add Consistency Regularization

Ensure predictions don't change too drastically:

```python
# Store base model predictions
with torch.no_grad():
    base_predictions = base_model(query_x)

# During TTT, add consistency loss:
adapted_predictions = adapted_model(query_x)
consistency_loss = (adapted_predictions - base_predictions).pow(2).mean()

total_loss = entropy_loss + pseudo_loss + 0.1 * consistency_loss
```

This prevents anchors from moving too far from their original positions.

## Diagnostic Questions

To understand which fix to apply, we need to answer:

### Q1: Is k-means clustering accurate?

**Check**: Compare k-means pseudo-labels to true labels (if available in test set)

```python
pseudo_labels = kmeans.fit_predict(embeddings)
true_labels = support_y  # If available
accuracy = (pseudo_labels == true_labels).mean()
print(f"K-means accuracy: {accuracy:.2%}")
```

If accuracy < 70%, k-means is too noisy.

### Q2: Is the support set representative?

**Check**: Compare support set distribution to test set distribution

```python
support_dist = {
    'normal': (support_y == 0).sum(),
    'attack': (support_y == 1).sum()
}
test_dist = {
    'normal': (test_y == 0).sum(),
    'attack': (test_y == 1).sum()
}
print(f"Support: {support_dist}")
print(f"Test: {test_dist}")
```

If distributions differ significantly (>10%), support set is not representative.

### Q3: Are anchors actually moving?

**Check**: Log anchor positions before and after TTT

```python
# Before TTT:
initial_anchors = compute_prototypes(support_x, support_y)
print(f"Initial anchors:\n{initial_anchors}")

# After TTT (step 200):
final_anchors = compute_prototypes(support_x, support_y)
print(f"Final anchors:\n{final_anchors}")

# Distance moved:
distance = (final_anchors - initial_anchors).norm(dim=1)
print(f"Anchor movement: {distance}")
```

If distance < 0.01, anchors barely moved.
If distance > 10, anchors moved too much.

### Q4: Are BatchNorm updates helping or hurting?

**Check**: Disable BatchNorm updates and see if performance improves

```python
# Try running TTT with NO parameter updates:
# (only update prototypes via k-means, don't update BatchNorm)
```

If performance improves, BatchNorm updates are the problem.

## Summary

**Is the anchor issue fixed?**

**NO.** The anchors ARE being updated, but they're moving in the wrong direction, causing:
- Base: 81.75%
- TTT: 21.01% (huge degradation!)

**Most likely causes**:
1. K-means pseudo-labels are noisy → Anchors move to wrong positions
2. Support set not representative → Anchors optimized for wrong distribution
3. BatchNorm-only updates → Embedding space shifts but TCN can't adapt

**Recommended next steps**:
1. Check k-means accuracy on support set
2. Try larger support set (300-500 samples)
3. Use confidence-based filtering for anchor computation
4. Consider disabling prototype updates (use initial prototypes only)

The core issue is: **Unsupervised anchor adaptation (k-means) is unreliable without good pseudo-labels.**
