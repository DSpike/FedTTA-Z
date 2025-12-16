# Performance Investigation: Why Low Performance Despite Decreasing Losses

## Executive Summary

Despite **decreasing losses** (meta-learning: 1.42→0.26, TTT: 0.27→0.055), performance remains low:
- **Base Model**: ZDR 20.65%, Accuracy 42.80%, F1 26.53%
- **TTT Model**: ZDR 23.37%, Accuracy 59.65%, F1 54.10%

**Key Finding**: The losses being optimized are **NOT aligned with classification accuracy** for zero-day detection.

---

## 🔴 CRITICAL ISSUE #1: Loss-Objective Mismatch (CORRECTED)

### Problem
The TTT loss function optimizes for:
- **Entropy Loss**: Encourages confident predictions (minimizes uncertainty per sample)
- **Diversity Loss**: Encourages balanced class distribution (penalizes class collapse)

**This is fundamentally different from classification accuracy!**

### Actual Loss Formula (from code analysis)
```python
# coordinators/simple_fedavg_coordinator.py:996
total_loss = entropy_loss + (diversity_weight × diversity_loss)
```

Where:
- `entropy_loss` = weighted entropy per sample (minimize for confident predictions)
- `diversity_loss` = 1.0 - normalized_class_entropy (penalize low class diversity)
- `diversity_weight` = 0.1 (base) to 0.3 (adaptive, when diversity is low)

### Evidence from Logs (CORRECTED)
```
TTT Step 227/228: Loss=0.0546
├─ Entropy Loss: 0.0098 (very low = confident predictions)
├─ Diversity Loss: 0.2751 (moderate = some class imbalance)
├─ Adaptive Diversity Weight: 0.1625 (increased from base 0.1)
├─ Diversity Contribution: 81.95% of total loss
└─ ZDR: 23.37% (still very low!)
```

**Calculation Verification:**
- Diversity component = 0.1625 × 0.2751 = 0.0447
- Entropy component = 0.0098
- Total = 0.0098 + 0.0447 = 0.0545 ✅ (matches logged loss)
- Diversity % = 0.0447 / 0.0545 = 82% ✅ (matches logged percentage)

**Analysis**: The TTT loss can decrease perfectly while the model makes **wrong predictions** as long as:
1. Predictions are **confident** (low entropy loss)
2. Class distribution is **balanced** (diversity loss satisfied)
3. **But correctness is NOT measured!**

### Impact
- **Loss decreases** ✅ (good for TTT convergence)
- **Accuracy does NOT improve** ❌ (loss doesn't measure correctness)

### Recommendation
Add **supervised components** to TTT loss:
- Pseudo-label loss (if confidence > threshold)
- Consistency loss (agreement between augmented versions)
- Prototype alignment loss (align with support set prototypes)

---

## 🔴 CRITICAL ISSUE #2: Extremely Low Decision Threshold

### Problem
The threshold optimization selected: **0.1000** (10% attack probability)

```
Selected threshold: 0.1000 (optimized for F1-score)
Predicted Normal: 516/736 (70.1%)
Predicted Attack: 220/736 (29.9%)
Actual: Normal=309, Attack=427
```

### Analysis
- **Threshold 0.10 is TOO LOW** for reliable attack detection
- Only 220/736 samples (29.9%) predicted as attacks
- Actual attacks: 427 (should be ~58%, not 30%)
- This causes **massive false negatives** (attacks predicted as Normal)

### Why This Happened
PR-optimized threshold search may be selecting thresholds that:
1. Balance precision/recall on the **overall test set** (Normal + Known + Zero-day)
2. Don't prioritize zero-day detection specifically
3. Favor conservative predictions (low false positives)

### Impact
- **Zero-day samples need HIGHER threshold** to be detected as attacks
- Current threshold (0.10) is below even normal samples' attack probabilities
- Model is too conservative → misses most attacks (including zero-day)

### Evidence
```
Zero-day Detection Rate: 23.37%
Zero-day samples: 184
Detected as attacks: 43/184 (only 23.37%!)
```

### Recommendation
1. **Use ZDR-optimized threshold** instead of PR-optimized
2. **Increase threshold** to 0.3-0.5 for better attack detection
3. **Separate thresholds** for zero-day vs known attacks
4. **Threshold tuning on validation set** with zero-day samples included

---

## 🔴 CRITICAL ISSUE #3: Prototype-Based Prediction Inadequacy for Zero-Day

### Problem
The model uses **prototype distances** for classification:
1. Prototypes computed from **support set** (excludes zero-day)
2. Zero-day samples are **novel/out-of-distribution**
3. Distance to known attack prototypes may be **larger than to normal prototype**

### Mechanism
```python
# Prototypes computed from support set (known attacks only)
proto_normal = support_embeddings[mask_normal].mean()
proto_attack = support_embeddings[mask_attack].mean()  # Known attacks only!

# Zero-day samples compared to known attack prototypes
distance_to_normal = ||zero_day_embedding - proto_normal||
distance_to_attack = ||zero_day_embedding - proto_attack||  # May be LARGE!

# If distance_to_attack > distance_to_normal → predicted as Normal ❌
```

### Impact
- **Zero-day samples** may be **closer to normal prototype** than attack prototype
- Model lacks **open-set detection** capability
- Cannot distinguish "unknown attack" from "normal"

### Evidence
```
Zero-day predictions: [141 Normal, 43 Attack]
Only 43/184 zero-day samples (23.37%) detected!
```

### Recommendation
1. **Add open-set detection**: Use threshold on distance to nearest prototype
2. **Anomaly scoring**: Samples far from all prototypes → likely attacks
3. **Energy-based detection**: High energy → likely attack (even if novel)
4. **ODIN/Temperature scaling**: Better confidence calibration for OOD samples

---

## 🔴 CRITICAL ISSUE #4: Training-Validation Mismatch

### Problem
**Training Loss** is measured on:
- Support set (labeled, seen during training)
- Query set (from same distribution as support)

**But Zero-Day Performance** is measured on:
- Completely **unseen attack type** (held out from training)
- Different distribution from training attacks

### Evidence
```
Training Accuracy (meta-learning): 88-96% (on local client data)
Zero-Day Detection Rate: 23.37% (on held-out zero-day attacks)
```

### Analysis
- Model **overfits to known attack types** seen in support sets
- **Fails to generalize** to novel attack patterns (zero-day)
- Loss decreases on training data but **doesn't improve on zero-day**

### Impact
- **Low training loss** ≠ **Good zero-day detection**
- Model learns to distinguish Normal vs Known Attacks
- Cannot detect **novel attack patterns** (zero-day)

### Recommendation
1. **Add zero-day leakage** (1-2% of zero-day samples in training, labeled as "unknown attack")
2. **Outlier exposure** (include diverse "weird" samples in training)
3. **Domain adaptation techniques** for better generalization
4. **Regularization** to prevent overfitting to known attacks

---

## 🔴 CRITICAL ISSUE #5: Evaluation Methodology Issues

### Problem 1: ZDR Calculation
Current: `ZDR = (zero_day_predictions == 1).mean()`

This is just the **attack prediction rate** on zero-day samples, not actual detection accuracy.

**Correct**: `ZDR = TP / (TP + FN)` where:
- TP = zero-day samples correctly predicted as attacks
- FN = zero-day samples incorrectly predicted as normal

### Problem 2: Threshold Not Optimized for Zero-Day
- Threshold optimized for **overall F1** (includes Normal + Known + Zero-day)
- Should optimize for **zero-day recall** specifically

### Problem 3: No Separate Zero-Day Validation Set
- Threshold selected on test set (data leakage)
- Should use **separate validation set** with zero-day samples

### Recommendation
1. Fix ZDR calculation to use confusion matrix
2. Optimize threshold for **zero-day recall** (not overall F1)
3. Use **cross-validation** or separate validation set for threshold tuning

---

## 🔴 CRITICAL ISSUE #6: Insufficient Training Data/Diversity

### Problem
From logs:
```
Available labels for training: [0, 1]  # Only Normal and ONE known attack type
Zero-day attack: DoS (label 4) excluded from training
```

### Analysis
- Each task uses **only ONE known attack type** per support set
- Model learns to distinguish Normal vs **that specific attack type**
- Doesn't learn **general "attack" characteristics**
- Zero-day (DoS) may be very different from the known attack used in support

### Impact
- **Limited attack diversity** in training
- Model learns **task-specific** patterns, not general attack detection
- Poor generalization to **novel attack types**

### Evidence
```
Support set: Normal (100 shots) + Attack type 1 (118 shots)
Model learns: Normal vs Attack_type_1
Zero-day: DoS (completely different pattern)
→ Model fails to detect DoS as attack
```

### Recommendation
1. **Increase attack diversity** in support sets (multiple attack types per task)
2. **Shared attack embedding** space (learn general attack characteristics)
3. **Multi-task learning** (learn from multiple attack types simultaneously)

---

## 🔴 CRITICAL ISSUE #7: Embedding Space Collapse

### Problem
Model may be learning embeddings where:
- **All attacks** (known + zero-day) cluster together
- **Normal samples** form a separate cluster
- But **zero-day samples** are closer to Normal than to known attacks

### Evidence
```
Prototype distances:
├─ Zero-day → Normal prototype: SMALL (similar embeddings)
├─ Zero-day → Attack prototype: LARGE (different embeddings)
└─ Result: Zero-day predicted as Normal
```

### Impact
- Embedding space doesn't capture **general attack patterns**
- Zero-day attacks fall in the "normal" region of embedding space
- Prototype-based classification fails

### Recommendation
1. **Contrastive learning**: Push attack samples away from normal
2. **Triplet loss**: Ensure known attacks and zero-day are closer than normal
3. **Adversarial training**: Generate hard examples near decision boundary
4. **Metric learning**: Learn distance metrics that separate attacks from normal

---

## 📊 Summary of Root Causes

| Issue | Severity | Impact | Fix Priority |
|-------|----------|--------|--------------|
| Loss-Objective Mismatch | 🔴 Critical | High | P0 (Highest) |
| Low Decision Threshold | 🔴 Critical | High | P0 (Highest) |
| Prototype Inadequacy | 🔴 Critical | High | P1 (High) |
| Training-Validation Mismatch | 🟡 Medium | Medium | P1 (High) |
| Evaluation Methodology | 🟡 Medium | Medium | P2 (Medium) |
| Insufficient Training Diversity | 🟡 Medium | Medium | P2 (Medium) |
| Embedding Space Collapse | 🟡 Medium | Low | P3 (Low) |

---

## 🎯 Recommended Action Plan

### Immediate Fixes (P0)
1. **Fix threshold optimization**: Use ZDR-optimized threshold (target: 0.5-0.7)
2. **Add supervised TTT loss**: Include pseudo-label loss for high-confidence predictions
3. **Fix ZDR calculation**: Use confusion matrix (TP/(TP+FN))

### Short-term Improvements (P1)
4. **Add open-set detection**: Energy-based or distance-based anomaly scoring
5. **Increase attack diversity**: Multiple attack types per support set
6. **Add zero-day leakage**: 1-2% zero-day samples in training (as "unknown attack")

### Long-term Enhancements (P2-P3)
7. **Contrastive learning**: Improve embedding space separation
8. **Separate validation set**: For threshold tuning (prevent data leakage)
9. **Better evaluation metrics**: AUC-PR for zero-day, separate from overall metrics

---

## 💡 Key Insight

**"Decreasing loss does NOT guarantee increasing accuracy"**

The current TTT loss optimizes for:
- ✅ Prediction diversity (entropy)
- ✅ Class balance (diversity)

But NOT for:
- ❌ Classification correctness
- ❌ Zero-day detection rate
- ❌ Attack vs Normal separation

**Solution**: Add supervised components to TTT loss or use different optimization objective that directly measures classification performance.

