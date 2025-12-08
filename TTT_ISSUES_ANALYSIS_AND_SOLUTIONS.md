# TTT Issues Analysis and Solutions

## 🔍 **Two Critical Issues Identified**

### **Issue 1: Pseudo-Label Loss Not Decreasing**
- **Observation**: Pseudo-label loss stays almost constant (0.010546 → 0.008053, only -0.002493)
- **Root Cause**: **Pseudo-label ratio is only 1.7%** - too few samples contribute to loss
- **Impact**: Pseudo-labels barely influence optimization, so loss plateaus

### **Issue 2: Zero-Day Excellent, Known Attacks Poor**
- **Observation**: Zero-day detection = 94.49% ✅, but Non-zero-day F1 = 70.57% ⚠️
- **Root Cause**: TTT adapts to overall distribution, prioritizing zero-day patterns
- **Impact**: Trade-off between zero-day detection and known attack performance

---

## 📊 **Issue 1: Why Pseudo-Label Loss Doesn't Decrease**

### **Problem Analysis**

From logs:
```
Pseudo-label Loss: 0.010546 → 0.008053 (-0.002493, only -23.6% decrease)
Pseudo ratio: 1.7% (only 1.7% of samples meet confidence threshold)
Threshold: 0.95 → 0.73 (very high threshold, very few samples qualify)
```

### **Root Causes**

1. **Threshold Too High**:
   - Initial: 0.95 (only 1.7% samples meet this)
   - Final: ~0.73 (still restrictive)
   - Current config: `pseudo_threshold=0.950`, `pseudo_min_threshold=0.732`
   - **Only samples with max_probs > 0.95 (initially) → 0.73 (finally) get pseudo-labels**

2. **Very Few Confident Samples**:
   - With 635 test samples, only ~11 samples (1.7%) meet the threshold
   - Pseudo-label loss averages over these 11 samples only
   - Even if these 11 samples improve, the average doesn't change much

3. **Loss Calculation**:
   ```python
   # Only confident samples contribute to pseudo-label loss
   confident_mask = max_probs > threshold  # Only 1.7% of samples
   batch_pseudo_loss = F.cross_entropy(logits[confident_mask], pred_labels[confident_mask])
   avg_pseudo_loss = total_pseudo_loss / total_confident_samples  # Average over ~11 samples
   ```

### **Why This Happens**

The optimized hyperparameters from Trial 6 set:
- `pseudo_threshold: 0.950` (very high - only top 5% most confident)
- `pseudo_min_threshold: 0.732` (still high - only top 27% most confident)

**This was optimized for zero-day detection**, which prioritizes precision over recall. However, it means:
- Very few pseudo-labels are generated
- Pseudo-label loss has minimal contribution to total loss
- Loss is dominated by entropy minimization (which works well)

---

## 📊 **Issue 2: Why Zero-Day Excellent But Known Attacks Poor**

### **Performance Breakdown**

| Category | Zero-Day | Non-Zero-Day | Issue |
|----------|----------|--------------|-------|
| **Accuracy** | 94.49% ✅ | 70.28% ⚠️ | 24% gap |
| **F1-Score** | 97.17% ✅ | 70.57% ⚠️ | 27% gap |
| **Precision** | 100.00% ✅ | 59.74% ⚠️ | 40% gap |
| **Recall** | 94.49% ✅ | 86.19% ✅ | Both high |

### **Root Cause Analysis**

#### **1. Distribution Mismatch During TTT Adaptation**

- **Test Set Composition**: 20% zero-day, 80% known attacks
- **TTT Adapts On**: Entire test set (both zero-day and known)
- **Problem**: TTT entropy minimization optimizes for the **overall distribution**

#### **2. Entropy Patterns Differ**

**Zero-Day Samples** (Unseen Patterns):
- Initially **high entropy** (model uncertain - never seen these patterns)
- TTT entropy minimization **reduces entropy** → improves confidence → better detection
- Zero-day samples benefit from entropy reduction

**Known Attack Samples** (Seen Patterns):
- Initially **low entropy** (model confident - seen during training)
- Some known attacks might have **moderate entropy** (similar to normal traffic)
- TTT might **increase entropy** on these or misalign decision boundaries
- Known attacks suffer from over-aggressive adaptation

#### **3. Decision Boundary Shift**

During TTT adaptation:
- Model adjusts decision boundaries to reduce overall entropy
- This **helps zero-day** (reduces uncertainty)
- But **hurts known attacks** if the boundary shifts too far from training distribution

#### **4. Gradient Dominance**

From test set (635 samples):
- **Zero-day**: 127 samples (20%) - **High entropy, large gradients**
- **Known attacks**: 508 samples (80%) - **Mixed entropy, smaller gradients**

**Problem**: Zero-day samples have larger entropy gradients → dominate optimization → model over-adapts to zero-day patterns

---

## 🔧 **Proposed Solutions**

### **Solution 1: Fix Pseudo-Label Loss (Lower Threshold)**

**Problem**: Threshold too high (0.95 → 0.73), only 1.7% samples qualify

**Solution**: Lower threshold to increase pseudo-label contribution

```python
# In config.py
pseudo_threshold: float = 0.75  # Lower from 0.95 (allows ~25% of samples)
pseudo_min_threshold: float = 0.60  # Lower from 0.73 (allows ~40% of samples)
```

**Expected Impact**:
- Pseudo-label ratio: 1.7% → 25-40%
- More samples contribute to pseudo-label loss
- Loss can decrease more significantly
- Better balance between entropy and pseudo-label terms

**Trade-off**: Lower precision on pseudo-labels (more noise), but more supervision signal

---

### **Solution 2: Balance Zero-Day vs Known Attacks**

**Problem**: TTT over-adapts to zero-day, hurting known attacks

**Solution A: Weighted Loss by Sample Type**

```python
# Weight known attacks more heavily to balance optimization
known_attack_weights = torch.ones(len(query_x))
known_attack_weights[known_attack_mask] = 2.0  # 2x weight for known attacks
zero_day_weights = torch.ones(len(query_x))
zero_day_weights[zero_day_mask] = 1.5  # 1.5x weight for zero-day

# Weighted entropy loss
weighted_entropy = (entropy_loss * known_attack_weights).mean()
```

**Solution B: Separate Adaptation Phases**

```python
# Phase 1: Adapt on known attacks first (preserve base model knowledge)
known_attack_query_x = query_x[~zero_day_mask]
adapted_model = ttt_adapt(adapted_model, known_attack_query_x, steps=150)

# Phase 2: Fine-tune on zero-day (adapt to unseen patterns)
zero_day_query_x = query_x[zero_day_mask]
adapted_model = ttt_adapt(adapted_model, zero_day_query_x, steps=100, lr=0.5*lr)
```

**Solution C: Regularization to Prevent Over-Adaptation**

```python
# Add regularization term to prevent drifting too far from base model
proximal_term = sum((p - p_base).norm(2)**2 for p, p_base in zip(model.params(), base_model.params()))
total_loss = entropy_loss + pseudo_loss + 0.01 * proximal_term
```

---

### **Solution 3: Adaptive Pseudo-Label Threshold**

**Problem**: Fixed threshold (even if adaptive) doesn't account for sample difficulty

**Solution**: Use percentile-based threshold (always use top K% confident samples)

```python
# Use top 30% most confident samples for pseudo-labels (adaptive)
confidence_scores = probs.max(dim=1)[0]
threshold = torch.quantile(confidence_scores, 0.70)  # 70th percentile
confident_mask = confidence_scores > threshold
```

**Benefits**:
- Always uses 30% of samples (fixed ratio)
- Adapts threshold to actual confidence distribution
- More stable pseudo-label contribution

---

### **Solution 4: Class-Balanced Pseudo-Labels**

**Problem**: Pseudo-labels might be biased toward one class

**Solution**: Ensure balanced pseudo-labels from both classes

```python
# Select top K% from each class separately
class_0_mask = pred_labels == 0
class_1_mask = pred_labels == 1

if class_0_mask.sum() > 0:
    class_0_conf = confidence_scores[class_0_mask]
    class_0_threshold = torch.quantile(class_0_conf, 0.70)
    class_0_confident = class_0_conf > class_0_threshold

if class_1_mask.sum() > 0:
    class_1_conf = confidence_scores[class_1_mask]
    class_1_threshold = torch.quantile(class_1_conf, 0.70)
    class_1_confident = class_1_conf > class_1_threshold

confident_mask = combine(class_0_confident, class_1_confident)
```

---

## 🎯 **Recommended Immediate Fixes**

### **Priority 1: Lower Pseudo-Label Threshold** (Quick Fix)

```python
# In config.py
pseudo_threshold: float = 0.75  # Lower from 0.950 (more samples qualify)
pseudo_min_threshold: float = 0.60  # Lower from 0.732 (more stable ratio)
```

**Expected**: Pseudo-label ratio increases from 1.7% to 25-40%

---

### **Priority 2: Add Known Attack Weighting** (Medium Fix)

Modify TTT adaptation to weight known attacks more heavily:

```python
# In TENTPseudoLabels.adapt()
known_attack_mask = ~zero_day_mask  # Assuming zero_day_mask is available
weights = torch.ones(len(query_x))
weights[known_attack_mask] = 1.5  # 1.5x weight for known attacks

weighted_entropy = (entropy_loss * weights).mean()
```

**Expected**: Better balance, known attack F1 improves from 70% to 75-80%

---

### **Priority 3: Adaptive Threshold Based on Percentile** (Advanced Fix)

Replace fixed threshold with percentile-based:

```python
def _adaptive_threshold_percentile(self, probs, percentile=0.70):
    """Use percentile instead of fixed threshold"""
    confidence_scores = probs.max(dim=1)[0]
    threshold = torch.quantile(confidence_scores, percentile)
    return threshold
```

**Expected**: More stable pseudo-label contribution, better adaptation

---

## 📝 **Implementation Plan**

1. **Phase 1**: Lower pseudo-label thresholds (quick test)
2. **Phase 2**: Add known attack weighting (if Phase 1 doesn't help enough)
3. **Phase 3**: Implement adaptive percentile threshold (if needed)

---

## 🎯 **Expected Results After Fixes**

### **Before Fixes**:
- Pseudo-label ratio: 1.7%
- Zero-day F1: 97.17% ✅
- Known attack F1: 70.57% ⚠️

### **After Fixes**:
- Pseudo-label ratio: 25-40% (15-25x increase)
- Zero-day F1: 95-97% (maintain high performance)
- Known attack F1: 75-82% (+5-12% improvement)

---

## 🔍 **Key Insights**

1. **Pseudo-label threshold too conservative**: Optimized for precision, but limits learning signal
2. **TTT over-optimizes for zero-day**: Due to higher entropy gradients from unseen patterns
3. **Trade-off is inherent**: Zero-day vs known attack performance needs explicit balancing
4. **Both issues are fixable**: Lower thresholds + balanced weighting should help

---

**Next Steps**: Implement Priority 1 fix first (lower thresholds) and test.



