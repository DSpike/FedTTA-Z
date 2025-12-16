# TTT Performance Trade-off Analysis: Zero-Day vs Non-Zero-Day Performance

## 🔍 **Problem Statement**

The optimized TTT configuration achieves **100% zero-day detection rate (ZDR)**, but it **degrades overall performance** compared to the base model:

### **Best Trial Results (Trial 1):**

| Metric | Base Model | TTT Model | Change |
|--------|------------|-----------|--------|
| **Zero-Day Detection Rate** | 95.45% | **100.00%** | **+4.55%** ✅ |
| **Overall Accuracy** | 74.55% | 71.82% | **-2.73%** ❌ |
| **Overall F1-Score** | 80.00% | 77.37% | **-2.63%** ❌ |
| **AUC-PR** | 80.92% | 82.67% | **+1.75%** ✅ |

**Key Finding**: TTT improves zero-day detection but **hurts non-zero-day performance**.

---

## 🎯 **Root Cause Analysis**

### **1. Pure TENT Configuration**

The best trial uses **pure TENT (entropy minimization only)** without pseudo-labels:

```json
{
  "use_pseudo_labels": false,
  "entropy_weight": 1.2290,
  "ttt_lr": 0.000508,
  "ttt_base_steps": 228
}
```

### **2. How Entropy Minimization Works**

- **Goal**: Make model predictions more confident (lower entropy)
- **Mechanism**: Minimizes prediction entropy across **ALL** test samples
- **Adaptation**: Adjusts model parameters to fit the test distribution

### **3. The Trade-off**

#### **✅ Why Zero-Day Detection Improves:**

1. **Initial Uncertainty**: Base model is uncertain about zero-day samples (unseen during training)
2. **Entropy Minimization**: Encourages confident predictions for all samples
3. **Adaptation**: Model learns to identify zero-day patterns during TTT
4. **Result**: Zero-day detection rate increases from 95.45% to 100%

#### **❌ Why Non-Zero-Day Performance Degrades:**

1. **Overfitting to Test Distribution**: 
   - Test set: 20-30% zero-day, 70-80% non-zero-day
   - Model adapts to this specific distribution

2. **Overconfidence in Wrong Predictions**:
   - Some non-zero-day samples may be initially misclassified
   - Entropy minimization forces confident predictions
   - Model becomes overconfident in **incorrect** predictions for non-zero-day samples

3. **Loss of Generalization**:
   - Model may drift away from training distribution
   - Adaptation prioritizes test distribution over training knowledge

### **4. Mathematical Explanation**

Entropy minimization optimizes:
```
Loss = -Σ p(y|x) * log(p(y|x))
```

This is **unaware** of:
- Which samples are zero-day vs non-zero-day
- Which predictions are correct vs incorrect
- Training distribution characteristics

It simply makes **all** predictions more confident, which can:
- ✅ Help uncertain zero-day samples (correctly identifies them)
- ❌ Hurt misclassified non-zero-day samples (increases confidence in wrong predictions)

---

## 💡 **Potential Solutions**

### **Solution 1: Enable Pseudo-Labels** ⭐ **RECOMMENDED**

**Approach**: Use `use_pseudo_labels=True` to provide supervision signal

**Benefits**:
- Uses confident predictions as ground truth
- Prevents overconfidence in wrong predictions
- Balances entropy minimization with supervised learning

**How it works**:
- Only high-confidence predictions (>threshold) are used as pseudo-labels
- Loss combines: `Entropy Loss + Pseudo-Label Cross-Entropy Loss`
- Model adapts with both unsupervised (entropy) and supervised (pseudo-labels) signals

**Expected Impact**:
- Maintains high zero-day detection rate
- Improves non-zero-day performance
- More balanced adaptation

**Implementation**:
```python
# In optimization, try:
"use_pseudo_labels": true,  # Changed from false
"pseudo_threshold": 0.85,   # High confidence threshold
"pseudo_weight": 2.0,       # Weight for pseudo-label loss
```

---

### **Solution 2: Weighted Entropy Minimization**

**Approach**: Only minimize entropy for confident samples

**Benefits**:
- Avoids forcing confidence on ambiguous/incorrect samples
- Reduces false positives (FPR)
- Prevents overfitting to wrong predictions

**How it works**:
- Filter entropy loss: `entropy_mask = max_probs > threshold` (e.g., 0.4)
- Only minimize entropy for samples with `max_probs > threshold`
- Ignores low-confidence samples (likely wrong predictions)

**Current Implementation**: Already implemented as "Filtered Entropy Minimization" with `entropy_threshold = 0.40`

**Limitation**: Still doesn't distinguish zero-day from non-zero-day

---

### **Solution 3: Regularization During TTT**

**Approach**: Add L2 penalty to prevent large parameter changes

**Benefits**:
- Prevents model from drifting too far from base model
- Maintains training distribution knowledge
- Reduces overfitting

**Implementation**:
```python
# Proximal regularization
proximal_term = ||θ_ttt - θ_base||²
total_loss = entropy_loss + λ * proximal_term
```

**Expected Impact**:
- Maintains base model performance on non-zero-day
- Moderate improvement in zero-day detection
- Balanced trade-off

---

### **Solution 4: Multi-Objective Optimization**

**Approach**: Optimize both zero-day AND non-zero-day performance

**Benefits**:
- Explicitly balances both objectives
- Can prioritize zero-day while maintaining non-zero-day performance

**Implementation**:
```python
# In optimize_hyperparameters.py, change objective:
# Instead of: metric_value = ttt_zdr
# Use: metric_value = 0.7 * ttt_zdr + 0.3 * ttt_non_zero_day_f1
```

**Expected Impact**:
- Balanced improvement in both metrics
- May sacrifice some zero-day performance for overall gains

---

## 📊 **Recommendation**

### **For Zero-Day Detection Priority** (Current Use Case):

✅ **Accept the trade-off**:
- Zero-day detection is the primary objective
- 100% ZDR is excellent
- 2-3% degradation in overall metrics is acceptable
- AUC-PR actually improves (+1.75%)

### **For Balanced Performance**:

1. **Enable pseudo-labels** (`use_pseudo_labels=True`)
2. **Re-run optimization** with pseudo-labels enabled
3. **Compare** zero-day vs overall performance trade-off

### **For Maximum Performance**:

1. **Use multi-objective optimization**:
   ```python
   metric = 0.6 * zdr + 0.4 * overall_f1
   ```
2. **Experiment with regularization** (L2 penalty)
3. **Tune hyperparameters** for balanced performance

---

## 🔬 **Next Steps**

1. **Re-run optimization with pseudo-labels enabled**:
   ```bash
   python optimize_hyperparameters.py --n_trials 10
   ```
   (Update `use_pseudo_labels` to be optimized or set to `True`)

2. **Compare results**:
   - Zero-day detection rate
   - Non-zero-day accuracy/F1
   - Overall performance

3. **Choose configuration** based on priority:
   - Zero-day detection priority → Current config (pure TENT)
   - Balanced performance → Pseudo-labels enabled
   - Maximum performance → Multi-objective optimization

---

## 📝 **Summary**

**The Issue**: Pure TENT (entropy-only) improves zero-day detection but hurts non-zero-day performance due to:
- Overfitting to test distribution
- Overconfidence in wrong predictions
- Loss of training distribution knowledge

**The Solution**: 
- For zero-day priority: Accept current trade-off ✅
- For balanced performance: Enable pseudo-labels ⭐
- For maximum performance: Multi-objective optimization

**The Trade-off**: Improving zero-day detection by 4.55% (95.45% → 100%) at the cost of 2-3% overall performance degradation is **acceptable for zero-day detection use cases**, where catching all zero-day attacks is more critical than perfect performance on known attacks.










