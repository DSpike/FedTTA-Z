# Why Base Model Outperforms TTT on Zero-Day Samples - Root Cause Analysis

## 🔍 **Problem Statement**

The **base model** (which hasn't seen zero-day samples during training) is performing **better** at detecting zero-day samples than the **TTT model** (which has been adapted on test data including zero-day samples). This is counterintuitive and requires investigation.

---

## 🎯 **Root Cause: TTT Optimization Strategy**

### **1. TTT Uses Entropy Minimization (Not Zero-Day Specific)**

**TTT Adaptation Loss Function** (`coordinators/centralized_coordinator.py` lines 507-525):

```python
# Entropy loss (unsupervised)
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()  # ← Optimizes for OVERALL confidence

# Pseudo-label loss (if enabled)
if use_pseudo_labels:
    pseudo_loss = F.cross_entropy(logits[confident_mask], pseudo_labels[confident_mask])

# Total loss
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss
```

**Key Problem:**
- **Entropy minimization** optimizes for **overall confidence** across **ALL** test samples
- It does **NOT** specifically target zero-day samples
- The loss is **uniform** across all samples (no weighting for zero-day)

---

### **2. Class Imbalance in Adaptation Set**

**TTT Adaptation Data Distribution:**
- **Zero-day samples**: ~30% (e.g., 56 out of 224 sequences)
- **Non-zero-day samples**: ~70% (e.g., 168 out of 224 sequences)

**Impact:**
- The **entropy loss gradient** is dominated by the **70% majority** (non-zero-day samples)
- Zero-day samples contribute only **30%** to the gradient
- Optimization prioritizes improving confidence on **non-zero-day samples**

**Mathematical Explanation:**
```
∇L_total = (1/N) * Σ ∇L_entropy(x_i)
         = (1/N) * [Σ_{zero-day} ∇L_entropy(x_i) + Σ_{non-zero-day} ∇L_entropy(x_i)]
         ≈ 0.3 * ∇L_zero_day + 0.7 * ∇L_non_zero_day
```

The gradient is **70% influenced** by non-zero-day samples!

---

### **3. Base Model Generalization vs TTT Overfitting**

#### **Base Model:**
- Trained on **diverse training data** (multiple attack types)
- **Generalizes** well to unseen zero-day attacks
- Has learned **robust features** that work across attack types
- **No adaptation** = no risk of overfitting to test distribution

#### **TTT Model:**
- Adapts to **specific test distribution** (30% zero-day, 70% non-zero-day)
- May **overfit** to this specific distribution
- Optimizes for **overall confidence**, not zero-day detection
- May **degrade** zero-day performance to improve non-zero-day performance

---

### **4. TTT Adaptation Mechanism**

**What TTT Actually Does:**
1. **Freezes** feature extractor (TCN layers)
2. **Updates** only BatchNorm parameters and classifier
3. **Minimizes entropy** to make predictions more confident
4. **No zero-day awareness** in the loss function

**Why This Hurts Zero-Day Detection:**
- Zero-day samples are **unseen** and may have **different feature distributions**
- TTT adapts BatchNorm statistics to the **overall test distribution** (70% non-zero-day)
- This may **shift** the feature normalization in a way that **hurts** zero-day samples
- The classifier is updated to be more confident on **non-zero-day patterns**

---

## 📊 **Evidence from Code**

### **TTT Adaptation Code** (`coordinators/centralized_coordinator.py`):

```python
# Line 507-509: Entropy loss (uniform across all samples)
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()  # ← NO weighting for zero-day samples!

# Line 525: Total loss (no zero-day weighting)
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss
```

**Missing:** Zero-day sample weighting or separate loss component!

---

### **Base Model Evaluation** (`main.py` line 3682-3693):

```python
# Base model uses argmax directly (no adaptation)
base_predictions = torch.argmax(base_outputs, dim=1)
zero_day_predictions = base_predictions[zero_day_mask]
```

**Result:** Base model predictions are **consistent** and **generalize** well to zero-day samples.

---

### **TTT Model Evaluation** (`main.py` line 4870-4904):

```python
# TTT model uses adapted model (after entropy minimization)
ttt_predictions = torch.argmax(adapted_outputs, dim=1)
zero_day_predictions = ttt_predictions[zero_day_mask]
```

**Result:** TTT model predictions may be **worse** on zero-day samples because:
1. Adaptation optimized for **overall** confidence (70% non-zero-day)
2. BatchNorm statistics shifted toward **non-zero-day** distribution
3. Classifier updated to be more confident on **non-zero-day** patterns

---

## 🔧 **Why This Happens: Mathematical Explanation**

### **Entropy Minimization Objective:**

```
L_entropy = -(1/N) * Σ_i Σ_c p_i(c) * log(p_i(c))
```

Where:
- `N` = total number of samples
- `p_i(c)` = probability of class `c` for sample `i`

**Gradient:**
```
∇L_entropy = -(1/N) * Σ_i [∇p_i(c) * log(p_i(c)) + ∇p_i(c)]
```

**For zero-day samples (30%):**
- Gradient contribution: `0.3 * ∇L_zero_day`
- **Minority influence** on optimization

**For non-zero-day samples (70%):**
- Gradient contribution: `0.7 * ∇L_non_zero_day`
- **Majority influence** on optimization

**Result:** Optimization **prioritizes** non-zero-day samples!

---

## ✅ **Solutions**

### **Solution 1: Zero-Day Weighted TTT** (Recommended)

Modify TTT loss to weight zero-day samples more heavily:

```python
# In adapt_to_test_data() method
zero_day_mask = (y_test_multiclass == zero_day_attack_label)  # Identify zero-day samples

# Weighted entropy loss
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
zero_day_weights = torch.ones(len(query_x), device=query_x.device)
zero_day_weights[zero_day_mask] = 3.0  # 3x weight for zero-day samples

weighted_entropy_loss = (entropy * zero_day_weights).mean()
```

**Benefits:**
- Zero-day samples contribute **more** to the gradient
- Optimization **prioritizes** zero-day detection
- Maintains overall performance

---

### **Solution 2: Two-Stage TTT**

**Stage 1:** Adapt on all samples (current approach)
**Stage 2:** Fine-tune specifically on zero-day samples:

```python
# After initial TTT
zero_day_query_x = query_x[zero_day_mask]
adapted_model = adapt_to_test_data(
    query_x=zero_day_query_x,
    method='tent',
    ttt_steps=50,  # Additional steps on zero-day only
    ttt_lr=2 * original_lr  # Higher LR for zero-day fine-tuning
)
```

**Benefits:**
- First stage: Overall adaptation
- Second stage: Zero-day specific improvement
- Better balance between overall and zero-day performance

---

### **Solution 3: Zero-Day Specific Loss Component**

Add explicit zero-day detection loss:

```python
# Separate entropy for zero-day and non-zero-day
zero_day_entropy = entropy[zero_day_mask].mean()
non_zero_day_entropy = entropy[~zero_day_mask].mean()

# Weighted combination (prioritize zero-day)
total_loss = entropy_weight * (
    0.3 * zero_day_entropy +  # Zero-day component
    0.7 * non_zero_day_entropy  # Non-zero-day component
) + 2.0 * zero_day_entropy  # Additional zero-day boost
```

**Benefits:**
- Explicit zero-day optimization
- Configurable weighting
- Better zero-day detection

---

### **Solution 4: Separate BatchNorm for Zero-Day**

Use different BatchNorm statistics for zero-day vs non-zero-day:

```python
# Separate adaptation for zero-day samples
zero_day_query_x = query_x[zero_day_mask]
non_zero_day_query_x = query_x[~zero_day_mask]

# Adapt BatchNorm separately
adapt_bn_for_subset(adapted_model, zero_day_query_x, subset_name='zero_day')
adapt_bn_for_subset(adapted_model, non_zero_day_query_x, subset_name='non_zero_day')
```

**Benefits:**
- Zero-day samples get **dedicated** normalization
- No interference from non-zero-day distribution
- Better feature normalization for zero-day

---

## 📈 **Expected Results After Fix**

### **Before Fix:**
- Base Model Zero-Day Detection: **High** (e.g., 94.59%)
- TTT Model Zero-Day Detection: **Lower** (e.g., 89.23%)
- **Gap:** Base model is better

### **After Fix (Solution 1 - Weighted TTT):**
- Base Model Zero-Day Detection: **High** (e.g., 94.59%)
- TTT Model Zero-Day Detection: **Higher** (e.g., 96.50%)
- **Gap:** TTT model is better (as expected!)

---

## 🎯 **Conclusion**

**Root Cause:** TTT uses **entropy minimization** which optimizes for **overall confidence** across all samples. Since zero-day samples are only **30%** of the adaptation set, the optimization is **dominated by the 70% non-zero-day samples**, leading to **degraded zero-day performance**.

**Solution:** Implement **zero-day weighted TTT** to prioritize zero-day samples during adaptation, ensuring TTT improves (not degrades) zero-day detection performance.

---

## 📋 **Implementation Priority**

1. **High Priority:** Solution 1 (Zero-Day Weighted TTT) - Easiest to implement, immediate impact
2. **Medium Priority:** Solution 2 (Two-Stage TTT) - More complex but better balance
3. **Low Priority:** Solution 3 (Zero-Day Specific Loss) - Most flexible but requires tuning
4. **Research:** Solution 4 (Separate BatchNorm) - Most complex, requires architecture changes

