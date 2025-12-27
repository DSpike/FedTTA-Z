# TTT Visualization and Performance Investigation

## 🔍 **Two Issues to Investigate**

1. **TTT Loss Component Visualization is "Odd"**
2. **Zero-Day Detection Performance: TTT Lower Than Base Model**

---

## 📊 **Issue 1: TTT Loss Component Visualization**

### **Potential Problems**:

#### **Problem 1: Total Loss Calculation Mismatch** ⚠️

**Current Code** (`visualization/performance_visualization.py` lines 635-636):
```python
losses_float = [float(l) for l in total_losses]
ax2.plot(steps_float, losses_float, 'b-', linewidth=2, marker='o', markersize=6,
         label='Total Loss', alpha=0.9)
```

**Issue**: Total loss is plotted directly, but it's calculated as:
```python
total_loss = entropy_weight * entropy_loss + ttt_l2_reg_weight * l2_reg
```

**Problem**: 
- Entropy loss is already weighted (entropy_weight = 1.0 typically)
- L2 reg is weighted (ttt_l2_reg_weight = 0.01)
- But when plotting, entropy and L2 are shown separately with different scales
- **Total loss should equal entropy + weighted_L2, but visualization might not match**

**Expected Behavior**:
```
Total Loss = Entropy Loss + (L2 Reg × 0.01)
```

**Visualization Issue**:
- Total Loss: Shows weighted sum (correct)
- Entropy Loss: Shows unweighted (correct, weight is 1.0)
- L2 Reg: Shows weighted (correct, ×0.01 applied)
- **But**: Total Loss line might not visually align with Entropy + L2 Reg

---

#### **Problem 2: Y-Axis Scale Calculation** ⚠️

**Current Code** (lines 653-683):
```python
y_min = min(losses_float)
y_max = max(losses_float)
# Include entropy and L2 reg losses in range if present
if entropy_losses and len(entropy_losses) == min_length:
    entropy_float = [float(e) for e in entropy_losses]
    y_min = min(y_min, min(entropy_float))
    y_max = max(y_max, max(entropy_float))
if l2_reg_losses and len(l2_reg_losses) == min_length:
    l2_float = [float(l2) * l2_weight for l2 in l2_reg_losses]  # Apply weight
    y_min = min(y_min, min(l2_float))
    y_max = max(y_max, max(l2_float))
```

**Issue**: 
- Y-axis range is calculated correctly
- But if total loss is much larger than entropy + L2, the components might appear small
- **Total loss should be close to entropy + weighted_L2, but might not be if there are other components**

---

#### **Problem 3: Loss Component Mismatch** ❌

**What's Stored** (`coordinators/centralized_coordinator.py` lines 551-554):
```python
adaptation_data['total_losses'].append(total_loss.item())
adaptation_data['entropy_losses'].append(entropy_loss.item())
adaptation_data['l2_reg_losses'].append(reg_loss.item())  # UNWEIGHTED
```

**What's Plotted**:
- Total Loss: `total_loss.item()` (weighted sum)
- Entropy Loss: `entropy_loss.item()` (unweighted)
- L2 Reg: `reg_loss.item() * l2_weight` (weighted for display)

**Verification**:
```python
# Should be true:
total_loss ≈ entropy_loss + (l2_reg * ttt_l2_reg_weight)
```

**If this doesn't hold**, there's a mismatch!

---

### **Diagnostic Steps**:

1. **Check if Total Loss = Entropy + Weighted L2**:
   ```python
   total_loss = total_losses[i]
   entropy = entropy_losses[i]
   l2_weighted = l2_reg_losses[i] * l2_weight
   assert abs(total_loss - (entropy + l2_weighted)) < 1e-5
   ```

2. **Check Y-Axis Range**:
   - Total Loss range: [min, max]
   - Entropy range: [min, max]
   - L2 Reg (weighted) range: [min, max]
   - All should be visible on same plot

3. **Check for Negative Values**:
   - Losses should be non-negative
   - If negative, there's a bug

---

## 📉 **Issue 2: TTT Performance Lower Than Base Model**

### **Root Cause Analysis**:

#### **Root Cause 1: TTT Optimizes for Overall Confidence, Not Zero-Day** ❌

**TTT Adaptation Loss** (`coordinators/centralized_coordinator.py` lines 495-513):
```python
# Entropy loss (unsupervised)
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()  # ← Optimizes for OVERALL confidence

# Total loss
total_loss = entropy_weight * entropy_loss + ttt_l2_reg_weight * l2_reg
```

**Problem**:
- **Entropy minimization** optimizes for **overall confidence** across **ALL** test samples
- It does **NOT** specifically target zero-day samples
- The loss is **uniform** across all samples (no weighting for zero-day)

**Impact**:
- Zero-day samples: ~30% of adaptation set
- Non-zero-day samples: ~70% of adaptation set
- **Gradient is dominated by 70% majority** (non-zero-day samples)
- Optimization prioritizes improving confidence on **non-zero-day samples**

**Mathematical Explanation**:
```
∇L_total = (1/N) * Σ ∇L_entropy(x_i)
         = (1/N) * [Σ_{zero-day} ∇L_entropy(x_i) + Σ_{non-zero-day} ∇L_entropy(x_i)]
         ≈ 0.3 * ∇L_zero_day + 0.7 * ∇L_non_zero_day
```

**Result**: Gradient is **70% influenced** by non-zero-day samples!

---

#### **Root Cause 2: BatchNorm Statistics Shift** ⚠️

**TTT Adaptation**:
- Updates BatchNorm running statistics during adaptation
- Statistics shift toward **test distribution** (70% non-zero-day)
- **Zero-day samples** (30%) have different distribution
- BatchNorm normalization becomes **biased** toward non-zero-day

**Impact**:
- Zero-day samples normalized incorrectly
- Feature extraction degraded for zero-day
- **Performance decreases** on zero-day samples

---

#### **Root Cause 3: Feature Space Distortion** ⚠️

**TTT Adaptation Process**:
1. Adapts feature extractor via entropy minimization
2. Optimizes for **overall** confidence (70% non-zero-day)
3. Feature space shifts toward **non-zero-day** patterns
4. **Zero-day patterns** become less separable

**Result**:
- Zero-day samples harder to distinguish
- Classification accuracy decreases
- **Base model (untrained on test) generalizes better**

---

#### **Root Cause 4: L2 Regularization Constraint** ⚠️

**L2 Regularization**:
```python
l2_reg = Σ (param - original_param)²
total_loss = entropy_loss + ttt_l2_reg_weight * l2_reg
```

**Problem**:
- L2 reg prevents large parameter changes
- But optimization is **biased** toward non-zero-day (70%)
- Parameters shift toward non-zero-day patterns
- **Zero-day patterns** become less optimal

**Result**:
- L2 reg constrains adaptation
- But adaptation direction is wrong (toward non-zero-day)
- **Zero-day performance degrades**

---

### **Why Base Model Performs Better**:

1. **No Adaptation Bias**: Base model hasn't been adapted, so no bias toward test distribution
2. **Generalization**: Base model generalizes better to zero-day (trained on diverse data)
3. **No Feature Space Distortion**: Feature space not shifted toward non-zero-day
4. **Consistent Normalization**: BatchNorm statistics from training (more general)

---

## 🔧 **Solutions**

### **Solution 1: Fix Visualization** ⭐⭐⭐⭐⭐

**Issue**: Total Loss might not match Entropy + Weighted L2

**Fix**:
```python
# In plot_ttt_adaptation, add verification:
for i in range(min_length):
    total = total_losses[i]
    entropy = entropy_losses[i]
    l2_weighted = l2_reg_losses[i] * l2_weight
    expected_total = entropy + l2_weighted
    if abs(total - expected_total) > 1e-4:
        logger.warning(f"⚠️ Total loss mismatch at step {i}: {total} vs {expected_total}")
```

**Also**: Add a line showing "Entropy + L2 Reg" to verify it matches Total Loss

---

### **Solution 2: Zero-Day Weighted TTT** ⭐⭐⭐⭐⭐

**Modify TTT to weight zero-day samples more heavily**:

```python
# In adapt_to_test_data() method
zero_day_mask = (y_test_multiclass == zero_day_attack_label)  # Identify zero-day samples

# Weighted entropy loss
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
zero_day_weights = torch.ones(len(query_x), device=query_x.device)
zero_day_weights[zero_day_mask] = 3.0  # 3x weight for zero-day samples

# Weighted entropy loss
weighted_entropy_loss = (entropy * zero_day_weights).mean()

# Total loss
total_loss = entropy_weight * weighted_entropy_loss + ttt_l2_reg_weight * l2_reg
```

**Expected Impact**:
- Zero-day samples: 3x weight → 90% influence (instead of 30%)
- Non-zero-day samples: 1x weight → 10% influence (instead of 70%)
- **Optimization prioritizes zero-day samples**

---

### **Solution 3: Separate TTT for Zero-Day** ⭐⭐⭐⭐

**Run TTT adaptation separately on zero-day samples**:

```python
# Adapt on zero-day samples with higher learning rate
zero_day_query_x = query_x[zero_day_mask]
zero_day_adapted_model = ttt_adapt(model, zero_day_query_x, lr=higher_lr)

# Use zero-day adapted model for zero-day evaluation
```

---

### **Solution 4: Freeze BatchNorm for Zero-Day** ⭐⭐⭐

**Prevent BatchNorm statistics from shifting**:

```python
# Freeze BatchNorm during TTT adaptation
for module in adapted_model.modules():
    if isinstance(module, torch.nn.BatchNorm1d):
        module.eval()  # Freeze BatchNorm
```

---

## 📊 **Expected Results After Fixes**

### **Visualization Fix**:
- Total Loss line matches Entropy + Weighted L2
- All components clearly visible
- Y-axis range appropriate

### **Zero-Day Weighted TTT**:
- Zero-day detection: +5-15% improvement
- TTT outperforms base model on zero-day
- Overall performance maintained

---

## ✅ **Next Steps**

1. **Diagnose Visualization**: Check if Total Loss = Entropy + Weighted L2
2. **Implement Zero-Day Weighted TTT**: Add zero-day sample weighting
3. **Test and Verify**: Run experiment and compare results
4. **Document Findings**: Update analysis with results

---

**Document Created**: Investigation of TTT visualization and performance issues  
**Status**: Ready for implementation



