# TTT Performance Improvement Opportunities

## 📊 **Current TTT Implementation Analysis**

### ✅ **Currently Implemented:**
1. ✅ L2 Regularization (prevents parameter drift)
2. ✅ Entropy Loss (TENT - unsupervised)
3. ✅ Pseudo-label Loss (supervised component)
4. ✅ Gradient Clipping (max_norm=1.0)
5. ✅ Confidence-based Rejection
6. ✅ Temperature Scaling (post-TTT)
7. ✅ EMA Teacher Model (if enabled)

### ❌ **Missing High-Impact Improvements:**

---

## 🚀 **Priority 1: High-Impact, Easy to Implement**

### **1. Learning Rate Scheduling (Warmup + Cosine Annealing)**
**Expected Gain:** +2-4% accuracy  
**Current Status:** Config exists (`ttt_warmup_steps=20`, `ttt_lr_min=4e-5`) but **NOT USED**

**Implementation:**
```python
# Add after optimizer setup
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def lr_lambda(step):
    # Warmup phase
    if step < ttt_config.ttt_warmup_steps:
        return step / ttt_config.ttt_warmup_steps
    # Cosine annealing
    progress = (step - ttt_config.ttt_warmup_steps) / (ttt_steps - ttt_config.ttt_warmup_steps)
    return ttt_config.ttt_lr_min / ttt_lr + (1 - ttt_config.ttt_lr_min / ttt_lr) * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)
# In loop: scheduler.step()
```

**Why:** Prevents overshooting early, smooth convergence later

---

### **2. Early Stopping**
**Expected Gain:** +1-2% accuracy, prevents overfitting  
**Current Status:** Config exists (`ttt_early_stopping=True`, `ttt_early_stopping_patience=15`) but **NOT IMPLEMENTED**

**Implementation:**
```python
best_loss = float('inf')
patience_counter = 0
patience = ttt_config.ttt_early_stopping_patience
min_delta = ttt_config.ttt_early_stopping_min_delta

for step in range(ttt_steps):
    # ... compute loss ...
    
    # Early stopping check
    if ttt_config.ttt_early_stopping:
        if total_loss.item() < best_loss - min_delta:
            best_loss = total_loss.item()
            patience_counter = 0
            # Save best model state
            best_model_state = copy.deepcopy(adapted_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at step {step+1}")
                adapted_model.load_state_dict(best_model_state)
                break
```

**Why:** Prevents overfitting on test data, improves generalization

---

### **3. Batch Processing for Large Query Sets**
**Expected Gain:** +1-3% accuracy (better gradient estimates)  
**Current Status:** Uses full query set at once

**Implementation:**
```python
ttt_batch_size = getattr(ttt_config, 'ttt_batch_size', 64)
# In loop:
for batch_idx in range(0, len(query_x), ttt_batch_size):
    batch_x = query_x[batch_idx:batch_idx + ttt_batch_size]
    # ... compute loss on batch ...
```

**Why:** Better gradient estimates, more stable training

---

### **4. Sharpened Pseudo-Labels with Temperature**
**Expected Gain:** +1-2% accuracy  
**Current Status:** Uses raw probabilities

**Implementation:**
```python
# Sharpen pseudo-labels with temperature
pseudo_label_temperature = getattr(ttt_config, 'pseudo_label_temperature', 0.33)
sharpened_probs = F.softmax(logits / pseudo_label_temperature, dim=1)
confidences, pseudo_labels = sharpened_probs.max(dim=1)
```

**Why:** Sharper pseudo-labels = more confident supervision signal

---

## 🚀 **Priority 2: Medium-Impact, Moderate Complexity**

### **5. Adaptive Learning Rate Based on Loss**
**Expected Gain:** +1-2% accuracy  
**Implementation:**
```python
# Reduce LR if loss plateaus
if step > 0 and total_loss.item() > prev_loss * 1.01:  # Loss increased
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.95  # Reduce LR by 5%
```

**Why:** Automatic LR adjustment prevents overshooting

---

### **6. Consistency Loss (Test-Time Augmentation)**
**Expected Gain:** +2-3% accuracy  
**Current Status:** Config exists (`consistency_weight=0.3`) but **NOT USED**

**Implementation:**
```python
# Add noise augmentation
jitter_sigma = getattr(ttt_config, 'jitter_sigma', 0.10)
augmented_x = query_x + torch.randn_like(query_x) * jitter_sigma

# Forward pass on original and augmented
logits_orig = adapted_model(query_x)
logits_aug = adapted_model(augmented_x)

# Consistency loss (KL divergence)
consistency_loss = F.kl_div(
    F.log_softmax(logits_aug / ttt_config.ttt_temperature, dim=1),
    F.softmax(logits_orig / ttt_config.ttt_temperature, dim=1),
    reduction='batchmean'
)

total_loss = entropy_loss + pseudo_loss + consistency_weight * consistency_loss
```

**Why:** Improves robustness to test-time distribution shifts

---

### **7. Curriculum Learning for Pseudo-Labels**
**Expected Gain:** +1-2% accuracy  
**Implementation:**
```python
# Start with high threshold, gradually lower
current_threshold = pseudo_threshold - (pseudo_threshold - pseudo_min_threshold) * (step / ttt_steps)
confident_mask = confidences > current_threshold
```

**Why:** Gradually includes more samples as model adapts

---

### **8. Momentum-Based Pseudo-Label Updates**
**Expected Gain:** +1-2% accuracy  
**Implementation:**
```python
# Maintain running average of pseudo-labels
if step == 0:
    pseudo_label_momentum = torch.zeros_like(pseudo_labels)
else:
    momentum = 0.9
    pseudo_label_momentum = momentum * pseudo_label_momentum + (1 - momentum) * pseudo_labels
```

**Why:** Smoother pseudo-label updates = more stable training

---

## 🚀 **Priority 3: Advanced Techniques**

### **9. Multiple Forward Passes (Monte Carlo Dropout)**
**Expected Gain:** +1-2% accuracy, better uncertainty estimation  
**Implementation:**
```python
# Multiple forward passes with dropout
n_samples = 5
logits_samples = []
for _ in range(n_samples):
    logits_samples.append(adapted_model(query_x))
logits = torch.stack(logits_samples).mean(dim=0)
uncertainty = torch.stack(logits_samples).std(dim=0).mean()
```

**Why:** Better uncertainty estimation, more robust predictions

---

### **10. Prototype-Based Adaptation**
**Expected Gain:** +2-4% accuracy  
**Current Status:** Config exists but not fully implemented

**Implementation:**
```python
# Compute prototypes from confident predictions
confident_embeddings = model.get_embeddings(query_x[confident_mask])
prototypes = confident_embeddings.mean(dim=0, keepdim=True)

# Prototype alignment loss
current_embeddings = model.get_embeddings(query_x)
prototype_loss = F.mse_loss(current_embeddings, prototypes.expand_as(current_embeddings))
```

**Why:** Better adaptation to test distribution

---

## 📈 **Expected Combined Improvement**

| Priority | Technique | Expected Gain | Complexity |
|----------|-----------|---------------|------------|
| P1 | LR Scheduling | +2-4% | Low |
| P1 | Early Stopping | +1-2% | Low |
| P1 | Batch Processing | +1-3% | Low |
| P1 | Sharpened Pseudo-Labels | +1-2% | Low |
| P2 | Adaptive LR | +1-2% | Medium |
| P2 | Consistency Loss | +2-3% | Medium |
| P2 | Curriculum Learning | +1-2% | Medium |
| P2 | Momentum Pseudo-Labels | +1-2% | Medium |
| P3 | Monte Carlo Dropout | +1-2% | High |
| P3 | Prototype Adaptation | +2-4% | High |

**Total Potential Gain:** +13-24% accuracy improvement

---

## 🎯 **Recommended Implementation Order**

### **Phase 1 (Quick Wins - 1-2 hours):**
1. ✅ Learning Rate Scheduling
2. ✅ Early Stopping
3. ✅ Batch Processing
4. ✅ Sharpened Pseudo-Labels

**Expected Gain:** +5-11% accuracy

### **Phase 2 (Medium Effort - 2-3 hours):**
5. ✅ Consistency Loss
6. ✅ Curriculum Learning
7. ✅ Adaptive Learning Rate

**Expected Gain:** +4-7% additional accuracy

### **Phase 3 (Advanced - 3-4 hours):**
8. ✅ Momentum Pseudo-Labels
9. ✅ Monte Carlo Dropout
10. ✅ Prototype-Based Adaptation

**Expected Gain:** +4-6% additional accuracy

---

## 💡 **Quick Implementation Checklist**

- [ ] Add LR scheduler with warmup + cosine annealing
- [ ] Implement early stopping with patience
- [ ] Add batch processing for query set
- [ ] Implement sharpened pseudo-labels with temperature
- [ ] Add consistency loss with test-time augmentation
- [ ] Implement curriculum learning for pseudo-label threshold
- [ ] Add adaptive learning rate adjustment
- [ ] Implement momentum-based pseudo-label updates
- [ ] Add Monte Carlo dropout for uncertainty
- [ ] Implement prototype-based adaptation

---

## 🔍 **Current Bottlenecks**

1. **No LR Scheduling:** Fixed LR throughout = suboptimal convergence
2. **No Early Stopping:** Risk of overfitting on test data
3. **Full Batch Processing:** May cause memory issues, less stable gradients
4. **Raw Pseudo-Labels:** Not sharpened = weaker supervision signal
5. **No Consistency Loss:** Missing robustness to distribution shifts

---

## 📝 **Next Steps**

1. **Implement Priority 1 techniques** (highest ROI)
2. **Re-run optimization** with new techniques
3. **Compare results** before/after
4. **Iterate** on Priority 2 if needed

**Estimated Time:** 2-3 hours for Priority 1, +5-11% accuracy gain







