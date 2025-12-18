# TTT Loss Oscillation Investigation

## 🔍 **Problem: Excessive Zigzag/Oscillation in TTT Total Loss**

The TTT total loss shows too much oscillation (zigzag pattern) during adaptation, indicating training instability.

---

## 🎯 **Root Causes**

### **1. Learning Rate Too High** ⚠️

**Current Setting**:
```python
'ttt_lr': 0.001  # AdamW optimizer
```

**Problem**:
- **0.001 is relatively high** for test-time adaptation
- Large parameter updates cause **overshooting**
- Model "bounces" around optimal point instead of converging smoothly

**Evidence**:
- Oscillation pattern suggests **overshooting** → **correction** → **overshooting**
- Loss decreases then increases repeatedly

---

### **2. No Learning Rate Scheduling** ❌

**Current Implementation**:
```python
optimizer = torch.optim.AdamW(params_to_update, lr=ttt_lr, weight_decay=1e-4)
# No scheduler - fixed learning rate throughout
```

**Problem**:
- **Fixed learning rate** throughout all 80 steps
- Should **decay** as adaptation progresses
- Early steps need higher LR, later steps need lower LR

**Impact**:
- Early steps: Large updates cause oscillation
- Later steps: Still using high LR when should be fine-tuning

---

### **3. Gradient Clipping Too Permissive** ⚠️

**Current Setting**:
```python
torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
```

**Problem**:
- **max_norm=1.0** is relatively high
- Allows large gradient steps
- Combined with high LR, causes overshooting

---

### **4. Batch Processing - All Samples at Once** ⚠️

**Current Implementation**:
```python
logits = adapted_model.forward_with_prototypes(query_x, prototypes_ttt)
# Processes ALL query samples at once
```

**Problem**:
- **No mini-batching** - all samples processed together
- Large batch → **noisy gradients** → oscillation
- Different samples may have conflicting gradients

---

### **5. Loss Component Competition** ⚠️

**Current Loss**:
```python
total_loss = entropy_weight * entropy_loss + ttt_l2_reg_weight * l2_reg
# entropy_weight = 1.0
# ttt_l2_reg_weight = 0.01
```

**Problem**:
- **Entropy loss**: Wants to change parameters (reduce uncertainty)
- **L2 reg loss**: Wants to keep parameters close to original
- **Competing objectives** → oscillation as model tries to balance both

**Mathematical Explanation**:
```
∇L = ∇L_entropy + λ * ∇L_l2
   = (wants to change) + 0.01 * (wants to stay same)
```

When entropy gradient is large, L2 reg tries to pull back → **oscillation**

---

### **6. Zero-Day Weighting Amplifies Instability** ⚠️

**Current Setting**:
```python
'ttt_zero_day_weight': 3.0  # 3x weight for zero-day samples
```

**Problem**:
- **3x weighting** amplifies gradient for zero-day samples
- If zero-day samples have **high variance**, this amplifies oscillation
- **30% of samples** get 3x weight → **90% gradient influence** from potentially noisy samples

---

### **7. No Gradient Smoothing/Momentum** ❌

**Current Optimizer**:
```python
optimizer = torch.optim.AdamW(params_to_update, lr=ttt_lr, weight_decay=1e-4)
# Default AdamW settings (beta1=0.9, beta2=0.999)
```

**Problem**:
- **AdamW has momentum**, but might not be enough for TTT
- **No exponential moving average** of gradients
- **No gradient smoothing** to reduce noise

---

## 🔧 **Solutions**

### **Solution 1: Reduce Learning Rate** ⭐⭐⭐⭐⭐

**Change**:
```python
'ttt_lr': 0.0005,  # Reduce from 0.001 to 0.0005 (50% reduction)
```

**Expected Impact**:
- **Smaller parameter updates** → less overshooting
- **Smoother convergence** → reduced oscillation
- **Trade-off**: Slower adaptation (may need more steps)

---

### **Solution 2: Add Learning Rate Scheduler** ⭐⭐⭐⭐⭐

**Implementation**:
```python
# In adapt_to_test_data method
optimizer = torch.optim.AdamW(params_to_update, lr=ttt_lr, weight_decay=1e-4)

# Add scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ttt_steps, eta_min=ttt_lr * 0.1
)

# In training loop
for step in range(ttt_steps):
    # ... forward pass, loss calculation ...
    optimizer.step()
    scheduler.step()  # Decay learning rate
```

**Expected Impact**:
- **High LR early**: Fast initial adaptation
- **Low LR later**: Fine-tuning without oscillation
- **Smooth decay**: Cosine annealing provides smooth transition

---

### **Solution 3: Reduce Gradient Clipping** ⭐⭐⭐⭐

**Change**:
```python
torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=0.5)  # Reduce from 1.0 to 0.5
```

**Expected Impact**:
- **Smaller gradient steps** → less overshooting
- **More stable updates** → reduced oscillation

---

### **Solution 4: Add Mini-Batching** ⭐⭐⭐⭐

**Implementation**:
```python
# Process in mini-batches instead of all at once
batch_size = 32  # Or get from config
for step in range(ttt_steps):
    optimizer.zero_grad()
    
    # Mini-batch processing
    total_loss = 0
    num_batches = 0
    for i in range(0, len(query_x), batch_size):
        batch_x = query_x[i:i+batch_size]
        # ... forward pass, loss calculation ...
        total_loss += batch_loss
        num_batches += 1
    
    total_loss = total_loss / num_batches  # Average across batches
    total_loss.backward()
    optimizer.step()
```

**Expected Impact**:
- **Smaller batches** → less noisy gradients
- **Averaging** → smoother gradient estimates
- **Reduced oscillation** → more stable training

---

### **Solution 5: Reduce L2 Regularization Weight** ⭐⭐⭐

**Change**:
```python
'ttt_l2_reg_weight': 0.005,  # Reduce from 0.01 to 0.005 (50% reduction)
```

**Expected Impact**:
- **Less competition** between entropy and L2 reg
- **Entropy loss dominates** → smoother optimization
- **Trade-off**: More parameter drift (but L2 still prevents excessive drift)

---

### **Solution 6: Add Gradient Smoothing** ⭐⭐⭐⭐

**Implementation**:
```python
# Use exponential moving average of gradients
grad_momentum = 0.9
grad_ema = {}

for step in range(ttt_steps):
    optimizer.zero_grad()
    # ... forward pass, loss calculation ...
    total_loss.backward()
    
    # Smooth gradients
    for name, param in adapted_model.named_parameters():
        if param.grad is not None:
            if name not in grad_ema:
                grad_ema[name] = param.grad.clone()
            else:
                grad_ema[name] = grad_momentum * grad_ema[name] + (1 - grad_momentum) * param.grad
                param.grad = grad_ema[name]
    
    optimizer.step()
```

**Expected Impact**:
- **Smoother gradients** → reduced noise
- **Less oscillation** → more stable updates

---

### **Solution 7: Reduce Zero-Day Weight (If Causing Issues)** ⭐⭐

**Change**:
```python
'ttt_zero_day_weight': 2.0,  # Reduce from 3.0 to 2.0
```

**Expected Impact**:
- **Less amplification** of zero-day gradients
- **More balanced** optimization
- **Trade-off**: Less focus on zero-day samples

---

## 📊 **Recommended Combination**

### **High Priority (Implement First)**:

1. **Reduce Learning Rate**: `0.001 → 0.0005`
2. **Add Learning Rate Scheduler**: CosineAnnealingLR
3. **Reduce Gradient Clipping**: `1.0 → 0.5`

### **Medium Priority**:

4. **Add Mini-Batching**: Process in batches of 32-64
5. **Reduce L2 Reg Weight**: `0.01 → 0.005` (if oscillation persists)

### **Low Priority (If Still Oscillating)**:

6. **Add Gradient Smoothing**: Exponential moving average
7. **Reduce Zero-Day Weight**: `3.0 → 2.0` (only if zero-day samples are noisy)

---

## 🎯 **Expected Results After Fixes**

### **Before (Oscillating)**:
```
Loss
1.0 |  ●     ●     ●
    |    ●     ●     ●
0.5 |      ●     ●     ●
    |        ●     ●     ●
0.0 |________________________
    0  20  40  60  80  Steps
```

### **After (Smooth)**:
```
Loss
1.0 |●
    |  ●
0.5 |    ●
    |      ●
0.0 |        ●●●●●●●●●
    0  20  40  60  80  Steps
```

---

## ✅ **Implementation Priority**

1. **Quick Fix**: Reduce LR to 0.0005 (1 line change)
2. **Better Fix**: Add LR scheduler (5-10 lines)
3. **Best Fix**: Add LR scheduler + reduce gradient clipping + mini-batching

---

**Document Created**: Investigation of TTT loss oscillation  
**Status**: Ready for implementation

