# TTT Performance Drop Investigation

## 🔍 **Potential Root Causes**

### **1. Center Loss Weight Increase (RECENT CHANGE)** ⚠️ **LIKELY CULPRIT**

**What Changed:**

- `center_loss_weight`: 0.08 → 0.5 (6.25x increase)
- Effective weight in loss: 0.05 \* 0.5 = 0.025 (2.5%)

**Impact on TTT:**

- **Tighter embeddings**: Center loss pulls embeddings closer to class centers
- **Less adaptable**: Tighter clusters = less flexibility for TTT to adapt
- **BatchNorm adaptation limited**: TENT only updates BatchNorm, which may not be enough to shift tightly clustered embeddings
- **Distribution mismatch**: Base model embeddings are now very compact, TTT can't easily shift them

**Evidence:**

- Center loss creates very tight clusters (good for base model)
- But TTT needs flexibility to adapt embeddings
- TENT-style (BatchNorm only) may not be sufficient to shift tightly clustered embeddings

---

### **2. TENT-Style Adaptation Too Restrictive**

**Current Implementation:**

- Only BatchNorm affine parameters are updated
- All other parameters (TCN, projections, prototypes) are frozen

**Problem:**

- If embeddings are tightly clustered (due to high center_loss_weight), BatchNorm updates alone may not be enough
- Need to update projection layers or prototypes to shift embeddings

**Solution Options:**

1. Unfreeze projection layers in addition to BatchNorm
2. Reduce center_loss_weight back to 0.08-0.15 range
3. Use full model adaptation (not just BatchNorm) for TTT

---

### **3. Loss Weight Changes Affecting Base Model**

**Recent Changes:**

- Loss weights now sum to 1.0 (fixed)
- Center loss: 0.5 weight (very high)
- Margin loss: 0.25 weight

**Impact:**

- Base model embeddings are now very different (tighter, more separated)
- TTT was optimized for previous embedding distribution
- TTT hyperparameters may not work with new embedding distribution

---

### **4. TTT Hyperparameters Mismatch**

**Current TTT Config:**

```python
ttt_lr: float = 0.002  # May be too high for tightly clustered embeddings
ttt_base_steps: int = 70  # May be too few for adaptation
ttt_l2_reg_weight: float = 0.01  # Regularization
pseudo_weight: float = 1.5
entropy_weight: float = 0.8
```

**Issues:**

- Learning rate (0.002) might be too high for tightly clustered embeddings
- Steps (70) might be insufficient
- L2 regularization might be preventing necessary adaptation

---

## 🎯 **Recommended Fixes**

### **Fix #1: Reduce Center Loss Weight (IMMEDIATE)**

**Problem**: Center loss weight (0.5) is too high, making embeddings too tight for TTT adaptation

**Solution**:

```python
center_loss_weight: float = 0.15  # Balance: tight enough for base model, loose enough for TTT
```

**Rationale**:

- 0.15 is 3x higher than original (0.05) but 3x lower than current (0.5)
- Still provides clustering benefit for base model
- Allows TTT to adapt embeddings more easily

---

### **Fix #2: Adjust TTT Learning Rate**

**Problem**: TTT learning rate might be too high for tightly clustered embeddings

**Solution**:

```python
ttt_lr: float = 0.001  # Reduce from 0.002 (50% reduction)
```

**Rationale**: Lower learning rate prevents overshooting when embeddings are tightly clustered

---

### **Fix #3: Increase TTT Steps**

**Problem**: 70 steps might be insufficient for adaptation

**Solution**:

```python
ttt_base_steps: int = 100  # Increase from 70 (43% more)
```

**Rationale**: More steps allow gradual adaptation without overshooting

---

### **Fix #4: Unfreeze Projection Layers (ADVANCED)**

**Problem**: BatchNorm-only adaptation may not be enough

**Solution**: Unfreeze projection layers in addition to BatchNorm:

```python
# In adapt_test_time(), after BatchNorm unfreezing:
# Also unfreeze projection layers
for name, module in adapted_model.named_modules():
    if 'projection' in name.lower() or 'fc' in name.lower():
        for param in module.parameters():
            param.requires_grad = True
            params_to_update.append(param)
```

**Rationale**: Allows TTT to shift embeddings more effectively

---

## 📊 **Priority Order**

1. **Fix #1** (Reduce center_loss_weight) - **HIGHEST PRIORITY**
2. **Fix #2** (Reduce TTT learning rate) - **HIGH PRIORITY**
3. **Fix #3** (Increase TTT steps) - **MEDIUM PRIORITY**
4. **Fix #4** (Unfreeze projections) - **LOW PRIORITY** (try if others don't work)

---

## 🔬 **Diagnostic Steps**

1. **Check embedding separability before/after TTT**
2. **Monitor TTT loss during adaptation** - is it decreasing?
3. **Compare base vs TTT embeddings** - are they too similar?
4. **Check if TTT is actually adapting** - are BatchNorm parameters changing?



