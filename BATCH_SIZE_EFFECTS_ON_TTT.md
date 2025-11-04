# Batch Size Effects on Test-Time Training (TTT) Performance

## Overview

Batch size is a critical hyperparameter in TTT adaptation that affects:

1. **Gradient Estimation Quality**
2. **Convergence Speed and Stability**
3. **Diversity Metrics Accuracy**
4. **Memory Usage**
5. **Computational Efficiency**

---

## Current Implementation

**Current Batch Size**: `32` (hardcoded in `_perform_advanced_ttt_adaptation`)

**Query Set Size**: `750` samples (from `ttt_adaptation_query_size`)

**Number of Batches**: `n_batches = ceil(750 / 32) = 24 batches per step`

**Total Gradient Updates**: `24 batches × 50 steps = 1,200 gradient updates`

---

## Effects of Batch Size on TTT Performance

### 1. **Gradient Estimation Quality**

#### **Small Batch Size (< 16)**:

- ✅ **Pros**:
  - More gradient updates per step (better exploration)
  - Faster adaptation to local data distribution
  - Better for small datasets
- ❌ **Cons**:
  - **High variance** in gradient estimates (noisy gradients)
  - Less stable optimization
  - May lead to slower convergence or oscillations
  - More sensitive to outliers

**Impact on Your System**:

- With batch size 8: `750 / 8 = 94 batches` → More updates but higher variance
- Gradient norm fluctuations: Higher variance → Less predictable convergence

#### **Large Batch Size (> 64)**:

- ✅ **Pros**:

  - **Lower variance** in gradient estimates (smoother gradients)
  - More stable optimization
  - Better gradient direction estimation
  - Faster convergence in terms of steps needed

- ❌ **Cons**:
  - Fewer gradient updates per step (less exploration)
  - May miss fine-grained data distribution details
  - Higher memory requirements
  - May oversmooth and miss important local patterns

**Impact on Your System**:

- With batch size 128: `750 / 128 = 6 batches` → Fewer updates but more stable
- Gradient norm: More stable → Better convergence behavior

#### **Medium Batch Size (16-64)**:

- ✅ **Balanced trade-off**:
  - Reasonable variance (not too noisy, not too smooth)
  - Good exploration-exploitation balance
  - **Current choice (32) is in this range** ✅

---

### 2. **Convergence Speed and Stability**

#### **Relationship Between Batch Size and Convergence**:

```
Small Batch (8-16):
  - More gradient updates per step → Faster adaptation
  - But higher variance → Slower overall convergence
  - Gradient norm: High fluctuations

Medium Batch (32-64):
  - Balanced updates and stability
  - Optimal convergence speed
  - Gradient norm: Moderate fluctuations

Large Batch (128+):
  - Fewer gradient updates → Slower adaptation
  - But lower variance → Faster convergence per step
  - Gradient norm: Low fluctuations
```

#### **Effect on Your Current Gradient Norm Issue**:

**Current Problem**: Gradient norm is **INCREASING** (+5.31% from early to late)

**Hypothesis**: Batch size 32 may be contributing to:

- Insufficient gradient smoothing → High variance → Erratic optimization
- Too many small gradient updates → May accumulate in wrong direction

**Potential Solutions**:

1. **Increase batch size to 64-128**:

   - More stable gradients → Better convergence direction
   - May reduce gradient norm fluctuations
   - Better gradient norm convergence

2. **Decrease batch size to 16-24**:
   - More updates per step → Better exploration
   - May help escape local optima
   - But may increase variance (risky)

**Recommendation**: Try **batch size 64** first (2x current) to test if gradient norm stabilizes.

---

### 3. **Diversity Metrics Accuracy**

#### **How Batch Size Affects Diversity Calculation**:

In your code, diversity is computed per batch:

```python
class_distribution = probs.mean(dim=0)  # Average probability per class
class_entropy = -torch.sum(class_distribution * torch.log(class_distribution + 1e-8))
```

**Small Batch Size**:

- ❌ **Less accurate** class distribution estimation
- Higher variance in diversity metrics
- May incorrectly trigger early stopping
- May overestimate or underestimate diversity

**Large Batch Size**:

- ✅ **More accurate** class distribution estimation
- Lower variance in diversity metrics
- More reliable diversity thresholds
- Better early stopping decisions

**Current Implementation**:

- With batch size 32: Each batch has ~32 samples
- Class distribution estimated from 32 samples → Moderate accuracy
- With batch size 64: Each batch has ~64 samples → Better accuracy

**Impact**: Better diversity metrics → Better adaptive diversity weight → Better convergence

---

### 4. **Memory Usage**

#### **Memory Requirements**:

```
Memory per batch = batch_size × sequence_length × input_dim × dtype_size

Current (batch_size=32):
  Memory = 32 × 30 × 23 × 4 bytes = ~88 KB per batch

With batch_size=64:
  Memory = 64 × 30 × 23 × 4 bytes = ~177 KB per batch

With batch_size=128:
  Memory = 128 × 30 × 23 × 4 bytes = ~354 KB per batch
```

**Impact**:

- Current batch size 32 is **memory-efficient** ✅
- Can easily increase to 64 or 128 without memory issues
- GPU memory should handle 128 easily

---

### 5. **Computational Efficiency**

#### **Time per Step**:

```
Time per step = n_batches × time_per_batch

Current (batch_size=32, n_batches=24):
  Time per step = 24 batches × ~0.5s = ~12s per step

With batch_size=64 (n_batches=12):
  Time per step = 12 batches × ~0.5s = ~6s per step (2x faster!)

With batch_size=128 (n_batches=6):
  Time per step = 6 batches × ~0.5s = ~3s per step (4x faster!)
```

**Impact**:

- **Larger batch size = Faster training** (fewer batches to process)
- But need to balance with gradient quality

---

### 6. **Effect on Dual Objectives (Entropy + Diversity)**

#### **Batch Size and Loss Component Balance**:

**Small Batch**:

- Entropy loss: High variance → Unstable
- Diversity loss: Less accurate → Unstable
- **Result**: Competing objectives harder to balance → Gradient norm increases

**Large Batch**:

- Entropy loss: Lower variance → More stable
- Diversity loss: More accurate → More stable
- **Result**: Better balance → Smoother gradient norm decrease

**Current Issue** (Gradient norm increasing):

- May be due to batch size 32 causing unstable loss component estimation
- Increasing to 64-128 may help stabilize both components

---

## Recommendations for Your System

### **Recommendation 1: Increase Batch Size to 64** ⭐ **RECOMMENDED**

**Rationale**:

- Better gradient stability → May help gradient norm decrease
- More accurate diversity metrics → Better adaptive diversity weight
- Faster training (2x fewer batches)
- Still memory-efficient

**Expected Effects**:

- ✅ Gradient norm: More stable, may decrease instead of increase
- ✅ Convergence: Faster per step, better overall
- ✅ Diversity metrics: More accurate
- ✅ Training time: ~2x faster

**Implementation**:

```python
# In coordinators/simple_fedavg_coordinator.py
batch_size = 64  # Changed from 32
```

### **Recommendation 2: Test Batch Size 128**

**Rationale**:

- Maximum gradient stability
- Best diversity metrics accuracy
- Fastest training (4x fewer batches)

**Expected Effects**:

- ✅ Gradient norm: Very stable, likely to decrease
- ⚠️ May oversmooth and miss local patterns
- ✅ Training time: ~4x faster

**Trade-off**: May reduce adaptation quality if data distribution is very local

### **Recommendation 3: Adaptive Batch Size**

**Rationale**:

- Start with small batch (exploration) → Increase to large batch (exploitation)
- Best of both worlds

**Implementation**:

```python
# Adaptive batch size based on step
initial_batch_size = 16
final_batch_size = 64
batch_size = initial_batch_size + (final_batch_size - initial_batch_size) * (step / ttt_steps)
```

---

## Experimental Testing Plan

### **Test 1: Batch Size 64**

1. Change `batch_size = 64` in `_perform_advanced_ttt_adaptation`
2. Run system and observe:
   - Gradient norm trend (should decrease instead of increase)
   - Loss convergence (should be faster)
   - Final performance (should maintain or improve)

### **Test 2: Batch Size 128**

1. Change `batch_size = 128`
2. Compare with batch size 64:
   - Gradient norm stability
   - Convergence speed
   - Final performance

### **Test 3: Compare Gradient Norm Behavior**

**Metrics to Track**:

- Gradient norm trend (early vs. late)
- Gradient norm variance (std dev across steps)
- Loss convergence rate
- Final gradient norm value

**Expected Results**:

- Batch size 64: Gradient norm should decrease or stabilize
- Batch size 128: Gradient norm should decrease more smoothly

---

## Summary

### **Current Situation**:

- Batch size: 32
- Gradient norm: **INCREASING** (+5.31%)
- Convergence: Partial (loss decreasing, but gradient norm not)

### **Expected Impact of Increasing Batch Size**:

| Batch Size | Gradient Stability  | Convergence Speed | Memory  | Recommendation                                         |
| ---------- | ------------------- | ----------------- | ------- | ------------------------------------------------------ |
| 16         | Low (high variance) | Slow              | Low     | ❌ Not recommended                                     |
| **32**     | **Medium**          | **Medium**        | **Low** | ⚠️ **Current (may be causing gradient norm increase)** |
| **64**     | **High**            | **Fast**          | **Low** | ✅ **RECOMMENDED**                                     |
| 128        | Very High           | Very Fast         | Medium  | ✅ **Test if 64 doesn't work**                         |

### **Key Takeaway**:

**Batch size 32 may be contributing to your gradient norm increasing issue**. Increasing to **64 or 128** should:

1. Provide more stable gradients
2. Help gradient norm decrease instead of increase
3. Improve convergence behavior
4. Maintain or improve performance

**Next Step**: Test batch size 64 and observe if gradient norm trend reverses (decreasing instead of increasing).

---

## References

1. **Gradient Estimation**:

   - Small batch: High variance → Noisy gradients
   - Large batch: Low variance → Smooth gradients

2. **Convergence Theory**:

   - Batch size affects gradient variance → Affects convergence rate
   - Optimal batch size balances exploration vs. exploitation

3. **Your Current Issue**:
   - Gradient norm increasing → May be due to high variance from batch size 32
   - Solution: Increase batch size to reduce variance → Stabilize gradient norm
