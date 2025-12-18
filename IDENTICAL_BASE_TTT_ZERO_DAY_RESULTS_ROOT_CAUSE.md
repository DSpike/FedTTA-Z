# Root Cause: Why Base and TTT Models Show Identical Zero-Day Results

## 🔍 **Problem Statement**

Both base model and TTT model report **identical results** for zero-day attack detection (BruteForce), which is unexpected since TTT should adapt and improve performance.

---

## 🐛 **CRITICAL BUG #1: Model Mismatch**

### **The Issue:**

**In `_evaluate_ttt_model()` (main.py lines 6309-6387):**

1. **Base Predictions** (line 6361):

   ```python
   binary_model = TransductiveLearner(...)  # NEW binary model created
   base_logits = binary_model.forward_with_prototypes(query_x, base_prototypes)
   base_predictions = torch.argmax(base_logits, dim=1)
   ```

   - Uses a **NEW binary_model** (2 classes)
   - Created fresh, weights copied from `coordinator.model`
   - **NOT the same model** used in base evaluation!

2. **TTT Adaptation** (line 6382):

   ```python
   adapted_model = self.coordinator.adapt_to_test_data(...)
   ```

   - Adapts `coordinator.model` (multiclass, 10 classes)
   - Returns **adapted multiclass model**
   - **NOT the binary_model!**

3. **TTT Predictions** (line 6413):
   ```python
   initial_logits = adapted_model(query_x)  # Uses adapted MULTICLASS model
   ```
   - Uses `adapted_model` (multiclass, after TTT)
   - Converts to binary: `(predictions != 0)`

### **The Problem:**

- **Base predictions**: From `binary_model` (binary, no TTT)
- **TTT predictions**: From `adapted_model` (multiclass, after TTT, converted to binary)
- **These are DIFFERENT models!** Not comparable!

### **Why Results Are Identical:**

1. **Binary conversion masks differences**: Both use `(predictions != 0)`, so even if multiclass predictions differ, binary might be same
2. **Binary model might produce similar results**: The binary_model is created from coordinator.model, so initial predictions might be similar
3. **TTT might not change predictions significantly**: If TTT doesn't improve zero-day detection, results stay the same

---

## 🐛 **CRITICAL BUG #2: TTT Adapts Wrong Model**

### **The Issue:**

**TTT adaptation adapts `coordinator.model` (multiclass), but base predictions use `binary_model` (binary).**

**Flow:**

1. `binary_model` created (line 6312) - **NOT adapted**
2. Base predictions from `binary_model` (line 6361)
3. TTT adapts `coordinator.model` (line 6382) - **DIFFERENT model**
4. TTT predictions from `adapted_model` (line 6413) - **DIFFERENT model**

**Result:** Comparing apples to oranges!

---

## 🔍 **Why Results Are Identical (Possible Reasons)**

### **Reason 1: Binary Conversion Hides Differences**

Both models convert to binary using `(predictions != 0)`:

- Base: `(binary_model_predictions != 0)`
- TTT: `(adapted_model_predictions != 0)`

**If both models predict:**

- Normal (0) → Binary: 0
- Any attack (1-9) → Binary: 1

**Even if multiclass predictions differ (e.g., base predicts class 1, TTT predicts class 5), binary conversion makes them identical (both = 1)!**

### **Reason 2: TTT Doesn't Change Zero-Day Predictions**

**TTT Optimization:**

- Uses entropy minimization: `entropy_loss = -(probs * log(probs)).mean()`
- Optimizes for **overall confidence**, not zero-day specific
- Zero-day samples: ~30% of adaptation set
- Non-zero-day samples: ~70% of adaptation set

**Gradient:**

```
∇L ≈ 0.3 * ∇L_zero_day + 0.7 * ∇L_non_zero_day
```

**Result:** Optimization is **dominated by non-zero-day samples**, so zero-day predictions might not change!

### **Reason 3: Base Model Already Good**

If base model already detects zero-day attacks well:

- Base: 90% detection rate
- TTT: 90% detection rate (no improvement)

**Why no improvement?**

- Only 10% of zero-day samples are misclassified
- These 10% are only 3% of total adaptation set (10% of 30%)
- TTT gradient has minimal influence from these samples
- TTT doesn't fix them

### **Reason 4: Same Threshold Used**

Both models might use the same optimal threshold:

- Base: Threshold = 0.5 (or optimized)
- TTT: Threshold = 0.5 (or same optimized value)

**If probabilities are similar and threshold is same, binary predictions will be identical!**

---

## 🔧 **How to Verify the Root Cause**

### **Step 1: Check if Predictions Are Actually Identical**

Add diagnostic logging:

```python
# After generating base and TTT predictions
prediction_diff = (base_predictions_np != ttt_predictions_np).sum()
logger.info(f"🔍 Prediction differences: {prediction_diff}/{len(base_predictions_np)} samples differ")

# Check zero-day specific
zero_day_base = base_predictions_np[is_zero_day_np]
zero_day_ttt = ttt_predictions_np[is_zero_day_np]
zero_day_diff = (zero_day_base != zero_day_ttt).sum()
logger.info(f"🔍 Zero-day prediction differences: {zero_day_diff}/{len(zero_day_base)} samples differ")
```

### **Step 2: Check if Models Are Different**

Add diagnostic logging:

```python
# Check if adapted_model is different from base model
base_model_params = sum(p.numel() for p in binary_model.parameters())
adapted_model_params = sum(p.numel() for p in adapted_model.parameters())
logger.info(f"🔍 Base model params: {base_model_params}, Adapted model params: {adapted_model_params}")

# Check parameter differences
param_diff = 0
for (name1, p1), (name2, p2) in zip(binary_model.named_parameters(), adapted_model.named_parameters()):
    if p1.shape == p2.shape:
        param_diff += (p1 != p2).sum().item()
logger.info(f"🔍 Parameter differences: {param_diff} parameters differ")
```

### **Step 3: Check if TTT Actually Adapted**

Add diagnostic logging:

```python
# Before TTT
base_confidence = base_attack_probs.mean().item()
logger.info(f"🔍 Base model confidence: {base_confidence:.4f}")

# After TTT
ttt_confidence = attack_probabilities.mean().item()
logger.info(f"🔍 TTT model confidence: {ttt_confidence:.4f}")

# Check if TTT changed anything
if abs(base_confidence - ttt_confidence) < 1e-6:
    logger.warning("⚠️ TTT confidence is IDENTICAL to base - TTT might not have adapted!")
```

---

## ✅ **Recommended Fixes**

### **Fix 1: Use Same Model for Base and TTT Predictions** (CRITICAL)

**Option A: Adapt binary_model for TTT**

```python
# Instead of adapting coordinator.model, adapt binary_model
adapted_binary_model = self.coordinator.adapt_to_test_data(
    query_x=query_x,
    query_y=None,
    config=self.config,
    method=method,
    model=binary_model  # Adapt the binary model, not coordinator.model
)
```

**Option B: Use coordinator.model for base predictions**

```python
# Use coordinator.model for base predictions (same as TTT)
base_logits = self.coordinator.model.forward_with_prototypes(query_x, base_prototypes)
```

### **Fix 2: Add Prediction Comparison Logging**

Add diagnostic logging to verify predictions are actually different:

```python
# After generating predictions
prediction_similarity = (base_predictions_np == ttt_predictions_np).mean()
logger.info(f"🔍 Prediction similarity: {prediction_similarity:.4f} ({prediction_similarity*100:.1f}% identical)")

if prediction_similarity > 0.99:
    logger.warning("⚠️ CRITICAL: Base and TTT predictions are >99% identical!")
    logger.warning("   This suggests TTT is not changing predictions significantly")
```

### **Fix 3: Zero-Day Weighted TTT** (From Previous Analysis)

Modify TTT loss to weight zero-day samples:

```python
# In adapt_to_test_data()
zero_day_weights = torch.ones(len(query_x), device=query_x.device)
zero_day_weights[zero_day_mask] = 3.0  # 3x weight for zero-day

weighted_entropy_loss = (entropy * zero_day_weights).mean()
```

---

## 📊 **Expected Results After Fix**

### **Before Fix:**

- Base and TTT predictions: **Identical** (or very similar)
- Zero-day metrics: **Same** for both models
- **Root cause**: Different models or TTT not adapting zero-day samples

### **After Fix:**

- Base and TTT predictions: **Different** (TTT should improve)
- Zero-day metrics: **TTT should be better** than base
- **Root cause fixed**: Same model used, TTT properly adapts

---

## 🎯 **Next Steps**

1. **Add diagnostic logging** to verify predictions are actually identical
2. **Fix model consistency** - use same model for base and TTT predictions
3. **Implement zero-day weighted TTT** to prioritize zero-day samples
4. **Re-run evaluation** and verify TTT improves zero-day detection
