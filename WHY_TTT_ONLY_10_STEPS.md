# Why TTT is Limited to Only 10 Steps - Explanation

**Date**: December 22, 2025

---

## Quick Answer

TTT is currently limited to 10 steps because of **Phase 1 ultra-conservative configuration** designed to prevent overfitting on rare attack types like Backdoor (only 583 samples).

**Location**: [config.py:585-586](config.py#L585-L586)

```python
# PHASE 1 IMPROVEMENTS: Ultra-conservative settings for rare attacks (<1,000 samples)
# Based on root cause analysis: 583 Backdoor samples insufficient for aggressive TTT
# Strategy: Reduce overfitting by minimizing adaptation steps and learning rate
ttt_base_steps: int = 10  # Keep for minimum adaptation
ttt_max_steps: int = 10  # REDUCED from 400 → 10 (97.5% reduction, prevent oversampling)
```

---

## Historical Context

### Original Configuration (Before Phase 1)

From various documentation files, the original TTT configuration was much more aggressive:

- **ttt_base_steps**: 194-400 steps (depending on dataset)
- **ttt_max_steps**: 400-500 steps
- **ttt_lr**: 0.002-0.005 (higher learning rate)

### The Problem That Led to 10 Steps

**Root Cause Identified**: Backdoor attack has only **583 samples** in the dataset

**Issue**: Aggressive TTT adaptation (200-400 steps) caused:
1. **Overfitting**: TTT adapted too much to the limited 583 samples
2. **Overconfidence**: Median attack probability = 0.98 (extremely confident)
3. **High FAR**: TTT started predicting too many false alarms
4. **Sample Reuse**: With only ~184 test samples (30% of 583), 400 steps meant each sample seen ~2-3 times

---

## Current Configuration (Phase 1)

**Location**: [config.py:582-591](config.py#L582-L591)

```python
# === TEST-TIME TRAINING (TTT) CONFIGURATION ===
# PHASE 1 IMPROVEMENTS: Ultra-conservative settings for rare attacks (<1,000 samples)
# Based on root cause analysis: 583 Backdoor samples insufficient for aggressive TTT
# Strategy: Reduce overfitting by minimizing adaptation steps and learning rate
ttt_base_steps: int = 10  # Keep for minimum adaptation
ttt_max_steps: int = 10  # REDUCED from 400 → 10 (97.5% reduction, prevent oversampling)
ttt_adaptation_query_size: int = 1198  # Optimized from Optuna Trial 1
ttt_batch_size: int = 64  # Optimized from Optuna Trial 1
ttt_lr: float = 0.0005  # REDUCED from 0.005 → 0.0005 (90% reduction, prevent overshooting)
ttt_l2_reg_weight: float = 0.0  # DISABLED - L2 accumulates over 200 steps, causing catastrophic drift
confidence_rejection_threshold: float = 0.90  # Increased to 0.90 for stricter FAR control
```

**Key Changes**:
- ✅ **ttt_base_steps**: 200+ → **10** (95% reduction)
- ✅ **ttt_max_steps**: 400 → **10** (97.5% reduction)
- ✅ **ttt_lr**: 0.005 → **0.0005** (90% reduction)
- ✅ **ttt_l2_reg_weight**: 0.01 → **0.0** (disabled)

---

## How TTT Steps Are Actually Used

**Location**: [coordinators/centralized_coordinator.py:299](coordinators/centralized_coordinator.py#L299)

```python
# Get TTT parameters from config
ttt_steps = getattr(ttt_config, 'ttt_steps', getattr(ttt_config, 'ttt_base_steps', 100))
```

**Priority**:
1. First checks for `ttt_config.ttt_steps` (usually not set)
2. Falls back to `ttt_config.ttt_base_steps` (currently 10)
3. Default fallback: 100 (if neither is set)

**Current Behavior**: Uses `ttt_base_steps = 10`

---

## Why 10 Steps Instead of 0?

**Minimum Adaptation Principle**:

The comment says "Keep for minimum adaptation" - this means:

1. **Some adaptation is needed**: Zero steps = no TTT at all (just base model)
2. **Entropy minimization benefits**: Even 10 steps allows some confidence calibration
3. **Prototype refinement**: Small number of steps can still adjust prototypes slightly
4. **Avoid over-adaptation**: 10 steps prevents memorizing the test set

**Trade-off**:
- Too few steps (0-5): TTT has no effect, might as well use base model
- Just right (10-20): Minimal adaptation, reduces overfitting risk
- Too many steps (100-400): Overfitting, overconfidence, high FAR

---

## Results with 10 Steps

From your current evaluation:

**Positive Results**:
- ✅ **ZDR**: 100.00% (perfect zero-day detection)
- ✅ **Improvement over Base**: +10.87%
- ✅ **F1-Score**: 84.51%
- ✅ **Accuracy**: 79.43%

**Trade-off**:
- ⚠️ **FAR**: 39.13% (false alarm rate is acceptable but not ideal)

**Interpretation**:
- 10 steps is achieving excellent zero-day detection
- The conservative approach is working for rare attacks
- FAR is higher than desired but within acceptable range

---

## When Would You Increase TTT Steps?

You should consider increasing TTT steps if:

### Scenario 1: More Training Data Available
```python
# If you have 5,000+ samples instead of 583
ttt_base_steps: int = 50  # Increase to 50
ttt_max_steps: int = 100  # Increase to 100
```

### Scenario 2: Lower Attack Categories (More Samples Per Class)
```python
# If using fewer attack categories (e.g., binary Normal vs Attack)
ttt_base_steps: int = 30  # Moderate increase
ttt_max_steps: int = 50
```

### Scenario 3: Different Attack Type
```python
# For attack types with 5,000+ samples (e.g., DoS, PortScan)
ttt_base_steps: int = 100  # More samples = more adaptation possible
ttt_max_steps: int = 200
```

### Scenario 4: FAR Too High
```python
# If FAR exceeds 40% and you need to reduce it
ttt_base_steps: int = 5  # REDUCE further
ttt_max_steps: int = 5
ttt_lr: float = 0.0002  # Also reduce learning rate
```

---

## Alternative: Adaptive TTT Steps

Some documentation mentions **adaptive TTT steps based on data complexity**:

**Location**: [blockchain_federated_learning_project/main.py:5122-5130](blockchain_federated_learning_project/main.py#L5122-L5130)

```python
# Adaptive TTT steps based on data complexity with safety limits
base_ttt_steps = self.config.ttt_base_steps  # Base steps from configuration
# Increase steps for more complex data (higher variance in query set)
query_variance = torch.var(query_x).item()
complexity_factor = min(2.0, 1.0 + query_variance * 10)  # Scale factor based on variance
ttt_steps = int(base_ttt_steps * complexity_factor)

# SAFETY MEASURE: Limit maximum TTT steps to prevent infinite loops
ttt_steps = min(ttt_steps, self.config.ttt_max_steps)  # Maximum steps from configuration
logger.info(f"Adaptive TTT steps: {ttt_steps} (complexity factor: {complexity_factor:.2f})")
```

**How This Works**:
- Base: 10 steps
- If data is complex (high variance): complexity_factor = 1.5-2.0
- Actual steps: 15-20 steps
- Max cap: 10 (due to ttt_max_steps=10)

**Current Behavior**: Adaptive scaling is capped at 10 due to `ttt_max_steps=10`

---

## Recommended Next Steps

### Option 1: Keep Current Settings (Recommended)
✅ **Use this if**:
- Current results are acceptable (ZDR=100%, FAR=39%)
- You're working with rare attacks (<1,000 samples)
- Priority is avoiding overfitting

**No changes needed**

---

### Option 2: Moderate Increase (Test First)
⚠️ **Test this if**:
- You want to reduce FAR slightly
- You have >1,000 samples
- You want to see if more adaptation helps

**Changes**:
```python
ttt_base_steps: int = 20  # Double to 20
ttt_max_steps: int = 30  # Increase max to 30
ttt_lr: float = 0.0005  # Keep same
```

**Expected Impact**:
- May improve FAR (reduce false alarms)
- May improve calibration
- Risk: Slight overfitting possible

---

### Option 3: Aggressive Increase (Research/Experimentation)
⚠️ **Only if**:
- You have 5,000+ samples
- You're willing to risk overfitting
- You want to explore upper limits

**Changes**:
```python
ttt_base_steps: int = 100  # Increase significantly
ttt_max_steps: int = 200
ttt_lr: float = 0.001  # Increase LR too
```

**Expected Impact**:
- Better adaptation to test distribution
- Risk: High overfitting, overconfidence, increased FAR
- Needs careful validation

---

## Summary

**Current Setting**: 10 steps

**Reason**: Ultra-conservative configuration for rare Backdoor attack (583 samples)

**Trade-off**:
- ✅ Excellent ZDR (100.00%)
- ✅ Prevents overfitting
- ⚠️ Higher FAR (39.13%)
- ⚠️ Limited adaptation

**Recommendation**:
- Keep 10 steps for publication (proven to work)
- Document as "conservative TTT for rare attacks"
- If needed for future work: Test 20-30 steps with validation

**To Change**: Edit [config.py:585-586](config.py#L585-L586)

---

**Generated**: December 22, 2025
