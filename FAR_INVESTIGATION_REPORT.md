# False Alarm Rate (FAR) Investigation Report - TTT Model

## Current Situation

**Base Model FAR**: 0.046 (4.6%) - TN=103, FP=5  
**TTT Model FAR**: 0.324 (32.4%) - TN=73, FP=35  
**FAR Increase**: +27.8 percentage points (7x worse) ❌

---

## Root Causes Identified

### 1. **Class-Balanced Entropy Loss Biases Toward Attack Class** ⚠️ **CRITICAL**

**Location**: `coordinators/simple_fedavg_coordinator.py:1225-1237`

**Problem**:
```python
# ✅ CLASS-BALANCED LOSS: Weight minority class (Attack) higher
class_weights = 1.0 / (class_distribution + 1e-8)
class_weights = class_weights / class_weights.sum() * len(class_weights)
```

- The entropy loss uses **inverse frequency weighting** that weights the minority class (Attack) **higher**
- This encourages the model to predict attacks more often to reduce weighted entropy
- If the model predicts more attacks, it gets weighted more, creating a positive feedback loop

**Impact**: Model is incentivized to predict "attack" class more frequently during TTT adaptation

---

### 2. **Pseudo-Labeling May Be Biased Toward Attack Class**

**Location**: `coordinators/simple_fedavg_coordinator.py:1010-1064`

**Problems**:
- If the base model initially predicts more attacks, pseudo-labels will be biased toward attacks
- The union mask (`confident_mask | low_uncertainty_mask`) allows more samples to be pseudo-labeled
- If more attack predictions are confident, more attack pseudo-labels are generated

**Impact**: Model learns from biased pseudo-labels that favor attacks

---

### 3. **FAR Penalty Is Too Weak**

**Location**: `coordinators/simple_fedavg_coordinator.py:1245-1248`

**Current Settings**:
- `ttt_far_penalty_weight = 0.12` (relatively weak)
- `ttt_max_attack_ratio = 0.10` (desires max 10% predicted as attacks)

**Problems**:
- Penalty only activates when `predicted_attack_ratio > 0.10`
- But if test set has ~67% attacks (224 attacks / 332 samples), penalty is always active
- The penalty weight (0.12) might be too weak compared to entropy loss weight (0.1) and pseudo-label weight (1.5)

**Impact**: FAR penalty doesn't effectively prevent excessive attack predictions

---

### 4. **Threshold Optimization Doesn't Prioritize FAR**

**Location**: `main.py:2981`

**Problem**:
```python
ttt_optimal_threshold, _, _, _, _ = find_optimal_threshold(
    y_test_binary, attack_probs, method='balanced', band=(0.1, 0.9))
```

- Uses `method='balanced'` which optimizes for **F1-score**, not FAR
- This favors **recall over precision**, leading to more false positives
- The FAR constraint check (lines 3017-3033) might fail to find a suitable threshold

**Impact**: Optimal threshold is chosen to maximize F1, not minimize FAR

---

### 5. **FAR Constraint Mechanism May Not Be Working**

**Location**: `main.py:3017-3033`

**Problem**:
- Constraint search might fail to find a threshold that satisfies both `max_far` and `min_zdr`
- If constraint search fails, it falls back to the F1-optimized threshold
- The search might be too restrictive (requires ZDR improvement while reducing FAR)

**Impact**: FAR constraint is often not applied, leading to high FAR

---

## Evidence from Code

### Confusion Matrix Comparison

**Base Model**:
```
TN=103, FP=5   (Normal: 108 samples)
FN=72, TP=152  (Attack: 224 samples)
FAR = 5/(5+103) = 0.046 (4.6%)
```

**TTT Model**:
```
TN=73, FP=35   (Normal: 108 samples) ← More false positives!
FN=3, TP=221   (Attack: 224 samples) ← Better recall
FAR = 35/(35+73) = 0.324 (32.4%)
```

**Analysis**:
- TTT model correctly reduces false negatives (72 → 3) ✅
- But increases false positives (5 → 35) ❌
- Net result: Better zero-day detection but much worse FAR

---

## Recommendations

### **Priority 1: Fix Class-Balanced Entropy Loss** 🔴 **CRITICAL**

**Option A**: Remove class-balanced weighting from entropy loss
```python
# Remove inverse frequency weighting
entropy_loss = entropy.mean()  # Simple unweighted entropy
```

**Option B**: Use balanced weighting that doesn't bias toward attacks
```python
# Weight by actual class distribution in adaptation data
# or use symmetric weighting
```

**Expected Impact**: -10 to -15% FAR reduction

---

### **Priority 2: Strengthen FAR Penalty** 🟠 **HIGH**

**Changes**:
1. Increase `ttt_far_penalty_weight` from 0.12 → 0.25-0.5
2. Adjust `ttt_max_attack_ratio` based on actual test distribution (currently 0.10, but test has ~67% attacks)

**Code Fix**:
```python
# Make FAR penalty stronger and adaptive to test distribution
far_penalty_weight = 0.3  # Increased from 0.12
max_attack_ratio = 0.7  # Based on actual test distribution (~67% attacks)
```

**Expected Impact**: -5 to -10% FAR reduction

---

### **Priority 3: Change Threshold Optimization Strategy** 🟠 **HIGH**

**Option A**: Optimize threshold to minimize FAR while maintaining ZDR
```python
# Use FAR-minimizing threshold instead of F1-optimized
ttt_optimal_threshold = find_threshold_minimize_far(
    y_test_binary, attack_probs, 
    max_far=base_far * 1.1,  # Allow slight increase
    min_zdr=base_zdr * 0.95  # Allow slight decrease
)
```

**Option B**: Use a higher threshold (e.g., 0.6-0.7) to reduce false positives

**Expected Impact**: -5 to -15% FAR reduction

---

### **Priority 4: Improve FAR Constraint Search** 🟡 **MEDIUM**

**Changes**:
1. Make constraint search more flexible (allow slight ZDR trade-off)
2. Add fallback to conservative threshold if search fails
3. Log why constraint search fails

**Expected Impact**: -3 to -8% FAR reduction

---

### **Priority 5: Add Post-Adaptation Threshold Calibration** 🟡 **MEDIUM**

**Changes**:
- After TTT adaptation, calibrate threshold on a small validation subset
- Use Platt scaling or isotonic regression for better calibration
- Ensure FAR doesn't exceed base model + 2%

**Expected Impact**: -2 to -5% FAR reduction

---

## Implementation Priority

1. **Immediate**: Fix class-balanced entropy loss (Priority 1)
2. **Short-term**: Strengthen FAR penalty + change threshold strategy (Priority 2-3)
3. **Medium-term**: Improve constraint search + add calibration (Priority 4-5)

---

## Expected Results After Fixes

| Fix | FAR Reduction | ZDR Impact |
|-----|--------------|------------|
| Remove class-balanced entropy | -10 to -15% | Minimal (-1 to -2%) |
| Strengthen FAR penalty | -5 to -10% | Minimal (-1%) |
| FAR-optimized threshold | -5 to -15% | Slight (-2 to -3%) |
| **Combined** | **-20 to -30%** | **-3 to -5%** |

**Target FAR**: 0.08-0.15 (still higher than base 0.046, but acceptable trade-off for higher ZDR)

---

## Summary

The high FAR in TTT model is caused by:
1. **Class-balanced entropy loss** biasing predictions toward attacks (main cause)
2. **Weak FAR penalty** during adaptation
3. **F1-optimized threshold** that favors recall over precision
4. **Ineffective FAR constraint** mechanism

**Most Critical Fix**: Remove or fix class-balanced entropy loss weighting

