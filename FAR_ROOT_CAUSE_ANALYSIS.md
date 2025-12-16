# FAR (False Alarm Rate) Root Cause Analysis

## 🔍 Investigation Summary

This document analyzes the root causes of high FAR in the TTT model **without modifying any code**.

---

## 📊 Primary Root Causes (Ranked by Impact)

### 1. **Threshold Optimization Strategy** ⭐⭐⭐⭐⭐ (CRITICAL)

**Location**: `main.py` lines 3940-4127

**Current Behavior**:
- Strategy is configurable via `config.threshold_optimization_strategy`
- Default: `'balanced_zdr_far'` (line 276 in config.py)
- Alternative: `'zdr_optimized'` (prioritizes ZDR over FAR)

**Problem**:
- **`zdr_optimized` strategy** (lines 4052-4122):
  - Searches for threshold that maximizes ZDR (zero-day detection rate)
  - Uses `ttt_zdr_max_far = 0.50` (allows up to 50% FAR!)
  - Threshold range: `np.linspace(0.05, 0.8, 200)` - very low thresholds (0.05-0.3) maximize ZDR
  - **Low threshold = More predictions as Attack = High FAR**

- **`balanced_zdr_far` strategy** (lines 3962-4049):
  - Tries to balance ZDR and FAR
  - Uses `max_far_allowed = 0.20` (line 3972)
  - Score formula: `score = zdr_at_thresh - 2.5 * far_at_thresh` (line 4011)
  - **BUT**: If no threshold satisfies both constraints, it falls back to PR-optimized
  - PR-optimized may still select low thresholds for high recall

**Evidence from Config**:
```python
# config.py line 252-253
ttt_zdr_target: float = 0.85  # Aggressive ZDR target
ttt_zdr_max_far: float = 0.50  # Allows 50% FAR! (very high)

# config.py line 286
max_far_for_zdr: float = 0.35  # Permissive for ZDR optimization
```

**Impact**: 
- **If `zdr_optimized` is used**: FAR can reach 30-50%+
- **If `balanced_zdr_far` fails**: Falls back to PR-optimized, which may still have high FAR
- **Root cause**: Low thresholds (0.05-0.3) maximize ZDR but cause high FAR

---

### 2. **Entropy Minimization Overconfidence** ⭐⭐⭐⭐

**Location**: `coordinators/centralized_coordinator.py` lines 305-307

**Current Behavior**:
```python
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss
```

**Problem**:
- **Entropy minimization** makes predictions more confident (lower entropy)
- If model becomes **overconfident about attacks**, it predicts more attacks
- **No class balancing** in entropy loss (unlike some implementations)
- **No FAR penalty** in loss function
- Model can freely increase false positives during adaptation

**Impact**:
- Overconfident attack predictions → More false positives → High FAR
- Especially problematic if test set has class imbalance (more attacks than normal)

---

### 3. **Pseudo-Labeling Bias** ⭐⭐⭐

**Location**: `coordinators/centralized_coordinator.py` lines 311-320

**Current Behavior**:
```python
confidences, pseudo_labels = probs.max(dim=1)
confident_mask = confidences > pseudo_threshold
if confident_mask.sum() > 0:
    pseudo_loss = F.cross_entropy(
        logits[confident_mask],
        pseudo_labels[confident_mask],
        reduction='mean'
    )
```

**Problem**:
- **If base model predicts attacks frequently**, pseudo-labels are biased toward attacks
- **Positive feedback loop**:
  1. Base model predicts many attacks
  2. Pseudo-labels become attack-biased
  3. TTT learns from biased pseudo-labels
  4. Model predicts even more attacks
  5. FAR increases

**Impact**:
- Biased pseudo-labels → Model learns to predict more attacks → High FAR
- Especially problematic if base model already has high FAR

---

### 4. **No FAR Penalty in TTT Loss** ⭐⭐⭐⭐

**Location**: `coordinators/centralized_coordinator.py` lines 322-323

**Current Behavior**:
```python
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss
# ❌ NO FAR PENALTY TERM!
```

**Problem**:
- TTT loss **only has entropy + pseudo-label terms**
- **No explicit penalty** for false positives
- Model can **freely increase false positives** during adaptation
- No constraint to keep FAR low

**Impact**:
- Model adapts to maximize entropy minimization (confidence)
- But has no incentive to reduce false positives
- FAR can increase during TTT adaptation

---

### 5. **Config Settings Too Permissive** ⭐⭐⭐

**Location**: `config.py` lines 252-286

**Current Settings**:
```python
ttt_zdr_target: float = 0.85  # Aggressive ZDR target
ttt_zdr_max_far: float = 0.50  # Allows 50% FAR! (very high)
max_far_for_zdr: float = 0.35  # Permissive
max_far_allowed: float = 0.20  # For balanced strategy (may not be enforced)
```

**Problem**:
- **`ttt_zdr_max_far = 0.50`** allows up to 50% FAR (unacceptable for production!)
- **`max_far_for_zdr = 0.35`** is permissive
- **`max_far_allowed = 0.20`** might not be strictly enforced if no threshold satisfies it

**Impact**:
- Permissive settings allow high FAR to achieve high ZDR
- System prioritizes ZDR over FAR

---

## 📈 Expected FAR Values by Scenario

| Scenario | Expected FAR | Status |
|----------|--------------|--------|
| **Ideal** | <10% | ✅ Excellent |
| **Acceptable** | 10-20% | ⚠️ Acceptable |
| **High** | 20-30% | ⚠️ High |
| **Very High** | >30% | ❌ Unacceptable |

**Current**: TTT FAR often 30-50%+ (unacceptable for production)

---

## 🔬 Diagnostic Steps (Without Code Changes)

### Step 1: Check Current Threshold Strategy

```python
# Check config.py line 276
threshold_optimization_strategy: str = 'balanced_zdr_far'  # or 'zdr_optimized'?
```

**If `zdr_optimized`**: This is likely the primary cause of high FAR.

### Step 2: Check Threshold Values

After running evaluation, check logs for:
```
📊 Final Threshold: X.XXXX (ZDR-optimized or Balanced ZDR-FAR)
```

**If threshold < 0.3**: This is causing high FAR (low threshold = more false positives).

### Step 3: Check FAR Values

After running evaluation, check logs for:
```
🔍 TTT Model FAR: X.XXXX
```

**If FAR > 0.20 (20%)**: High FAR detected.

### Step 4: Check Probability Distribution

After running evaluation, check logs for:
```
📊 TTT Probability Analysis:
  ├─ Attack prob mean: X.XXXX
```

**If mean attack prob for normal samples > 0.5**: This is the direct cause of high FAR.

---

## ✅ Recommended Solutions (Configuration Changes Only)

### Solution 1: Use Balanced ZDR-FAR Strategy (RECOMMENDED)

**Action**: Ensure `config.py` line 276 is set to:
```python
threshold_optimization_strategy: str = 'balanced_zdr_far'
```

**Expected Impact**: Reduces FAR to 15-25% (from 30-50%+)

---

### Solution 2: Stricter FAR Constraints

**Action**: Modify `config.py`:
```python
# Line 252-253: Reduce max FAR allowed
ttt_zdr_max_far: float = 0.25  # REDUCED from 0.50 (stricter)

# Line 286: Reduce max FAR for ZDR optimization
max_far_for_zdr: float = 0.20  # REDUCED from 0.35 (stricter)

# Line 289: Stricter max FAR for balanced strategy
max_far_allowed: float = 0.15  # REDUCED from 0.20 (stricter)
```

**Expected Impact**: Reduces FAR by 10-15 percentage points

---

### Solution 3: Increase FAR Penalty Weight

**Action**: Modify `config.py` line 4011 in `main.py` (if you have access):
```python
# Current: score = zdr_at_thresh - 2.5 * far_at_thresh
# Change to: score = zdr_at_thresh - 5.0 * far_at_thresh  # Penalize FAR more
```

**Expected Impact**: Reduces FAR by 5-10 percentage points

---

### Solution 4: Reduce Entropy Weight

**Action**: Modify `config.py`:
```python
# Find entropy_weight in TTT config
# Reduce it to prevent overconfidence
entropy_weight: float = 0.5  # REDUCED from default (if > 1.0)
```

**Expected Impact**: Reduces overconfidence, lowers FAR by 5-10 percentage points

---

## 📝 Summary

**Primary Causes** (in order of impact):
1. **Threshold optimization strategy** prioritizing ZDR over FAR
2. **No FAR penalty** in TTT loss function
3. **Entropy minimization** causing overconfidence
4. **Pseudo-labeling bias** toward attacks
5. **Permissive config settings** allowing high FAR

**Quick Fix** (configuration only):
- Set `threshold_optimization_strategy = 'balanced_zdr_far'`
- Reduce `ttt_zdr_max_far` from 0.50 to 0.25
- Reduce `max_far_allowed` from 0.20 to 0.15

**Expected Result**: FAR reduced from 30-50% to 15-25%

---

## 🔍 How to Verify

After making configuration changes, run evaluation and check:

1. **Threshold value**: Should be > 0.4 (not too low)
2. **FAR value**: Should be < 0.25 (25%)
3. **ZDR value**: Should still be > 0.80 (80%) if using balanced strategy
4. **Normal sample attack prob**: Should be < 0.3 (mean)

If FAR is still high after these changes, the issue is likely in the TTT loss function (entropy minimization overconfidence), which would require code changes to fix.





