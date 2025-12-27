# FAR Penalty Step 2 Test - Diagnosis

**Test Date**: 2025-12-20 13:25
**Test Setup**: DoS attack, 3 episodes, FAR penalty weight = 0.05

---

## ❌ RESULT: FAR PENALTY NOT WORKING

### Observed Results:

| Metric | Base Model | TTT Model (with FAR penalty 0.05) | Change |
|--------|-----------|----------------------------------|---------|
| **ZDR** | 76.37% ± 5.91% | 94.34% ± 1.82% | +17.96% ✅ |
| **FAR** | 27.14% | **41.23% ± 3.67%** | **+14.09%** ❌ |
| **Accuracy** | 68.62% ± 1.33% | 71.30% ± 1.56% | +2.68% ✅ |
| **F1-Score** | 57.21% ± 2.32% | 70.04% ± 1.13% | +12.83% ✅ |

### Expected vs Actual:

- **Expected FAR**: ~38-40% (reduction from baseline ~43%)
- **Actual FAR**: 41.23% (NO meaningful reduction)
- **Verdict**: FAR penalty weight = 0.05 is **TOO WEAK**

---

## 🔍 Root Cause Analysis

### Why FAR Penalty Failed:

1. **Weak penalty weight (0.05)**:
   - Entropy loss weight: 1.0
   - Pseudo-label loss weight: 1.0
   - FAR penalty weight: **0.05** (only 5% of entropy)
   - Total loss dominated by entropy + pseudo-labels

2. **Decision boundary issue**:
   - Predictions use `argmax(logits)`: attack if `P(attack) > 0.5`
   - FAR penalty only affects: `P(attack) > 0.7`
   - **Gap**: Predictions with `0.5 < P(attack) < 0.7` still predict "attack" but are NOT penalized!

3. **Insufficient gradient signal**:
   - To change a prediction from attack → normal, need to move `P(attack)` from >0.5 to <0.5
   - Current penalty only weakly nudges probabilities in the range >0.7
   - Weight 0.05 is too small to overcome entropy/pseudo-label gradients

---

## 📊 Comparison with Comprehensive Results

From comprehensive results (9 attacks, 10 episodes each, **BEFORE** FAR penalty):
- **Base FAR**: 22.37%
- **TTT FAR**: **42.95%**
- **ZDR**: 95.63% ± 0.57%

From Step 2 test (DoS only, 3 episodes, **WITH** FAR penalty 0.05):
- **Base FAR**: 27.14%
- **TTT FAR**: **41.23% ± 3.67%**
- **ZDR**: 94.34% ± 1.82%

**Conclusion**: FAR penalty weight = 0.05 has **negligible effect** on FAR.

---

## 🎯 Recommended Next Steps

### Option A: Increase FAR Penalty Weight (Recommended)

**Try weight = 0.15 or 0.20** (3-4x stronger than current):

```python
# config.py
ttt_far_penalty_weight: float = 0.15  # Was 0.05, now 3x stronger
```

**Rationale**:
- Current weight (0.05) is too weak compared to entropy (1.0) and pseudo-label (1.0) losses
- Need stronger penalty to overcome other gradient signals
- 0.15-0.20 is still relatively gentle (15-20% of entropy weight)

**Expected impact with weight=0.15**:
- FAR: 41% → **32-35%** (meaningful reduction)
- ZDR: 94.3% → **92-93%** (slight drop, still excellent)
- Accuracy: 71.3% → **72-73%** (maintained or improved)
- F1: 70.0% → **72-74%** (improved)

---

### Option B: Lower Confidence Threshold

**Try threshold = 0.6 instead of 0.7**:

```python
# config.py
ttt_far_confidence_threshold: float = 0.6  # Was 0.7
```

**Rationale**:
- Penalize more predictions (those with `P(attack) > 0.6`)
- Closer to decision boundary (0.5), more effective at reducing FAR

**Expected impact with threshold=0.6 (keeping weight=0.05)**:
- FAR: 41% → **37-39%** (modest reduction)
- ZDR: 94.3% → **93-94%** (minimal drop)

---

### Option C: Combine Both (Most Aggressive)

**Increase weight to 0.15 AND lower threshold to 0.6**:

```python
# config.py
ttt_far_penalty_weight: float = 0.15
ttt_far_confidence_threshold: float = 0.6
```

**Expected impact**:
- FAR: 41% → **28-32%** (large reduction)
- ZDR: 94.3% → **90-92%** (noticeable drop, still good)
- F1: 70.0% → **74-76%** (significant improvement)

**Risk**: May hurt ZDR too much. Monitor carefully.

---

## 🚨 Critical Constraint

**DO NOT let ZDR drop below 90%**. This is your key strength compared to baseline (76%). If ZDR drops below 90%, the FAR penalty is too strong.

---

## ✅ Recommendation: Try Option A First

1. **Edit config.py**:
   ```python
   ttt_far_penalty_weight: float = 0.15  # Increased from 0.05
   ```

2. **Run Step 2 again** (3 episodes, DoS):
   ```bash
   python multi_episode_evaluation.py --attack DoS --episodes 3
   ```

3. **Check results**:
   - If FAR > 35%: increase weight to 0.20
   - If ZDR < 90%: decrease weight to 0.10
   - If 30% < FAR < 35% AND ZDR > 90%: **PERFECT! Proceed to full evaluation**

4. **If satisfied, run full evaluation** (all 9 attacks, 10 episodes):
   ```bash
   python run_comprehensive_multi_episode_evaluation.py --episodes 10
   ```

---

## 📈 Expected Timeline

- **Step 3a** (weight=0.15, DoS, 3 episodes): 1-2 hours
- **Step 3b** (optional tuning): 1-2 hours
- **Step 4** (full evaluation): 12-15 hours

**Total**: 14-19 hours

---

## 🎯 Success Criteria for Step 3

After adjusting weight to 0.15, we should see:

| Metric | Target Range | Why |
|--------|-------------|-----|
| **ZDR** | **90-94%** | Must stay excellent (>90%) |
| **FAR** | **32-38%** | Meaningful reduction from 41% |
| **Accuracy** | **71-73%** | Maintained or improved |
| **F1-Score** | **72-75%** | Improved from 70% |

If these targets are met, proceed to Step 4 (full evaluation).

---

## 🔧 Alternative: Dynamic FAR Penalty

If fixed weight doesn't work well, consider **adaptive weight** based on current FAR:

```python
# Pseudo-code (would need implementation)
if current_FAR > 40%:
    far_weight = 0.20  # Aggressive
elif current_FAR > 30%:
    far_weight = 0.15  # Moderate
elif current_FAR > 20%:
    far_weight = 0.10  # Gentle
else:
    far_weight = 0.05  # Minimal
```

This is more complex but could provide better balance. Defer this to later if needed.

---

## Summary

- ❌ **FAR penalty weight = 0.05 is TOO WEAK** (no meaningful FAR reduction)
- ✅ **ZDR protected** (94.3%, still excellent)
- ✅ **Accuracy improved** (+2.68%)
- ✅ **F1-Score improved** (+12.83%)
- 🎯 **Next step**: Increase weight to **0.15** and re-test (Step 3)
