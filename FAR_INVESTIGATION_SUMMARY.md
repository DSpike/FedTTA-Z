# FAR (False Alarm Rate) Investigation Summary

## 🔍 Root Causes of High FAR

Based on code analysis, here are the **primary causes** of high FAR:

### 1. **Threshold Optimization Strategy** ⭐⭐⭐⭐⭐ (PRIMARY CAUSE)

**Location**: `main.py` lines 3940-4050

**Problem**:
- Current strategy: `balanced_zdr_far` (configurable)
- If strategy falls back to `zdr_optimized`, it prioritizes ZDR over FAR
- Search range: `np.linspace(0.3, 0.9, 300)` may find thresholds that favor ZDR
- `max_far_allowed = 0.20` might not be strict enough

**Evidence**:
- `ttt_zdr_max_far = 0.50` allows up to 50% FAR (very high!)
- `max_far_for_zdr = 0.35` is permissive
- Low thresholds (0.3-0.5) maximize ZDR but cause high FAR

**Impact**: 
- Low threshold → More predictions as Attack → More false positives → High FAR

---

### 2. **Entropy Minimization Overconfidence** ⭐⭐⭐⭐

**Location**: `coordinators/centralized_coordinator.py` lines 305-307

**Problem**:
```python
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()
```

- Entropy minimization makes predictions more confident
- If model becomes overconfident about attacks, it predicts more attacks
- No class balancing, but overconfidence bias can favor attacks
- No explicit FAR penalty in loss function

**Impact**:
- Overconfident attack predictions → More false positives → High FAR

---

### 3. **Pseudo-Labeling Bias** ⭐⭐⭐

**Location**: `coordinators/centralized_coordinator.py` lines 311-320

**Problem**:
```python
confidences, pseudo_labels = probs.max(dim=1)
confident_mask = confidences > pseudo_threshold
pseudo_loss = F.cross_entropy(logits[confident_mask], pseudo_labels[confident_mask])
```

- If base model predicts attacks frequently, pseudo-labels are biased toward attacks
- TTT learns from these biased pseudo-labels
- Creates positive feedback loop: more attack predictions → more attack pseudo-labels → more attack predictions

**Impact**:
- Biased pseudo-labels → Model learns to predict more attacks → High FAR

---

### 4. **No FAR Penalty in TTT Loss** ⭐⭐⭐⭐

**Location**: `coordinators/centralized_coordinator.py` lines 322-323

**Problem**:
```python
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss
# ❌ NO FAR PENALTY!
```

- TTT loss only has entropy + pseudo-label terms
- No explicit penalty for false positives
- Model isn't directly penalized for predicting normal samples as attacks

**Impact**:
- No FAR constraint during adaptation → Model can freely increase false positives

---

### 5. **Config Settings Too Permissive** ⭐⭐⭐

**Location**: `config.py` lines 252-286

**Problem**:
- `ttt_zdr_max_far = 0.50` (allows 50% FAR - very high!)
- `max_far_for_zdr = 0.35` (permissive)
- `max_far_allowed = 0.20` (might not be enforced strictly)

**Impact**:
- Permissive settings allow high FAR to achieve high ZDR

---

## 📊 Expected FAR Values

| Scenario | Expected FAR | Status |
|----------|--------------|--------|
| **Ideal** | <10% | ✅ Excellent |
| **Acceptable** | 10-20% | ⚠️ Acceptable |
| **High** | 20-30% | ⚠️ High |
| **Very High** | >30% | ❌ Unacceptable |

**Current**: TTT FAR often 30-50%+ (unacceptable for production)

---

## ✅ Recommended Fixes

### Fix 1: Add FAR Penalty to TTT Loss (CRITICAL)

```python
# In coordinators/centralized_coordinator.py, add to total_loss:

# Calculate FAR penalty
normal_mask = (y_test_binary == 0)  # Assuming you have access to labels
if normal_mask.sum() > 0:
    normal_probs = probs[normal_mask, 1]  # Attack probabilities for normal samples
    # Penalize high attack probabilities for normal samples
    far_penalty = normal_probs.mean()  # Mean attack prob for normal samples
    far_penalty_weight = 0.3  # Configurable weight
    total_loss = total_loss + far_penalty_weight * far_penalty
```

### Fix 2: Stricter Threshold Constraints

```python
# In config.py:
max_far_allowed: float = 0.15  # REDUCED from 0.20 (stricter)
ttt_zdr_max_far: float = 0.25  # REDUCED from 0.50 (much stricter)
```

### Fix 3: Add Prediction Ratio Constraint

```python
# In TTT adaptation, add constraint:
predicted_attack_ratio = (probs[:, 1] > 0.5).float().mean()
if predicted_attack_ratio > 0.4:  # Max 40% predicted as attacks
    ratio_penalty = (predicted_attack_ratio - 0.4) ** 2
    total_loss = total_loss + 0.5 * ratio_penalty
```

### Fix 4: Use Higher Thresholds

```python
# In threshold optimization, prioritize FAR:
# Instead of: score = zdr_at_thresh - 2.5 * far_at_thresh
# Use: score = zdr_at_thresh - 5.0 * far_at_thresh  # Penalize FAR more
```

---

## 🔬 Diagnostic Commands

Run these to investigate FAR:

```python
# 1. Check confusion matrices
python -c "
import json
data = json.load(open('performance_plots/performance_metrics_.json'))
base = data['base_model']
ttt = data['ttt_adapted_model']
print(f'Base FAR: {base.get(\"far\", 0)}')
print(f'TTT FAR: {ttt.get(\"far\", 0)}')
"

# 2. Check thresholds
python -c "
import json
data = json.load(open('performance_plots/performance_metrics_.json'))
print(f'Base threshold: {data[\"base_model\"].get(\"optimal_threshold\", 0.5)}')
print(f'TTT threshold: {data[\"ttt_adapted_model\"].get(\"optimal_threshold\", 0.5)}')
"

# 3. Run investigation script
python investigate_far_causes.py
```

---

## 📝 Next Steps

1. **Immediate**: Add FAR penalty to TTT loss (Fix 1)
2. **Short-term**: Stricter threshold constraints (Fix 2)
3. **Medium-term**: Add prediction ratio constraint (Fix 3)
4. **Long-term**: Comprehensive FAR-aware TTT adaptation





