# Comprehensive FAR Reduction Plan

**Goal**: Reduce FAR from 41.59% to <10% while maintaining ZDR >90%

**Current Status**:
- ZDR: 93.65% ± 0.81% ✅ Excellent
- FAR: 41.59% ❌ Too high (SOTA: 3.68%)
- Accuracy: 70.49%
- F1-Score: 69.04%

---

## Root Cause Analysis

### Why TTT Has High FAR:

1. **Entropy minimization makes model overconfident**
   - Pushes probabilities to extremes (median attack prob: 0.976)
   - Low entropy = high confidence predictions
   - Even uncertain samples get pushed to confident "attack" predictions

2. **No calibration after adaptation**
   - TTT changes feature distributions
   - But decision boundaries stay the same (prototypes are fixed)
   - Mismatch between adapted features and original boundaries

3. **Current FAR penalty ineffective**
   - Weight 0.15 is too weak vs entropy minimization
   - Only penalizes excess confidence above threshold
   - Entropy minimization is stronger force

---

## Solution Strategy: Multi-Pronged Approach

### Strategy 1: Temperature Scaling (Post-TTT Calibration) ⭐ PRIMARY

**What**: Scale logits by temperature T>1 before softmax
**Why**: Softens overconfident predictions without retraining
**Expected Impact**: FAR reduction of 50-70% (41% → 12-20%)

**Implementation**:
```python
# After TTT adaptation, before predictions
calibrated_logits = logits / temperature  # temperature = 1.5-3.0
calibrated_probs = F.softmax(calibrated_logits, dim=1)
```

**Advantages**:
- Simple, no retraining needed
- Proven effective for overconfident models
- Doesn't affect ZDR (changes ALL predictions uniformly)

**Tuning**: Find optimal temperature on validation set

---

### Strategy 2: Per-Attack Threshold Optimization

**What**: Find optimal decision threshold per attack type using ROC curves
**Why**: Different attacks have different optimal thresholds
**Expected Impact**: FAR reduction of 20-30% (41% → 28-32%)

**Implementation**:
```python
# For each attack type:
fpr, tpr, thresholds = roc_curve(y_true, y_score)
# Find threshold where FAR < 10% while maximizing ZDR
optimal_threshold = thresholds[np.argmax(tpr[fpr < 0.1])]
```

**Advantages**:
- Attack-specific optimization
- Balances precision-recall trade-off
- Can target specific FAR constraints

---

### Strategy 3: Ensemble Base + TTT Predictions

**What**: Weighted combination of base model and TTT model predictions
**Why**: Base model has lower FAR (23.20%), TTT has higher ZDR (93.65%)
**Expected Impact**: FAR reduction of 30-40% (41% → 24-28%)

**Implementation**:
```python
# Weighted ensemble
ensemble_probs = alpha * base_probs + (1-alpha) * ttt_probs  # alpha = 0.3-0.5
# Or voting ensemble
final_pred = (base_pred + ttt_pred) >= 1  # At least one predicts attack
```

**Advantages**:
- Combines strengths of both models
- Base model provides conservative baseline
- TTT model catches zero-day attacks

---

### Strategy 4: Alternative TTT Objectives

**What**: Replace pure entropy minimization with gentler objectives
**Why**: Entropy minimization is too aggressive
**Expected Impact**: FAR reduction of 40-50% (41% → 20-24%)

**Options**:
1. **Confidence Regularization**: Penalize over-confidence
   ```python
   confidence_loss = ((probs.max(dim=1)[0] - target_confidence)**2).mean()
   ```

2. **Soft Entropy**: Entropy minimization with upper bound
   ```python
   entropy_loss = torch.clamp(entropy, min=0.1, max=0.8).mean()
   ```

3. **Margin-based**: Maximize margin between top-2 classes
   ```python
   top2_probs = torch.topk(probs, 2, dim=1)[0]
   margin_loss = -torch.log(top2_probs[:, 0] - top2_probs[:, 1] + eps).mean()
   ```

---

## Implementation Plan

### Phase 1: Temperature Scaling (Quick Win)
**Time**: 2-3 hours
**Steps**:
1. Add temperature parameter to config
2. Apply temperature scaling after TTT adaptation
3. Grid search optimal temperature (1.0, 1.5, 2.0, 2.5, 3.0)
4. Evaluate on all 9 attack types
5. **Expected Result**: FAR 12-20%, ZDR >90%

### Phase 2: Threshold Optimization (If FAR still >10%)
**Time**: 3-4 hours
**Steps**:
1. Compute ROC curves for each attack type
2. Find thresholds satisfying FAR < 10%
3. Store per-attack thresholds in config
4. Re-evaluate with optimized thresholds
5. **Expected Result**: FAR 8-12%, ZDR >85%

### Phase 3: Ensemble (If ZDR drops <90%)
**Time**: 2-3 hours
**Steps**:
1. Implement weighted ensemble
2. Grid search ensemble weight (0.2, 0.3, 0.4, 0.5)
3. Balance FAR vs ZDR trade-off
4. **Expected Result**: FAR <10%, ZDR >90%

### Phase 4: Alternative Objectives (If all else fails)
**Time**: 1-2 days
**Steps**:
1. Implement confidence regularization
2. Test different TTT objectives
3. Re-train with new objective
4. Full evaluation on all attacks
5. **Expected Result**: FAR <8%, ZDR >88%

---

## Success Criteria

### Minimum Acceptable (Mid-tier venues):
- FAR < 15%
- ZDR > 85%
- F1-Score > 75%
- Accuracy > 80%

### Target (Top-tier venues):
- FAR < 10% ✅ PRIMARY GOAL
- ZDR > 90% ✅ MAINTAIN
- F1-Score > 85%
- Accuracy > 90%

### Ideal (IEEE TIFS, TDSC):
- FAR < 5%
- ZDR > 92%
- F1-Score > 90%
- Accuracy > 95%

---

## Testing Protocol

For each strategy:
1. Run on single attack (DoS) first
2. Verify FAR reduction without ZDR loss
3. Run comprehensive evaluation (9 attacks × 10 episodes)
4. Compute confidence intervals
5. Compare with SOTA
6. Update Excel comparison

---

## Fallback Options

If all technical solutions fail:

### Option A: Reframe as "High-Recall Security System"
- Target: Military/critical infrastructure applications
- Emphasize: Missing attacks more costly than false alarms
- Venue: Domain-specific security conferences

### Option B: Comparative Study
- Compare multiple TTT approaches
- Analyze FAR-ZDR trade-offs
- Contribute: Understanding of TTT behavior
- Venue: Meta-learning workshops

### Option C: Pivot Research
- Use infrastructure for different problem
- E.g., federated learning, concept drift
- Venue: Broader ML conferences

---

## Next Steps

**IMMEDIATE** (Start now):
1. Implement temperature scaling
2. Test on DoS attack
3. If successful, run comprehensive evaluation

**Estimated Timeline**:
- Temperature scaling: 3 hours
- Comprehensive eval: 3 hours
- Analysis + Excel update: 1 hour
- **Total: ~7 hours (1 day)**

**Decision Point**:
- If FAR < 10%: Proceed with paper writing
- If FAR 10-15%: Try threshold optimization
- If FAR >15%: Try ensemble approach
