# Real TTT Improvement Strategies for Backdoor Attacks

**Date**: December 21, 2025
**Finding**: TTT didn't work "previously" - it was the same lucky single run
**Goal**: Find ways to ACTUALLY make TTT work for Backdoor

---

## The Truth About "Previous Success"

### What You Observed
- Dec 19 results: TTT ZDR 95.65%, FAR 0.00% ✅
- Dec 21 results: TTT ZDR 95.65%, FAR 0.00% ✅
- "It worked before, what changed?"

### The Reality
**These are THE SAME RUN** - identical results because:
1. Same random seed (42)
2. Same test set split (184 sequences)
3. Same batch composition
4. Same model initialization

**100-episode average reveals truth**:
- TTT ZDR: 88.69% ± 1.79% ❌
- TTT FAR: 45.11% ± 2.31% ❌
- ZDR change: -4.64% (worse)
- FAR change: +8.88% (worse)

**Conclusion**: TTT never worked - the single run was a statistical outlier (3.9σ).

---

## Real Problem: How to Make TTT ACTUALLY Work

### Current Situation (100-Episode Truth)

| Metric | Base | TTT | Change | Status |
|--------|------|-----|--------|--------|
| ZDR | 93.33% | 88.69% | -4.64% | ❌ Worse |
| FAR | 36.23% | 45.11% | +8.88% | ❌ Worse |
| Variance | 0.00 | 1.79% | ∞ | ❌ Unstable |

### Why TTT Fails
1. Insufficient data (583 samples)
2. Aggressive hyperparameters (43.9x oversampling)
3. Poor embeddings (prototypes not separated)
4. Overconfidence (entropy minimization backfires)
5. Extreme imbalance (63:1 Normal:Backdoor)

---

## Strategy 1: Conservative TTT Hyperparameters

### Current Settings (Problematic)
```python
ttt_lr = 0.005                    # Too high
ttt_max_steps = 400                # Way too many
ttt_batch_size = 64               # OK
ttt_confidence_reg_weight = 0.4   # Too weak
pseudo_threshold = 0.8            # Too lenient
pseudo_weight = 1.5               # Too strong
entropy_weight = 0.8              # OK
```

### Recommended Settings for Rare Attacks (<1,000 samples)
```python
# OPTION A: Conservative (Prevent Overfitting)
ttt_lr = 0.001                    # 80% reduction
ttt_max_steps = 25                # 93.75% reduction
ttt_confidence_reg_weight = 0.8   # 100% increase
pseudo_threshold = 0.95           # Much stricter
pseudo_weight = 0.5               # 66% reduction
entropy_weight = 0.3              # Reduce aggressive confidence

# OPTION B: Ultra-Conservative (Minimal Adaptation)
ttt_lr = 0.0005                   # 90% reduction
ttt_max_steps = 10                # 97.5% reduction
ttt_confidence_reg_weight = 1.0   # Maximum
pseudo_threshold = 0.98           # Almost no pseudo-labels
pseudo_weight = 0.2               # Minimal influence
entropy_weight = 0.1              # Minimal confidence push
```

### Expected Impact
- **Option A**: Reduce oversampling 400→25 = 16x repetition (vs current 44x)
- **Option B**: Reduce oversampling 400→10 = 4x repetition (minimal overfitting)
- **Goal**: Prevent memorization, allow gentle adaptation

**Test this**: Run 100-episode evaluation with Option A, then Option B

---

## Strategy 2: Data Augmentation (Address Root Cause)

### The Core Problem
- Current: 583 Backdoor samples
- Needed: >1,000 samples for stable TTT
- Gap: 417 samples (72% shortfall)

### Approach 1: SMOTE at Test Time
```python
from imblearn.over_sampling import SMOTE

# Before TTT adaptation
if zero_day_samples < 1000:
    # Augment test set
    smote = SMOTE(sampling_strategy={'Backdoor': 1500}, random_state=42)
    X_test_aug, y_test_aug = smote.fit_resample(X_test, y_test)

    # Now run TTT on augmented data
    adapted_model = ttt_adapt(base_model, X_test_aug, y_test_aug)
```

**Pros**:
- Addresses root cause directly
- Creates 1,500 Backdoor samples (2.5x increase)
- Reduces oversampling ratio: 44x → 17x

**Cons**:
- Synthetic samples may not capture real variance
- Risk of learning from synthetic noise

### Approach 2: Cross-Dataset Transfer
```python
# Collect Backdoor samples from multiple datasets
backdoor_cicids = load_backdoor_from_cicids()  # ~500 samples
backdoor_ciciot = load_backdoor_from_ciciot()  # ~300 samples
backdoor_unsw = load_backdoor_from_unsw()      # 583 samples

# Combine and harmonize features
total_backdoor = 1,383 samples

# Use for TTT adaptation
```

**Pros**:
- Real samples, not synthetic
- More diverse attack patterns
- Exceeds 1,000 threshold

**Cons**:
- Feature alignment needed
- Different distributions may confuse model

### Approach 3: Intelligent Resampling
```python
# Instead of random oversampling, use variance-based sampling
# Prioritize diverse Backdoor samples

from sklearn.cluster import KMeans

# Cluster Backdoor samples
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(backdoor_samples)

# Sample proportionally from each cluster
augmented_samples = []
for cluster_id in range(10):
    cluster_samples = backdoor_samples[clusters == cluster_id]
    # Oversample each cluster equally
    augmented_samples.append(resample(cluster_samples, n_samples=100))

# Result: 1,000 samples with maximum diversity
```

**Pros**:
- Maintains diversity
- Avoids repetitive memorization
- Balanced representation of Backdoor variants

**Cons**:
- More complex
- Still uses synthetic samples

**Test this**: Try all 3 approaches, measure 100-episode performance

---

## Strategy 3: Fix Embedding Quality

### Current Problem
- Prototypes NOT well-separated
- Backdoor overlaps with Normal
- Prototype accuracy: 0.0000

### Solution 1: Contrastive Pre-Training
```python
# Add contrastive loss during base model training
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        self.margin = margin

    def forward(self, embeddings, labels):
        # Pull same class together
        # Push different classes apart
        # Focus on Backdoor vs Normal separation
        pass

# Modified training
for epoch in epochs:
    loss = classification_loss + 0.5 * contrastive_loss
```

**Impact**:
- Better separation of Backdoor from Normal
- TTT adaptation starts from better embeddings
- Expected ZDR improvement: +5-10%

### Solution 2: Prototype Margin Loss (Already Implemented)
```python
# Already in config:
prototype_margin_loss_weight = 1.0
prototype_margin = 4.5

# But may need tuning for Backdoor
# Increase margin for better separation
prototype_margin = 6.0  # Increase from 4.5
```

### Solution 3: Backdoor-Focused Meta-Learning
```python
# During meta-learning, create tasks specifically with Backdoor variants
# Current: Backdoor excluded (it's zero-day)
# New: Include synthetic Backdoor variants in support set

meta_tasks = []
for episode in range(n_episodes):
    # Generate synthetic Backdoor variant
    backdoor_variant = add_noise_to(backdoor_samples, noise_level=0.1)

    # Create task: Normal vs Backdoor_variant
    support_set = sample(Normal=100, Backdoor_variant=100)
    query_set = sample(Normal=20, Backdoor_variant=20)

    meta_tasks.append((support_set, query_set))

# Train model to distinguish Backdoor variants
# → Better generalization to unseen Backdoor at test time
```

**Test this**: Retrain base model with contrastive loss + Backdoor meta-tasks

---

## Strategy 4: Attack-Specific TTT

### The Insight
Not all attacks need the same TTT strategy:
- DoS (4,089 samples): Aggressive TTT works
- Backdoor (583 samples): Aggressive TTT fails

### Implementation
```python
class AdaptiveTTT:
    def __init__(self):
        self.configs = {
            'high_data': {  # >2,000 samples
                'lr': 0.005,
                'steps': 400,
                'confidence_reg': 0.4
            },
            'medium_data': {  # 1,000-2,000 samples
                'lr': 0.002,
                'steps': 100,
                'confidence_reg': 0.6
            },
            'low_data': {  # <1,000 samples
                'lr': 0.0005,
                'steps': 10,
                'confidence_reg': 1.0
            }
        }

    def adapt(self, model, test_data, zero_day_samples):
        # Choose config based on data availability
        if zero_day_samples > 2000:
            config = self.configs['high_data']
        elif zero_day_samples > 1000:
            config = self.configs['medium_data']
        else:
            config = self.configs['low_data']

        # Apply TTT with appropriate config
        return ttt_adapt(model, test_data, **config)
```

**Expected Impact**:
- Backdoor (583): Ultra-conservative TTT
- DoS (4,089): Aggressive TTT
- Each attack gets optimal strategy

**Test this**: Implement and run 100-episode evaluation

---

## Strategy 5: Ensemble Revisited (But Different)

### Why Previous Ensemble Failed
- Base model: ZDR 93.33%, FAR 36.23%
- TTT model: ZDR 88.69%, FAR 45.11%
- Ensemble of "good + bad" = "mediocre"

### New Idea: Sample-Level Ensemble
Instead of always using ensemble, use **adaptive selection**:

```python
def adaptive_ensemble(base_probs, ttt_probs, base_confidence, ttt_confidence):
    # Use base model when:
    # 1. Base is very confident AND
    # 2. TTT is uncertain
    if base_confidence > 0.9 and ttt_confidence < 0.6:
        return base_probs

    # Use TTT model when:
    # 1. TTT is very confident AND
    # 2. Base is uncertain
    elif ttt_confidence > 0.9 and base_confidence < 0.6:
        return ttt_probs

    # Use weighted average when both uncertain or both confident
    else:
        # Weight by confidence
        weight = base_confidence / (base_confidence + ttt_confidence)
        return weight * base_probs + (1-weight) * ttt_probs
```

**Rationale**:
- Base model good at conservatively identifying Normal
- TTT model good at aggressively detecting attacks
- Combine strengths adaptively, not blindly

**Test this**: Implement adaptive ensemble, run 100-episode evaluation

---

## Strategy 6: Confidence Calibration

### The Overconfidence Problem
TTT makes confident wrong predictions:
- Predicts "Attack" with 0.95 confidence
- But it's actually Normal
- → False alarm

### Solution: Temperature Scaling Post-TTT
```python
class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0

    def calibrate(self, model, validation_set):
        # Find optimal temperature on validation set
        # Temperature > 1 → Less confident (flatten probabilities)
        # Temperature < 1 → More confident (sharpen probabilities)

        best_temp = 1.0
        best_far = float('inf')

        for temp in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            probs_calibrated = model(X_val) / temp
            far = compute_far(probs_calibrated, y_val)

            if far < best_far:
                best_far = far
                best_temp = temp

        self.temperature = best_temp

    def predict(self, model, X_test):
        logits = model(X_test)
        # Scale logits by temperature
        probs = softmax(logits / self.temperature)
        return probs

# Usage
calibrator = TemperatureScaling()
calibrator.calibrate(ttt_model, X_val)
predictions = calibrator.predict(ttt_model, X_test)
```

**Expected Impact**:
- Reduce overconfidence
- Lower FAR (fewer false alarms)
- Slight ZDR decrease (trade-off)
- Target: FAR 45% → 30%, ZDR 88.69% → 85%

**Test this**: Implement temperature scaling, run 100-episode evaluation

---

## Strategy 7: Early Stopping with Validation

### Current Problem
TTT runs for 400 steps regardless of performance:
- Early steps: Useful adaptation
- Later steps: Overfitting to noise

### Solution: Monitor Validation Performance
```python
def ttt_with_early_stopping(model, X_test, y_test, X_val, y_val):
    best_val_loss = float('inf')
    best_model = None
    patience = 5
    wait = 0

    for step in range(max_steps):
        # Adapt on test data
        loss = adapt_step(model, X_test, y_test)

        # Evaluate on validation set
        val_loss = evaluate(model, X_val, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1

        # Early stop if no improvement
        if wait >= patience:
            print(f"Early stopping at step {step}")
            return best_model

    return best_model
```

**Rationale**:
- Stop when adaptation starts overfitting
- For Backdoor: Likely stops at ~10-20 steps
- For DoS: Can continue to ~100-200 steps

**Test this**: Implement early stopping, run 100-episode evaluation

---

## Priority Testing Plan

### Phase 1: Quick Wins (Test Immediately)
1. **Conservative Hyperparameters** (Option B)
   - Expected improvement: Modest
   - Effort: Minimal (config change)
   - Test: 100-episode evaluation

2. **Temperature Scaling**
   - Expected improvement: Moderate (FAR reduction)
   - Effort: Low (add calibration layer)
   - Test: 100-episode evaluation

### Phase 2: Medium Effort (Next Week)
3. **Attack-Specific TTT**
   - Expected improvement: Significant
   - Effort: Moderate (code refactoring)
   - Test: 100-episode for all attacks

4. **Early Stopping with Validation**
   - Expected improvement: Moderate
   - Effort: Moderate (add validation loop)
   - Test: 100-episode evaluation

### Phase 3: High Effort (Research Direction)
5. **Data Augmentation (SMOTE)**
   - Expected improvement: High (addresses root cause)
   - Effort: High (testing, validation)
   - Test: 100-episode evaluation

6. **Contrastive Pre-Training**
   - Expected improvement: Very High (better embeddings)
   - Effort: Very High (retrain base model)
   - Test: Full pipeline rerun

7. **Backdoor Meta-Learning**
   - Expected improvement: Very High (fundamental improvement)
   - Effort: Very High (research project)
   - Test: Full experimental redesign

---

## Expected Outcomes

### Conservative Estimate (Phase 1 only)
- Current: ZDR 88.69%, FAR 45.11%
- After Phase 1: ZDR 90-92%, FAR 35-40%
- Improvement: +2-3% ZDR, -5-10% FAR

### Optimistic Estimate (Phase 1 + 2)
- After Phase 2: ZDR 92-94%, FAR 30-35%
- Improvement: +4-5% ZDR, -10-15% FAR

### Best Case (All Strategies)
- After Phase 3: ZDR 94-96%, FAR 20-25%
- Improvement: +6-7% ZDR, -20-25% FAR
- **Matches or exceeds base model** (ZDR 93.33%, FAR 36.23%)

---

## Recommendation: Start with Phase 1

**Immediate Actions**:
1. Implement **Conservative Hyperparameters (Option B)**
2. Implement **Temperature Scaling**
3. Run **100-episode evaluation**
4. Compare with current results

**Success Criteria**:
- ZDR > 90% (vs current 88.69%)
- FAR < 40% (vs current 45.11%)
- Variance < 1.5% (vs current 1.79%)

If Phase 1 succeeds, proceed to Phase 2. If not, consider Phase 3 (data augmentation).

---

## Conclusion

**Truth**: TTT never worked for Backdoor - the "previous success" was a lucky outlier.

**Real Challenge**: Make TTT actually work through systematic improvements.

**Path Forward**:
1. Quick wins (Phase 1) for immediate improvement
2. Medium effort (Phase 2) for significant gains
3. Research investment (Phase 3) for fundamental solution

**Key Insight**: Don't chase the 0% FAR outlier - it's not real. Focus on improving the 88.69% ZDR / 45.11% FAR average through principled methods.
