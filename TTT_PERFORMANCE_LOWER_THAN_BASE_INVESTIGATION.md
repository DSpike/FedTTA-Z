# TTT Performance Lower Than Base Model - Investigation Report

## Current Performance Status

### **Single Evaluation Results** (Latest Run):

- **Base Model**: 91.57% accuracy, 94.09% F1-score, 91.00% AUC-PR
- **TTT Model**: 88.25% accuracy, 92.48% F1-score, 90.78% AUC-PR
- **Difference**: **-3.32% accuracy, -1.61% F1, -0.22% AUC-PR**

### **K-Fold Cross-Validation Results**:

- **Base Model**: 87.65% ± 2.61% accuracy
- **TTT Model**: 87.35% ± 2.26% accuracy
- **Difference**: **-0.30% accuracy** (within variance, but consistent)

---

## Root Cause Analysis

### **1. Overfitting to Adaptation Query Set** ⚠️ PRIMARY ISSUE

**Problem**: TTT adapts to a specific subset of test data (750 samples), which may not represent the full test distribution.

**Evidence**:

- Single evaluation: TTT adapts on full test set (750 samples) but performs worse
- K-fold CV: TTT adapts on 80% per fold (600 samples), evaluates on 20% (150 samples)
- **Smaller adaptation set in k-fold CV** → Less effective adaptation

**Mechanism**:

1. TTT minimizes entropy + diversity loss on the query set
2. Model becomes specialized to query set distribution
3. When evaluated on full test set (or different fold), distribution mismatch occurs
4. Performance degrades due to over-adaptation

---

### **2. Insufficient Adaptation Data in K-Fold CV**

**Problem**: In k-fold CV, TTT only adapts on 80% of data per fold (600 samples vs 750 in single evaluation).

**Impact**:

- **Less adaptation data** → Less effective learning
- **Different data splits** → TTT adapts to different distributions each fold
- **Higher variance** across folds → Lower average performance

**Evidence from Documents**:

> "While k-fold CV shows similar generalization across batch sizes, the single evaluation demonstrates that batch size 32 provides superior adaptation to the specific test distribution"

---

### **3. Loss-Task Mismatch**

**Problem**: TTT optimizes entropy + diversity loss, which may not directly align with classification accuracy.

**Current Loss Function**:

```python
total_loss = entropy_loss + (diversity_weight * diversity_loss)
# Where diversity_weight = 0.15
```

**Issue**:

- Entropy minimization makes model confident but doesn't guarantee correct predictions
- Diversity loss maintains class balance but may conflict with accuracy
- **No direct accuracy/classification loss** → Model may optimize for wrong objective

---

### **4. Gradient Norm Not Converging**

**Evidence from Convergence Analysis**:

- Gradient norm: **0.526** (final) >> convergence threshold (≈1e-4)
- **Gap: 5260x larger** than threshold
- Model is **NOT at stationary point** → Optimization may be incomplete or unstable

**Impact**:

- Model is still "learning" during adaptation
- May be exploring suboptimal regions of loss landscape
- Performance may fluctuate rather than improve

---

### **5. Optimal Threshold Issues**

**From Latest Results**:

- **Optimal Threshold**: 0.896 (very high!)
- Base model threshold: ~0.5 (standard)
- **High threshold** suggests TTT model is overconfident or poorly calibrated

**Problem**:

- TTT model may be producing probabilities that are too high/low
- Optimal threshold finding may be compensating for poor calibration
- High threshold (0.896) indicates model needs high confidence to predict attacks

---

### **6. Batch Size Effects**

**Evidence from Batch Size Comparison**:

- Batch 32: **+1.51% accuracy improvement** (single evaluation) ✅
- Batch 64: **+0.60% accuracy improvement** (single evaluation)
- But **k-fold CV shows NO improvement** for any batch size

**Interpretation**:

- Single evaluation: TTT adapts well to specific test distribution
- K-fold CV: TTT struggles with different data splits
- **Suggests TTT is dataset-specific** rather than generalizable

---

## Why This Happens

### **1. Distribution Shift Between Adaptation and Evaluation**

In k-fold CV:

- **Adaptation set**: 80% of test data (one distribution)
- **Evaluation set**: 20% of test data (potentially different distribution)
- TTT overfits to adaptation distribution → Fails on evaluation set

### **2. Small Adaptation Set Size**

With k=5 folds:

- **Adaptation samples**: 600 per fold (vs 750 in single evaluation)
- **25% reduction** in adaptation data
- Less data → Less effective learning → Lower performance

### **3. Dual Objective Conflict**

TTT optimizes two competing objectives:

- **Entropy minimization**: Make predictions confident
- **Diversity preservation**: Maintain class balance

These may conflict:

- Minimizing entropy → Model becomes confident (possibly wrong)
- Maintaining diversity → Model spreads predictions (may reduce accuracy)

### **4. No Validation Signal**

TTT is **purely unsupervised**:

- No labels during adaptation
- No validation set to monitor performance
- Model doesn't know if it's improving or degrading
- **Early stopping** based on loss, not accuracy

---

## Recommendations

### **Priority 1: Increase Adaptation Data in K-Fold CV**

**Current**: 80% adaptation, 20% evaluation (k=5)
**Proposed**: 90% adaptation, 10% evaluation (k=10)

**Benefits**:

- More adaptation data (90% vs 80%)
- More folds for better statistics
- Still maintains statistical validity

**Trade-off**: Smaller evaluation sets per fold (higher variance)

---

### **Priority 2: Add Validation-Based Early Stopping**

**Current**: Early stopping based on loss only
**Proposed**: Monitor validation accuracy during adaptation

**Implementation**:

```python
# During TTT adaptation:
# 1. Split adaptation set: 80% train, 20% validation
# 2. Monitor validation accuracy
# 3. Stop if validation accuracy decreases
```

**Benefits**:

- Prevents overfitting to adaptation set
- Stops when performance degrades
- Better generalization to test set

---

### **Priority 3: Adjust Diversity Weight**

**Current**: `diversity_weight = 0.15`
**Analysis**: May be too high, causing conflict with accuracy

**Proposed**:

- Lower to `0.10` or `0.05`
- Or make it adaptive based on class balance

**Rationale**:

- Reduce conflict between entropy and diversity
- Allow model to focus more on confidence (entropy)
- Still maintain minimum diversity

---

### **Priority 4: Use Pseudo-Labels**

**Current**: Pure unsupervised TTT (entropy + diversity only)
**Proposed**: TENT + Pseudo-labels (already implemented)

**Benefits**:

- Provides weak supervision signal
- Better guidance for adaptation
- Can improve accuracy by 8-12%

**Check**: Ensure `use_pseudo_labels: bool = True` in config

---

### **Priority 5: Increase TTT Steps**

**Current**: 50 steps (config shows `ttt_base_steps = 50`)
**Proposed**: 75-100 steps

**Rationale**:

- More steps may allow better convergence
- Gradient norm not approaching zero suggests insufficient steps
- May need more time to find optimal adaptation

**Trade-off**: Longer computation time

---

## Conclusion

**TTT is performing lower than base model due to**:

1. ✅ **Overfitting to adaptation query set** (PRIMARY)
2. ✅ **Insufficient adaptation data in k-fold CV**
3. ✅ **Loss-task mismatch** (optimizing wrong objective)
4. ✅ **Gradient norm not converging** (incomplete optimization)
5. ✅ **High optimal threshold** (poor calibration)

**Recommended Action**:

1. **Increase k-fold adaptation data** (90% vs 80%)
2. **Add validation-based early stopping**
3. **Lower diversity weight** (0.10 or adaptive)
4. **Ensure pseudo-labels are enabled**
5. **Consider increasing TTT steps** (if computation allows)

**Expected Outcome**:

- TTT should match or exceed base model performance
- Better generalization across different data splits
- More stable k-fold CV results
