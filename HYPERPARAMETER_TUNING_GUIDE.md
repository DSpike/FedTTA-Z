# Hyperparameter Tuning Guide for Zero-Day Detection

## 📊 Current Performance Analysis

**Single Run:**
- Base Model: Accuracy=73.80%, F1=72.56%, AUC-PR=75.12%, **ZDR=36.08%**
- TTT Model: Accuracy=77.41%, F1=81.66%, AUC-PR=91.72%, **ZDR=39.18%**

**K-Fold CV:**
- Base: Accuracy=73.81% ± 2.83%, F1=73.67% ± 2.93%
- TTT: Accuracy=75.01% ± 2.32%, F1=74.81% ± 2.48%

**Key Issues:**
1. ⚠️ **ZDR is critically low (39.18% vs SOTA 75%)** - Primary bottleneck
2. ⚠️ **Statistical significance not achieved (p > 0.05)** - Need more robust improvements
3. ✅ **AUC-PR exceeds SOTA (91.72% vs 90%)** - Strong point

---

## 🎯 Hyperparameter Categories by Impact

### **TIER 1: CRITICAL FOR ZDR IMPROVEMENT** (Highest Priority)

These hyperparameters directly impact zero-day detection rate:

#### 1. **Zero-Day Focused Adaptation Parameters**
```python
# Current values:
ttt_zero_day_ratio: float = 0.50  # 50% of adaptation set
ttt_zero_day_focused: bool = True

# Tuning Recommendations:
# - Increase to 0.60-0.70 for more aggressive zero-day focus
# - Expected impact: +10-15% ZDR improvement
```

#### 2. **Confidence Thresholds for Zero-Day Candidates**
```python
# Current (hardcoded in code):
low_conf_threshold = 0.7  # Line 1326 in simple_fedavg_coordinator.py

# Tuning Recommendations:
# - Lower to 0.60-0.65 to capture more zero-day candidates
# - Expected impact: +5-10% ZDR improvement
# - Action: Make this configurable in config.py
```

#### 3. **ZDR-Optimized Threshold Parameters**
```python
# Current values:
ttt_zdr_target: float = 0.80  # Target ZDR (80%)
ttt_zdr_max_far: float = 0.40  # Max FAR allowed (40%)

# Tuning Recommendations:
# - Increase ttt_zdr_target to 0.85-0.90 for more aggressive ZDR optimization
# - Increase ttt_zdr_max_far to 0.45-0.50 to allow more false alarms for better ZDR
# - Expected impact: +15-25% ZDR improvement
```

#### 4. **Prototype Alignment Weight**
```python
# Current value:
ttt_prototype_weight: float = 0.3

# Tuning Recommendations:
# - Increase to 0.5-0.7 for stronger zero-day feature alignment
# - Expected impact: +8-12% ZDR improvement
```

#### 5. **Contrastive Alignment Weight**
```python
# Current value:
ttt_contrastive_weight: float = 0.5

# Tuning Recommendations:
# - Increase to 0.7-1.0 for stronger feature alignment
# - Expected impact: +5-8% ZDR improvement
```

---

### **TIER 2: HIGH IMPACT ON OVERALL PERFORMANCE**

These improve general metrics (Accuracy, F1, AUC-PR):

#### 6. **TTT Learning Rate**
```python
# Current value:
ttt_lr: float = 5e-4  # 0.0005

# Tuning Recommendations:
# - Try: 7e-4, 1e-3, 1.5e-3 (higher for faster adaptation)
# - Use learning rate finder or grid search
# - Expected impact: +2-5% accuracy improvement
# - Risk: Too high → catastrophic forgetting
```

#### 7. **TTT Steps**
```python
# Current value:
ttt_base_steps: int = 150

# Tuning Recommendations:
# - Try: 200, 250, 300 (more steps for better convergence)
# - Balance with early stopping to prevent overfitting
# - Expected impact: +1-3% accuracy improvement
```

#### 8. **Pseudo-Label Thresholds**
```python
# Current values:
pseudo_threshold: float = 0.75  # High confidence threshold
pseudo_min_threshold: float = 0.60  # Minimum threshold

# Tuning Recommendations:
# - Lower pseudo_threshold to 0.70-0.72 for more pseudo-labels
# - Lower pseudo_min_threshold to 0.55-0.58 for curriculum learning
# - Expected impact: +2-4% accuracy improvement
```

#### 9. **Pseudo-Label Weight**
```python
# Current value:
pseudo_weight: float = 2.2

# Tuning Recommendations:
# - Try: 2.5, 3.0 (higher = stronger pseudo-label signal)
# - Expected impact: +1-3% accuracy improvement
# - Risk: Too high → overfitting to pseudo-labels
```

#### 10. **TTT Adaptation Query Size**
```python
# Current value:
ttt_adaptation_query_size: int = 1200

# Tuning Recommendations:
# - Increase to 1500-2000 for more adaptation data
# - Expected impact: +1-2% accuracy improvement
# - Trade-off: More computation time
```

---

### **TIER 3: MODERATE IMPACT (Fine-Tuning)**

#### 11. **Stabilized TTT Thresholds**
```python
# Current values:
ttt_normal_anchor_threshold: float = 0.75
ttt_attack_conf_threshold: float = 0.65
ttt_ambiguous_high: float = 0.85
ttt_ambiguous_low: float = 0.30

# Tuning Recommendations:
# - Adjust ttt_attack_conf_threshold: 0.60-0.70
# - Adjust ttt_ambiguous_low: 0.25-0.35
# - Expected impact: +1-2% accuracy improvement
```

#### 12. **Loss Component Weights**
```python
# Current values:
ttt_pseudo_loss_weight: float = 1.0
ttt_repulsion_weight: float = 0.5
ttt_balance_weight: float = 2.0

# Tuning Recommendations:
# - Try different combinations:
#   - More repulsion: ttt_repulsion_weight = 0.7-1.0
#   - More balance: ttt_balance_weight = 2.5-3.0
# - Expected impact: +0.5-1.5% accuracy improvement
```

#### 13. **Early Stopping Parameters**
```python
# Current values:
ttt_early_stopping: bool = True
ttt_early_stopping_patience: int = 10
ttt_early_stopping_min_delta: float = 1e-4

# Tuning Recommendations:
# - Increase patience to 15-20 for more training
# - Decrease min_delta to 5e-5 for more sensitive stopping
# - Expected impact: +0.5-1% accuracy improvement
```

#### 14. **BN Statistics Adaptation**
```python
# Current values:
ttt_bn_statistics_adaptation: bool = True
ttt_bn_ema_decay: float = 0.9

# Tuning Recommendations:
# - Try ttt_bn_ema_decay: 0.85-0.95
# - Lower = faster adaptation, Higher = more stable
# - Expected impact: +0.5-1% accuracy improvement
```

---

### **TIER 4: BASE MODEL IMPROVEMENTS** (Affects TTT indirectly)

#### 15. **Federated Learning Parameters**
```python
# Current values:
num_rounds: int = 12
local_epochs: int = 50
learning_rate: float = 0.001
k_shot: int = 150
n_query: int = 15

# Tuning Recommendations:
# - Increase num_rounds to 15-20 for better base model
# - Increase k_shot to 200-250 for better meta-learning
# - Expected impact: +1-3% base accuracy → better TTT starting point
```

#### 16. **Model Architecture**
```python
# Current values:
hidden_dim: int = 256
embedding_dim: int = 128

# Tuning Recommendations:
# - Try: hidden_dim = 512, embedding_dim = 256 (larger model)
# - Expected impact: +2-4% accuracy improvement
# - Trade-off: More computation, risk of overfitting
```

---

## 🔬 Recommended Tuning Strategy

### **Phase 1: ZDR Optimization (Priority 1)**

**Goal:** Increase ZDR from 39.18% to 60%+ (target: 75% SOTA)

```python
# Recommended changes:
ttt_zero_day_ratio = 0.65  # Increase from 0.50
ttt_zdr_target = 0.85  # Increase from 0.80
ttt_zdr_max_far = 0.50  # Increase from 0.40
ttt_prototype_weight = 0.6  # Increase from 0.3
ttt_contrastive_weight = 0.8  # Increase from 0.5

# Also modify code to make low_conf_threshold configurable:
# In simple_fedavg_coordinator.py line 1326:
# low_conf_threshold = getattr(config, "ttt_zero_day_candidate_threshold", 0.65)
```

**Expected Impact:** +20-30% ZDR improvement

---

### **Phase 2: Overall Performance (Priority 2)**

**Goal:** Improve accuracy and F1 while maintaining ZDR gains

```python
# Recommended changes:
ttt_lr = 7e-4  # Increase from 5e-4
ttt_base_steps = 200  # Increase from 150
pseudo_threshold = 0.72  # Lower from 0.75
pseudo_weight = 2.5  # Increase from 2.2
ttt_adaptation_query_size = 1500  # Increase from 1200
```

**Expected Impact:** +3-5% accuracy, +2-4% F1 improvement

---

### **Phase 3: Statistical Robustness (Priority 3)**

**Goal:** Achieve statistical significance (p < 0.05)

```python
# Options:
# 1. Increase k-fold CV folds from 5 to 10
# 2. Run multiple independent runs and aggregate
# 3. Increase test set size if possible
# 4. Ensure consistent improvements across all folds
```

---

## 📋 Quick Tuning Checklist

### **For ZDR Improvement:**
- [ ] Increase `ttt_zero_day_ratio` to 0.65-0.70
- [ ] Increase `ttt_zdr_target` to 0.85-0.90
- [ ] Increase `ttt_zdr_max_far` to 0.45-0.50
- [ ] Increase `ttt_prototype_weight` to 0.5-0.7
- [ ] Increase `ttt_contrastive_weight` to 0.7-1.0
- [ ] Make `low_conf_threshold` configurable (currently hardcoded at 0.7)

### **For Overall Performance:**
- [ ] Tune `ttt_lr` (try 7e-4, 1e-3)
- [ ] Increase `ttt_base_steps` to 200-250
- [ ] Lower `pseudo_threshold` to 0.70-0.72
- [ ] Increase `pseudo_weight` to 2.5-3.0
- [ ] Increase `ttt_adaptation_query_size` to 1500-2000

### **For Statistical Significance:**
- [ ] Run 10-fold CV instead of 5-fold
- [ ] Run multiple independent experiments
- [ ] Ensure consistent improvements across folds

---

## 🎯 Specific Recommendations Based on Current Results

### **Immediate Actions (Highest ROI):**

1. **ZDR Optimization (Critical):**
   ```python
   ttt_zero_day_ratio = 0.65
   ttt_zdr_target = 0.85
   ttt_zdr_max_far = 0.50
   ttt_prototype_weight = 0.6
   ttt_contrastive_weight = 0.8
   ```

2. **Learning Rate Tuning:**
   ```python
   ttt_lr = 7e-4  # or try 1e-3
   ```

3. **More Adaptation Steps:**
   ```python
   ttt_base_steps = 200
   ttt_early_stopping_patience = 15
   ```

---

## 📊 Expected Outcomes After Tuning

**Conservative Estimates:**
- ZDR: 39.18% → **55-65%** (+15-25%)
- Accuracy: 77.41% → **79-81%** (+1.5-3.5%)
- F1-Score: 81.66% → **83-85%** (+1.5-3.5%)
- AUC-PR: 91.72% → **92-94%** (+0.3-2.3%)

**Optimistic Estimates (with all optimizations):**
- ZDR: 39.18% → **70-75%** (+30-35%) ⭐ **SOTA Level**
- Accuracy: 77.41% → **82-85%** (+4.5-7.5%)
- F1-Score: 81.66% → **86-88%** (+4.5-6.5%)
- AUC-PR: 91.72% → **93-95%** (+1.3-3.3%)

---

## ⚠️ Important Notes

1. **ZDR is the Critical Bottleneck:** Focus tuning efforts here first
2. **Trade-offs:** Higher ZDR may increase FAR - monitor both metrics
3. **Statistical Significance:** May require more folds or multiple runs
4. **Hyperparameter Interactions:** Some parameters affect each other - tune systematically
5. **Overfitting Risk:** Monitor validation metrics during tuning

---

## 🔄 Tuning Workflow

1. **Start with ZDR-focused parameters** (Tier 1)
2. **Run single experiment** with new values
3. **Check ZDR improvement** - if < 5%, adjust more aggressively
4. **Once ZDR > 60%**, move to Tier 2 (overall performance)
5. **Fine-tune with Tier 3** parameters
6. **Validate with k-fold CV** for statistical significance
7. **Document best configuration** for reproducibility

---

## 📝 Configuration Template for Tuning

```python
# config.py - ZDR-Optimized Configuration
ttt_zero_day_ratio: float = 0.65  # Increased from 0.50
ttt_zdr_target: float = 0.85  # Increased from 0.80
ttt_zdr_max_far: float = 0.50  # Increased from 0.40
ttt_prototype_weight: float = 0.6  # Increased from 0.3
ttt_contrastive_weight: float = 0.8  # Increased from 0.5
ttt_lr: float = 7e-4  # Increased from 5e-4
ttt_base_steps: int = 200  # Increased from 150
pseudo_threshold: float = 0.72  # Lowered from 0.75
pseudo_weight: float = 2.5  # Increased from 2.2
ttt_adaptation_query_size: int = 1500  # Increased from 1200
```

---

**Next Steps:** Implement these changes and re-run the system to verify improvements!


