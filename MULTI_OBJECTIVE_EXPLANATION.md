# Multi-Objective Optimization Configuration

## 🎯 **Balanced Multi-Objective Function**

The `multi_objective` metric optimizes for **three balanced components**:

### **Component 1: Zero-Day Detection Rate (ZDR) - 30% Weight**
- **What it measures:** How well the TTT-adapted model detects **unseen zero-day attacks**
- **Why it's important:** Core capability for identifying novel attack types
- **Weight:** 30% (important but not dominant)

### **Component 2: Non-Zero-Day F1-Score - 35% Weight**
- **What it measures:** How well the model detects **known attack types** (attacks seen during training)
- **Why it's important:** IDS must also detect known attacks effectively
- **Weight:** 35% (equally important as overall F1)

### **Component 3: Overall F1-Score - 35% Weight**
- **What it measures:** Overall performance on **all test samples** (Normal + Known Attacks + Zero-Day)
- **Why it's important:** Ensures balanced performance across all classes
- **Weight:** 35% (equally important as non-zero-day F1)

---

## 📊 **Mathematical Formula**

```
Multi-Objective Score = 
    (0.30 × Zero-Day ZDR) + 
    (0.35 × Non-Zero-Day F1) + 
    (0.35 × Overall F1)
```

All components are normalized to 0-1 range before weighting.

---

## ✅ **Why This Balance?**

### **Problem with Zero-Day-Only Optimization:**
- ❌ Overfits to zero-day patterns
- ❌ Degrades performance on known attacks
- ❌ High false positive rate on normal traffic
- ❌ Not suitable for production IDS

### **Problem with Overall-Only Optimization:**
- ❌ Ignores zero-day detection capability
- ❌ Optimizes for "easier" known attacks
- ❌ Fails on novel attack types

### **Solution: Balanced Multi-Objective (30/35/35)**
- ✅ Detects both zero-day AND known attacks
- ✅ Maintains balanced overall performance
- ✅ Suitable for production IDS deployment
- ✅ Prevents overfitting to a single metric

---

## 🔍 **How It Works During Optimization**

For each trial:

1. **Base Model Evaluation:**
   - Evaluated on test set (40% Normal, 35% Known Attacks, 25% Zero-Day)
   - Metrics: Accuracy, F1, AUC-PR, ZDR, Non-Zero-Day F1

2. **TTT-Adapted Model Evaluation:**
   - Same test set
   - Metrics: Accuracy, F1, AUC-PR, ZDR, Non-Zero-Day F1

3. **Multi-Objective Score Calculation:**
   ```python
   zdr_score = ttt_zdr  # Zero-day detection rate (0.0-1.0)
   non_zero_day_f1_score = ttt_non_zero_day_f1  # Known attack F1 (0.0-1.0)
   overall_f1_score = ttt_f1  # Overall F1 (0.0-1.0)
   
   metric_value = (
       0.30 * zdr_score +
       0.35 * non_zero_day_f1_score +
       0.35 * overall_f1_score
   )
   ```

4. **Optuna Optimization:**
   - Maximizes `metric_value`
   - Explores hyperparameter space to find best balanced configuration

---

## 📈 **Example Calculation**

If a trial achieves:
- **Zero-Day ZDR:** 0.85 (85% detection)
- **Non-Zero-Day F1:** 0.78 (78% F1 on known attacks)
- **Overall F1:** 0.82 (82% overall F1)

**Multi-Objective Score:**
```
= (0.30 × 0.85) + (0.35 × 0.78) + (0.35 × 0.82)
= 0.255 + 0.273 + 0.287
= 0.815
```

**Score: 0.815 (81.5%)**

---

## 🚀 **Optimization Command**

```bash
python optimize_hyperparameters.py --metric multi_objective --n_trials 20
```

This will:
- Run 20 trials
- Optimize for balanced multi-objective score
- Find hyperparameters that work well for BOTH zero-day and known attacks
- Save best configuration to `best_hyperparameters.json`

---

## 📝 **Logging Output**

During optimization, you'll see:

```
🎯 Balanced Multi-Objective Score (multi_objective):
  Components (balanced for both zero-day AND known attacks):
    Zero-day ZDR: 0.8500 × 0.30 = 0.2550
    Non-zero-day F1: 0.7800 × 0.35 = 0.2730
    Overall F1: 0.8200 × 0.35 = 0.2870
  Combined Score: 0.8150
  📊 Balance: 30% zero-day + 35% known attacks + 35% overall = 100%
```

---

## ✅ **Summary**

The multi-objective function ensures that the optimized hyperparameters produce a model that:
1. ✅ Detects zero-day attacks effectively (30% weight)
2. ✅ Detects known attacks effectively (35% weight)
3. ✅ Maintains overall balanced performance (35% weight)

This is **ideal for production IDS** that needs to handle both known and unknown threats.









