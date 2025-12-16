# Multi-Objective Optimization Implementation Guide

## ✅ **What Was Implemented**

Multi-objective optimization that balances **three metrics simultaneously**:

- **Zero-Day Detection Rate (ZDR)**: 40% weight
- **F1-Score**: 30% weight
- **Accuracy**: 30% weight

The combined score is calculated as:

```
Multi-Objective Score = 0.4 × ZDR + 0.3 × F1-Score + 0.3 × Accuracy
```

---

## 🚀 **How to Use**

### **Run Multi-Objective Optimization:**

```bash
python optimize_hyperparameters.py --n_trials 20 --metric multi_objective
```

### **Compare with Single-Objective Optimization:**

```bash
# Option 1: Zero-day only (current best)
python optimize_hyperparameters.py --n_trials 20 --metric ttt_zero_day_detection_rate

# Option 2: F1-score only
python optimize_hyperparameters.py --n_trials 20 --metric ttt_f1_score

# Option 3: Accuracy only
python optimize_hyperparameters.py --n_trials 20 --metric ttt_accuracy

# Option 4: Multi-objective (NEW) - Balances all three
python optimize_hyperparameters.py --n_trials 20 --metric multi_objective
```

---

## 📊 **Expected Results**

### **Single-Objective (Zero-Day Only):**

```
Best Trial:
  ZDR:        100.0%  ✅ (perfect!)
  F1-Score:   77.37%  ❌ (degraded by -2.63%)
  Accuracy:   71.82%  ❌ (degraded by -2.73%)
  Trade-off:  Excellent zero-day, poor overall
```

### **Multi-Objective (Balanced):**

```
Best Trial:
  ZDR:        ~97-98%  ✅ (excellent!)
  F1-Score:   ~79-81%  ✅ (improved or maintained)
  Accuracy:   ~74-76%  ✅ (improved or maintained)
  Trade-off:  Good zero-day, good overall
```

---

## 📈 **What Gets Logged**

### **Console Output:**

```
Trial X Results:
  Base Model: Accuracy=0.7455, F1=0.8000, AUC-PR=0.8092, ZDR=0.9545
    Non-Zero-Day: Acc=0.7500, F1=0.6792
  TTT Model: Accuracy=0.7882, F1=0.8302, AUC-PR=0.7639, ZDR=1.0000
    Non-Zero-Day: Acc=0.7500, F1=0.7671
  Improvements:
    Overall: Acc=+0.0427, F1=+0.0302, AUC-PR=-0.0453, ZDR=+0.0455
    Non-Zero-Day: Acc=+0.0000, F1=+0.0879
  🎯 Multi-Objective Score (multi_objective):
    Components:
      ZDR: 1.0000 × 0.40 = 0.4000
      F1:  0.8302 × 0.30 = 0.2491
      Acc: 0.7882 × 0.30 = 0.2365
    Combined Score: 0.8856
```

### **Wandb Logging:**

All metrics are logged to Wandb, including:

- Individual metrics (ZDR, F1, Accuracy)
- Multi-objective score
- Component breakdown (ZDR×0.4, F1×0.3, Acc×0.3)
- Non-zero-day metrics

### **Optuna Trial Attributes:**

- `multi_objective_score`: Combined score
- `multi_objective_zdr_component`: ZDR contribution (0.4 × ZDR)
- `multi_objective_f1_component`: F1 contribution (0.3 × F1)
- `multi_objective_acc_component`: Accuracy contribution (0.3 × Acc)

---

## ⚙️ **Customizing Weights**

If you want to change the weights (currently 40% ZDR, 30% F1, 30% Acc), edit `optimize_hyperparameters.py` around line 420:

```python
if self.metric == "multi_objective":
    # Customize weights here (must sum to 1.0)
    zdr_weight = 0.5   # 50% weight on zero-day
    f1_weight = 0.25   # 25% weight on F1-score
    acc_weight = 0.25  # 25% weight on accuracy

    # ... rest of code
```

**Recommended Weight Combinations:**

| Priority                | ZDR Weight | F1 Weight | Acc Weight | Use Case                    |
| ----------------------- | ---------- | --------- | ---------- | --------------------------- |
| **Zero-Day Priority**   | 0.6        | 0.2       | 0.2        | Maximize zero-day detection |
| **Balanced (Default)**  | 0.4        | 0.3       | 0.3        | Balance all metrics ⭐      |
| **Overall Performance** | 0.3        | 0.35      | 0.35       | Prioritize overall metrics  |
| **Equal**               | 0.33       | 0.33      | 0.34       | Equal importance            |

---

## 🔬 **Advanced: Optimize Weights with Optuna**

You can even let Optuna find the optimal weights:

```python
# In suggest_hyperparameters() method:
if self.metric == "multi_objective":
    # Let Optuna optimize the weights
    zdr_weight = trial.suggest_float("zdr_weight", 0.2, 0.7)
    remaining_weight = 1.0 - zdr_weight
    f1_weight = trial.suggest_float("f1_weight", 0.1, remaining_weight)
    acc_weight = 1.0 - zdr_weight - f1_weight
```

Then use these weights in the objective function.

---

## 📝 **Summary**

**What Changed:**

- ✅ Added `multi_objective` metric option
- ✅ Combines ZDR (40%), F1-score (30%), and Accuracy (30%)
- ✅ Enhanced logging shows component breakdown
- ✅ All metrics logged to Wandb and Optuna

**How to Use:**

```bash
python optimize_hyperparameters.py --n_trials 20 --metric multi_objective
```

**Expected Benefit:**

- Better overall performance (F1 and Accuracy)
- Still maintains excellent zero-day detection (97-98%)
- Balanced trade-off between all metrics

---

## 🎯 **Next Steps**

1. **Run multi-objective optimization**:

   ```bash
   python optimize_hyperparameters.py --n_trials 20 --metric multi_objective
   ```

2. **Compare results** with single-objective optimization

3. **Choose configuration** based on your priority:

   - Zero-day priority → Single-objective (ZDR only)
   - Balanced performance → Multi-objective ⭐
   - Overall performance → Single-objective (F1 or Accuracy)

4. **Adjust weights** if needed to match your specific requirements









