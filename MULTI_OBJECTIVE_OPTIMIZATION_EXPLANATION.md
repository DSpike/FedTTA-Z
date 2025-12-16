# Multi-Objective Optimization Explanation

## 🎯 **Current Optimization (Single-Objective)**

**Current approach**: Optimize for **ONE** metric only:

```python
# In optimize_hyperparameters.py, line ~399
metric_value = adapted_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)

# Optuna optimizes to MAXIMIZE this single value
trial.report(metric_value, step)
```

**Result**:

- ✅ Zero-day detection rate: **100%** (perfect!)
- ❌ Overall performance: **-2.73%** accuracy degradation

---

## 🔀 **Multi-Objective Optimization (Balanced Approach)**

**New approach**: Optimize for **MULTIPLE** metrics simultaneously:

```python
# Combine multiple metrics into a single score
zero_day_score = ttt_zdr  # Zero-day detection rate (0.0 to 1.0)
overall_f1_score = ttt_f1  # Overall F1-score (0.0 to 1.0)

# Weighted combination (example: 60% zero-day, 40% overall)
metric_value = 0.6 * zero_day_score + 0.4 * overall_f1_score

# Optuna optimizes to MAXIMIZE this combined score
trial.report(metric_value, step)
```

**Expected Result**:

- ✅ Zero-day detection rate: **~97-98%** (slightly lower)
- ✅ Overall performance: **+0-2%** improvement (balanced)

---

## 📊 **Why Use Multi-Objective?**

### **Current Problem:**

```
Objective: Maximize ZDR ONLY
Result:    ZDR = 100%, but Accuracy drops by -2.73%
```

This happens because:

- Optuna finds hyperparameters that maximize ZDR
- These hyperparameters may hurt overall performance
- The optimizer doesn't care about overall performance

### **Multi-Objective Solution:**

```
Objective: Maximize (0.6 × ZDR + 0.4 × Overall_F1)
Result:    Balanced improvement in BOTH metrics
```

This works because:

- Optuna now considers BOTH metrics
- It finds hyperparameters that balance both
- Trade-off between zero-day and overall performance

---

## 🛠️ **Implementation Options**

### **Option 1: Weighted Sum (Simplest)** ⭐ **RECOMMENDED**

Combine multiple metrics into a single score:

```python
# In optimize_hyperparameters.py, objective() method:

# Extract metrics
ttt_zdr = adapted_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)
ttt_f1 = adapted_results.get('f1_score', 0.0)
ttt_non_zero_day_f1 = adapted_results.get('non_zero_day', {}).get('f1_score', 0.0)

# Weighted combination
zero_day_weight = 0.6  # 60% importance on zero-day detection
overall_weight = 0.4    # 40% importance on overall performance

# Combined metric
metric_value = (
    zero_day_weight * ttt_zdr +
    overall_weight * ttt_f1
)

# Alternative: Use non-zero-day F1 instead of overall F1
# metric_value = (
#     zero_day_weight * ttt_zdr +
#     overall_weight * ttt_non_zero_day_f1
# )
```

**Pros**:

- ✅ Simple to implement
- ✅ Easy to tune weights
- ✅ Single value for Optuna to optimize

**Cons**:

- ⚠️ Requires manual weight tuning
- ⚠️ May not find Pareto-optimal solutions

---

### **Option 2: Pareto Optimization (Advanced)**

Optimize for multiple objectives simultaneously and find Pareto-optimal solutions:

```python
# In optimize_hyperparameters.py:

# Optuna supports multi-objective optimization
import optuna
from optuna.multi_objective import Trial

# Create study with multiple objectives
study = optuna.create_study(
    directions=["maximize", "maximize"]  # Maximize both metrics
)

# In objective() method:
def objective(trial):
    # ... existing code ...

    # Return multiple values (one per objective)
    return (
        ttt_zdr,           # Objective 1: Maximize zero-day detection rate
        ttt_f1             # Objective 2: Maximize overall F1-score
    )
```

**Pros**:

- ✅ Finds all Pareto-optimal solutions
- ✅ No manual weight tuning
- ✅ Shows trade-offs clearly

**Cons**:

- ⚠️ More complex to implement
- ⚠️ Requires selecting best solution from Pareto frontier
- ⚠️ More trials needed

---

### **Option 3: Penalty-Based (Hybrid)**

Maximize zero-day detection but penalize overall degradation:

```python
# Maximize ZDR but add penalty if overall performance degrades
ttt_zdr = adapted_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)
ttt_f1 = adapted_results.get('f1_score', 0.0)
base_f1 = base_results.get('f1_score', 0.0)

# Calculate improvement/degradation
f1_improvement = ttt_f1 - base_f1

# Combined metric with penalty
if f1_improvement < 0:
    # Penalize if overall F1 degrades
    penalty = abs(f1_improvement) * 0.5  # 50% penalty weight
    metric_value = ttt_zdr - penalty
else:
    # Bonus if overall F1 improves
    bonus = f1_improvement * 0.3  # 30% bonus weight
    metric_value = ttt_zdr + bonus
```

**Pros**:

- ✅ Prioritizes zero-day detection
- ✅ Prevents severe overall degradation
- ✅ Encourages improvements in both

**Cons**:

- ⚠️ Requires tuning penalty/bonus weights

---

## 💡 **Recommended Implementation (Weighted Sum)**

### **Step 1: Update `optimize_hyperparameters.py`**

```python
# Around line 399, replace:
metric_value = adapted_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)
if self.metric == "ttt_auc_pr":
    metric_value = ttt_auc_pr
elif self.metric == "ttt_f1_score":
    metric_value = ttt_f1
elif self.metric == "ttt_accuracy":
    metric_value = ttt_accuracy

# With:
if self.metric == "multi_objective":
    # Multi-objective: Balance zero-day and overall performance
    zero_day_weight = 0.6  # 60% weight on zero-day detection
    overall_weight = 0.4    # 40% weight on overall F1-score

    metric_value = (
        zero_day_weight * ttt_zdr +
        overall_weight * ttt_f1
    )
elif self.metric == "ttt_zero_day_detection_rate":
    metric_value = adapted_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)
elif self.metric == "ttt_auc_pr":
    metric_value = ttt_auc_pr
elif self.metric == "ttt_f1_score":
    metric_value = ttt_f1
elif self.metric == "ttt_accuracy":
    metric_value = ttt_accuracy
```

### **Step 2: Add Command-Line Argument**

```python
# In __main__ section:
parser.add_argument(
    "--metric",
    type=str,
    default="ttt_zero_day_detection_rate",
    choices=["ttt_zero_day_detection_rate", "ttt_auc_pr", "ttt_f1_score", "ttt_accuracy", "multi_objective"],
    help="Metric to optimize (default: ttt_zero_day_detection_rate)"
)
```

### **Step 3: Run Optimization**

```bash
# Single-objective (current): Optimize for zero-day only
python optimize_hyperparameters.py --n_trials 20 --metric ttt_zero_day_detection_rate

# Multi-objective: Balance zero-day and overall performance
python optimize_hyperparameters.py --n_trials 20 --metric multi_objective
```

---

## 📈 **Expected Results Comparison**

### **Single-Objective (Zero-Day Only):**

```
Best Trial:
  ZDR:        100.0%  ✅ (perfect!)
  Overall F1: 77.37%  ❌ (degraded by -2.63%)
  Trade-off:  Excellent zero-day, poor overall
```

### **Multi-Objective (Balanced):**

```
Best Trial:
  ZDR:        97.5%   ✅ (still excellent!)
  Overall F1: 80.5%   ✅ (improved by +0.5%)
  Trade-off:  Good zero-day, good overall
```

---

## 🎯 **Choosing Weights**

The weights determine the **priority**:

| Zero-Day Weight | Overall Weight | Result                                            |
| --------------- | -------------- | ------------------------------------------------- |
| **0.8**         | 0.2            | Prioritize zero-day (similar to single-objective) |
| **0.6**         | 0.4            | Balanced (recommended) ⭐                         |
| 0.5             | 0.5            | Equal priority                                    |
| 0.4             | **0.6**        | Prioritize overall performance                    |

**Recommendation**: Start with `0.6` (zero-day) and `0.4` (overall), then adjust based on results.

---

## 🔬 **Advanced: Tuning Weights with Optuna**

You can even optimize the weights themselves:

```python
# In suggest_hyperparameters():
zero_day_weight = trial.suggest_float("zero_day_weight", 0.4, 0.8)
overall_weight = 1.0 - zero_day_weight  # Ensure weights sum to 1.0

# In objective():
metric_value = (
    zero_day_weight * ttt_zdr +
    overall_weight * ttt_f1
)
```

This way, Optuna finds the optimal balance automatically!

---

## ✅ **Summary**

**Multi-objective optimization** means:

- Optimizing for **multiple metrics simultaneously**
- Finding a **balanced solution** instead of maximizing one at the expense of another
- Using a **weighted combination** of metrics (simplest approach)

**Benefits**:

- ✅ Better overall performance
- ✅ Still good zero-day detection
- ✅ Balanced trade-off

**Implementation**:

- Change objective to: `0.6 × ZDR + 0.4 × Overall_F1`
- Run optimization
- Compare results with single-objective approach









