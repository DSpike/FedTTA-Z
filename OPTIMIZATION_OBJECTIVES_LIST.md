# Optimization Objectives List

## 📊 **Available Optimization Metrics**

The hyperparameter optimization supports **5 different objective functions**. Each optimizes for different aspects of the IDS performance:

---

### **1. Zero-Day Detection Rate Only**
**Metric Name:** `ttt_zero_day_detection_rate`  
**Default:** ✅ (Default metric)

**What it optimizes:**
- Zero-Day Detection Rate (ZDR) from TTT-adapted model
- Formula: `TP / (TP + FN)` for zero-day attacks only
- Range: 0.0 - 1.0 (0% - 100%)

**Use case:**
- When your primary goal is detecting **novel/unknown attacks**
- Research focused on zero-day detection capability
- **Warning:** May sacrifice performance on known attacks

**Optimization direction:** `maximize`

---

### **2. AUC-PR (Area Under Precision-Recall Curve)**
**Metric Name:** `ttt_auc_pr`

**What it optimizes:**
- Area Under the Precision-Recall Curve for TTT-adapted model
- Best metric for **imbalanced datasets** (typical in IDS)
- Range: 0.0 - 1.0 (0% - 100%)

**Use case:**
- When you have **imbalanced data** (few attacks, many normal samples)
- When precision and recall balance is important
- Standard metric for cybersecurity/IDS research

**Optimization direction:** `maximize`

---

### **3. F1-Score**
**Metric Name:** `ttt_f1_score`

**What it optimizes:**
- Harmonic mean of Precision and Recall: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- Overall balanced performance metric
- Range: 0.0 - 1.0 (0% - 100%)

**Use case:**
- When you want balanced precision and recall
- General-purpose IDS optimization
- Good for production systems

**Optimization direction:** `maximize`

---

### **4. Accuracy**
**Metric Name:** `ttt_accuracy`

**What it optimizes:**
- Overall classification accuracy: `(TP + TN) / (TP + TN + FP + FN)`
- Simple percentage of correct predictions
- Range: 0.0 - 1.0 (0% - 100%)

**Use case:**
- Simple optimization target
- When class balance is reasonable
- **Warning:** Can be misleading with imbalanced data

**Optimization direction:** `maximize`

---

### **5. Multi-Objective (BALANCED) ⭐ RECOMMENDED**
**Metric Name:** `multi_objective`

**What it optimizes:**
- **Balanced combination** of three metrics:
  - **30%** Zero-Day Detection Rate (ZDR)
  - **35%** Non-Zero-Day F1-Score (Known attacks)
  - **35%** Overall F1-Score (All classes)

**Formula:**
```python
multi_objective_score = (
    0.30 × Zero-Day_ZDR +
    0.35 × Non-Zero-Day_F1 +
    0.35 × Overall_F1
)
```

**Why balanced?**
- ✅ Optimizes for **BOTH** zero-day AND known attack detection
- ✅ Prevents overfitting to zero-day only
- ✅ Suitable for **production IDS** systems
- ✅ Maintains overall balanced performance

**Use case:**
- **Production IDS deployment** (detects both known and unknown attacks)
- When you need comprehensive threat detection
- **Recommended** for most real-world scenarios

**Optimization direction:** `maximize`

---

## 🎯 **How to Use**

### **Command Line:**
```bash
# Zero-day detection only (default)
python optimize_hyperparameters.py --metric ttt_zero_day_detection_rate --n_trials 20

# AUC-PR (best for imbalanced data)
python optimize_hyperparameters.py --metric ttt_auc_pr --n_trials 20

# F1-Score (balanced performance)
python optimize_hyperparameters.py --metric ttt_f1_score --n_trials 20

# Accuracy (simple metric)
python optimize_hyperparameters.py --metric ttt_accuracy --n_trials 20

# Multi-objective (RECOMMENDED - balanced)
python optimize_hyperparameters.py --metric multi_objective --n_trials 20
```

### **In Python Code:**
```python
from optimize_hyperparameters import HyperparameterOptimizer

# Multi-objective optimization (recommended)
optimizer = HyperparameterOptimizer(
    study_name="balanced_ids_optimization",
    n_trials=20,
    direction="maximize",
    metric="multi_objective"  # Use balanced multi-objective
)

best_trial = optimizer.optimize()
```

---

## 📋 **Quick Comparison**

| Metric | Zero-Day Focus | Known Attacks | Overall Balance | Best For |
|--------|---------------|---------------|-----------------|----------|
| `ttt_zero_day_detection_rate` | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | Research (zero-day only) |
| `ttt_auc_pr` | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Imbalanced datasets |
| `ttt_f1_score` | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General IDS |
| `ttt_accuracy` | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Simple optimization |
| `multi_objective` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Production IDS** ⭐ |

---

## 💡 **Recommendations**

### **For Research (Zero-Day Detection):**
```bash
--metric ttt_zero_day_detection_rate
```

### **For Production IDS (Comprehensive Detection):**
```bash
--metric multi_objective  # RECOMMENDED ⭐
```

### **For Imbalanced Data:**
```bash
--metric ttt_auc_pr
```

---

## 📝 **Current Configuration**

From your recent optimization run:
- **Metric Used:** `multi_objective`
- **Weights:** 30% ZDR + 35% Non-Zero-Day F1 + 35% Overall F1
- **Best Trial:** Trial 5 with score: 0.8189
- **Result:** Balanced hyperparameters suitable for production IDS

---

## 🔍 **Multi-Objective Breakdown Example**

From a typical trial log:
```
🎯 Balanced Multi-Objective Score (multi_objective):
  Components (balanced for both zero-day AND known attacks):
    Zero-day ZDR: 0.8500 × 0.30 = 0.2550
    Non-zero-day F1: 0.7800 × 0.35 = 0.2730
    Overall F1: 0.8200 × 0.35 = 0.2870
  Combined Score: 0.8150
  📊 Balance: 30% zero-day + 35% known attacks + 35% overall = 100%
```

This shows how the balanced multi-objective ensures all three aspects are considered.









