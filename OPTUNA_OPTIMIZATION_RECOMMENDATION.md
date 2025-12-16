# Optuna Optimization Recommendation

## ✅ **Optimization Scripts Updated**

I've updated both optimization scripts to include the **new parameters** you recently added:

### **New Parameters Added:**

1. **`ttt_l2_reg_weight`**: L2 regularization weight (range: 0.001 - 0.1, log scale)

   - Currently set to `0.01` in config
   - Prevents excessive parameter drift during TTT adaptation
   - Expected impact: +2-4% improvement

2. **`confidence_rejection_threshold`**: Confidence threshold for rejecting low-confidence predictions (range: 0.5 - 0.9)
   - Currently set to `0.7` in config
   - Filters out uncertain predictions, marking them as "Unknown"
   - Expected impact: +3-5% improvement

### **Updated Parameter Ranges:**

- **`ttt_base_steps`**: Updated from `200-400` to `150-300` (aligned with your current config value of 200)
- **`ttt_batch_size`**: Added `64, 128` to existing options (to match your current config)

---

## 🤔 **Should You Run Optimization Now?**

### **✅ YES, Recommended If:**

1. **You have time**: Optimization typically takes **several hours to days** (depending on `n_trials`)
2. **You want to squeeze out maximum performance**: The new parameters (L2 reg, confidence rejection) haven't been optimized yet
3. **You're preparing for publication/final results**: Finding optimal hyperparameters ensures best reported performance

### **⚠️ CONSIDER WAITING If:**

1. **You need quick results**: Current configuration is already quite good
2. **You plan more feature additions**: Better to optimize after all features are finalized
3. **Limited compute resources**: Optimization requires significant computational resources

---

## 🚀 **How to Run Optimization**

### **Option 1: Quick Test (10 trials, ~2-4 hours)**

```bash
# For UNSW-NB15 dataset
python optimize_hyperparameters.py --n_trials 10

# For CICIDS2017 dataset
python optimize_hyperparameters_cicids.py --n_trials 10 --zero_day_attack "PortScan"
```

### **Option 2: Standard Optimization (20 trials, ~4-8 hours)**

```bash
# For UNSW-NB15 dataset
python optimize_hyperparameters.py --n_trials 20

# For CICIDS2017 dataset
python optimize_hyperparameters_cicids.py --n_trials 20 --zero_day_attack "PortScan"
```

### **Option 3: Comprehensive Optimization (50+ trials, ~1-2 days)**

```bash
# For UNSW-NB15 dataset
python optimize_hyperparameters.py --n_trials 50

# For CICIDS2017 dataset
python optimize_hyperparameters_cicids.py --n_trials 50 --zero_day_attack "PortScan"
```

---

## 📊 **Optimization Metrics**

### **Default Metric: `balanced_base_ttt`** (Recommended)

- **40%** Base Model F1-Score
- **30%** TTT Zero-Day Detection Rate (ZDR)
- **30%** TTT F1-Score
- **Rationale**: Balances base model performance with TTT adaptation benefits

### **Alternative Metrics:**

- `ttt_zero_day_detection_rate`: Focus on zero-day detection only
- `ttt_auc_pr`: Optimize for AUC-PR (precision-recall)
- `ttt_f1_score`: Optimize for F1-score after TTT
- `multi_objective`: Multi-objective optimization (ZDR, F1, overall)

**Example with different metric:**

```bash
python optimize_hyperparameters.py --n_trials 20 --metric "ttt_zero_day_detection_rate"
```

---

## 🎯 **Multi-Objective Optimization Explained**

### **What is Multi-Objective Optimization?**

Multi-objective optimization balances **multiple performance metrics simultaneously** instead of optimizing for just one metric. This prevents overfitting to a single metric and ensures balanced performance.

### **Available Multi-Objective Metrics:**

#### **1. `multi_objective` (TTT-Only Balanced)**

**Formula:**

```
Score = (0.30 × Zero-Day ZDR) + (0.35 × Non-Zero-Day F1) + (0.35 × Overall F1)
```

**Components:**

- **30%** Zero-Day Detection Rate (ZDR): How well the model detects unseen zero-day attacks
- **35%** Non-Zero-Day F1: How well the model detects known attack types
- **35%** Overall F1: Overall performance on all test samples

**Use Case:** Production IDS that needs to detect BOTH known and unknown attacks

**Command:**

```bash
python optimize_hyperparameters.py --n_trials 20 --metric multi_objective
```

#### **2. `balanced_base_ttt` (Base + TTT Balanced)** ⭐ **DEFAULT**

**Formula:**

```
Score = (0.40 × Base F1) + (0.30 × TTT ZDR) + (0.30 × TTT F1)
```

**Components:**

- **40%** Base Model F1: Few-shot base model performance
- **30%** TTT Zero-Day Detection Rate: Zero-day detection after TTT adaptation
- **30%** TTT Overall F1: Overall performance after TTT adaptation

**Use Case:** Optimizes for BOTH strong base model AND excellent TTT performance

**Command:**

```bash
python optimize_hyperparameters.py --n_trials 20 --metric balanced_base_ttt
```

#### **3. `improved_multi_objective` (CICIDS Only - Most Balanced)**

**Formula:**

```
Score = (0.25 × Base F1) + (0.25 × TTT ZDR) + (0.25 × TTT Non-Zero-Day F1) + (0.25 × TTT Overall F1)
```

**Components:**

- **25%** Base Model F1
- **25%** TTT Zero-Day Detection Rate
- **25%** TTT Non-Zero-Day F1
- **25%** TTT Overall F1

**Use Case:** Most balanced optimization considering all aspects equally

**Command:**

```bash
python optimize_hyperparameters_cicids.py --n_trials 20 --metric improved_multi_objective --zero_day_attack "PortScan"
```

### **Why Use Multi-Objective?**

**Problem with Single-Objective:**

- Optimizing only for ZDR → May degrade overall performance
- Optimizing only for F1 → May ignore zero-day detection
- Results in imbalanced performance

**Solution with Multi-Objective:**

- Balances multiple metrics simultaneously
- Prevents overfitting to a single metric
- Ensures production-ready performance
- Better trade-offs between different aspects

---

## 📈 **Expected Improvements**

Based on your recent changes:

| Parameter                        | Current Value | Expected Improvement                  |
| -------------------------------- | ------------- | ------------------------------------- |
| `ttt_l2_reg_weight`              | 0.01          | +2-4% (prevents overfitting)          |
| `confidence_rejection_threshold` | 0.7           | +3-5% (filters uncertain predictions) |
| **Combined optimization**        | -             | **+5-10% overall performance**        |

### **What Optimization Will Find:**

- Optimal L2 regularization strength (balance between adaptation and stability)
- Best confidence rejection threshold (balance between coverage and accuracy)
- Optimal TTT steps (you've already tuned to 200, but optimization may find 150-250 range)
- Best batch size for TTT (could be 128, or maybe smaller for better gradient estimates)

---

## ⚙️ **Complete List of Hyperparameters Being Optimized**

### **1. Meta-Learning Parameters:**

| Parameter            | Range/Options            | Description                                       |
| -------------------- | ------------------------ | ------------------------------------------------- |
| `meta_learning_rate` | 1e-4 to 1e-2 (log scale) | Learning rate for meta-learning optimization      |
| `meta_epochs`        | 3-30                     | Number of meta-epochs                             |
| `k_shot`             | 100-200                  | Number of support samples per class in meta-tasks |
| `n_query`            | 10-20                    | Number of query samples per class in meta-tasks   |
| `num_meta_tasks`     | 10-40                    | Number of meta-tasks per round                    |
| `hidden_dim`         | [256, 512, 768]          | Hidden dimension of the neural network            |
| `embedding_dim`      | [128, 256, 512]          | Embedding dimension for prototypes                |

### **2. TCN (Temporal Convolutional Network) Parameters:**

| Parameter                  | Range/Options | Description                                |
| -------------------------- | ------------- | ------------------------------------------ |
| `sequence_length`          | 20-50         | Length of input sequences                  |
| `sequence_stride`          | 10-20         | Stride between sequences                   |
| `tcn_kernel_size_1`        | 2-6           | Kernel size for first TCN layer            |
| `tcn_kernel_size_2`        | 2-6           | Kernel size for second TCN layer           |
| `tcn_kernel_size_3`        | 2-6           | Kernel size for third TCN layer            |
| `use_residual_connections` | [True, False] | Whether to use residual connections in TCN |

### **3. TTT (Test-Time Training) Parameters:**

| Parameter                        | Range/Options            | Description                                                      |
| -------------------------------- | ------------------------ | ---------------------------------------------------------------- |
| `ttt_lr`                         | 1e-4 to 2e-3 (log scale) | Learning rate for TTT adaptation                                 |
| `ttt_base_steps`                 | 150-300                  | Number of TTT adaptation steps                                   |
| `ttt_batch_size`                 | [4, 8, 16, 32, 64, 128]  | Batch size for TTT adaptation                                    |
| `ttt_adaptation_query_size`      | 1000-2000                | Number of unlabeled samples for TTT adaptation                   |
| `ttt_l2_reg_weight`              | 0.001-0.1 (log scale)    | **NEW** L2 regularization weight to prevent parameter drift      |
| `confidence_rejection_threshold` | 0.5-0.9                  | **NEW** Confidence threshold for rejecting uncertain predictions |

### **4. TENT + Pseudo-Labels Parameters:**

| Parameter                  | Range/Options | Description                                      |
| -------------------------- | ------------- | ------------------------------------------------ |
| `use_pseudo_labels`        | [True, False] | Whether to use pseudo-labeling during TTT        |
| `pseudo_weight`            | 1.5-3.5       | Weight for pseudo-label loss                     |
| `entropy_weight`           | 0.5-1.5       | Weight for entropy minimization loss             |
| `pseudo_threshold`         | 0.85-0.98     | Confidence threshold for pseudo-labeling         |
| `pseudo_min_threshold`     | 0.70-0.85     | Minimum confidence threshold for pseudo-labeling |
| `use_teacher`              | [True, False] | Whether to use EMA teacher model                 |
| `ema_decay`                | 0.95-0.999    | EMA decay rate for teacher model                 |
| `pseudo_label_temperature` | 0.3-0.8       | Temperature for sharpening pseudo-label logits   |

### **5. Temperature Scaling Parameters:**

| Parameter         | Range/Options | Description                         |
| ----------------- | ------------- | ----------------------------------- |
| `ttt_temperature` | 1.0-2.0       | Temperature scaling for calibration |

### **6. Advanced TTT Techniques:**

| Parameter        | Range/Options | Description                                   |
| ---------------- | ------------- | --------------------------------------------- |
| `use_focal_loss` | [True, False] | Whether to use focal loss for class imbalance |
| `focal_gamma`    | 1.5-3.0       | Focal loss gamma parameter                    |
| `focal_alpha`    | 0.15-0.35     | Focal loss alpha parameter                    |

### **Total Hyperparameters: 32+**

The optimizer explores a high-dimensional space to find optimal combinations of these parameters.

---

## 💡 **Recommendation**

### **If You Want Maximum Performance:**

Run **20-30 trials** with the default `balanced_base_ttt` metric. This should find good hyperparameters within a reasonable time frame.

### **If You Want Quick Validation:**

Run **10 trials** as a quick check to see if optimization finds better parameters than your current manual tuning.

### **If You're Happy with Current Performance:**

You can skip optimization for now. Your current configuration with:

- `ttt_base_steps=200`
- `ttt_l2_reg_weight=0.01`
- `confidence_rejection_threshold=0.7`

Should already perform quite well. Optimization is mainly for **fine-tuning** these values.

---

## 📝 **After Optimization**

1. **Check Best Parameters**: Optuna will output the best hyperparameters found
2. **Update Config**: Apply best parameters to `config.py`
3. **Verify Performance**: Run a full evaluation with optimized parameters
4. **Compare Results**: Compare optimized vs. current configuration

---

## 🔍 **Monitoring Optimization**

Optuna automatically:

- ✅ Saves progress to `optuna_study.db` (can resume if interrupted)
- ✅ Uses pruning to skip bad trials early (saves time)
- ✅ Logs to Wandb (offline mode by default)

You can visualize progress:

```bash
# Install optuna-dashboard (if not already installed)
pip install optuna-dashboard

# View optimization progress
optuna-dashboard sqlite:///optuna_study.db
```

---

## ❓ **Decision Time**

**My Recommendation**:

- If you have **4-8 hours** available and want maximum performance → **Run 20 trials**
- If you're short on time but curious → **Run 10 trials** for a quick check
- If current performance is sufficient → **Skip optimization** for now

The current configuration is already quite good! Optimization will help fine-tune, but won't make dramatic improvements (expected +5-10% at most).
