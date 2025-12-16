# Optimization Trial Configuration

## ✅ **Updated Configuration for New Optimization Trial**

The optimization script has been updated with improved parameter ranges based on recent findings.

---

## 📊 **Key Updates**

### **1. `meta_epochs` Range Updated**
- **Previous**: 3-30
- **New**: 20-35
- **Reason**: Based on analysis showing optimal convergence at 20-30 epochs without overfitting

### **2. `transductive_steps` Now Included**
- **Range**: 10-20
- **Purpose**: Optimize prototype refinement steps per task
- **Impact**: Small but can improve performance by 0.5-1%

---

## 🚀 **Run Optimization**

### **Basic Command (20 trials, default settings):**
```bash
python optimize_hyperparameters_cicids.py
```

### **Custom Configuration:**
```bash
# Run with 15 trials, balanced_base_ttt metric (default)
python optimize_hyperparameters_cicids.py --n_trials 15

# Run with 10 trials for faster optimization
python optimize_hyperparameters_cicids.py --n_trials 10

# Run with different zero-day attack
python optimize_hyperparameters_cicids.py --n_trials 15 --zero_day_attack "DDoS"

# Run with different metric
python optimize_hyperparameters_cicids.py --n_trials 15 --metric "ttt_zero_day_detection_rate"
```

---

## 📋 **Optimization Parameters**

### **Federated Learning:**
- `num_clients`: 3-5
- `num_rounds`: 5-20
- `dirichlet_alpha`: 2.0-5.0

### **Meta-Learning:**
- `meta_epochs`: **20-35** (updated range)
- `learning_rate`: 1e-4 to 1e-2 (log scale)
- `k_shot`: 30-100
- `n_query`: 10-20
- `num_meta_tasks`: 20-50
- `hidden_dim`: [256, 512, 768]
- `embedding_dim`: [128, 256, 512]

### **TCN:**
- `sequence_length`: 20-50
- `sequence_stride`: 10-20
- `tcn_kernel_size_1/2/3`: 2-6 each
- `use_residual_connections`: [True, False]

### **TTT (Test-Time Training):**
- `ttt_lr`: 1e-4 to 2e-3 (log scale)
- `ttt_base_steps`: 200-400
- `ttt_batch_size`: [4, 8, 16, 32]
- `ttt_adaptation_query_size`: 1000-2000

### **Transductive Optimization:**
- `transductive_steps`: **10-20** (newly added)
- Controls prototype refinement per task

### **TENT + Pseudo-Labels:**
- `use_pseudo_labels`: [True, False]
- `pseudo_weight`: 1.5-3.5
- `entropy_weight`: 0.5-1.5
- `pseudo_threshold`: 0.85-0.98
- `pseudo_min_threshold`: 0.70-0.85
- `use_teacher`: [True, False] (EMA teacher model)
- `ema_decay`: 0.95-0.999
- `pseudo_label_temperature`: 0.3-0.8

### **Advanced TTT:**
- `ttt_temperature`: 1.0-2.0
- `use_focal_loss`: [True, False]
- `focal_gamma`: 1.5-3.0
- `focal_alpha`: 0.15-0.35

### **FedProx:**
- `fedprox_mu`: 0.001-0.1 (log scale)

---

## 🎯 **Optimization Metrics**

### **Default: `balanced_base_ttt`** (Recommended)
- **Formula**: `0.40 * base_f1_score + 0.30 * ttt_zero_day_detection_rate + 0.30 * ttt_f1_score`
- **Purpose**: Optimizes for BOTH strong base model AND excellent TTT zero-day detection
- **Use Case**: Best for comprehensive evaluation

### **Alternative Metrics:**
- `ttt_zero_day_detection_rate`: Zero-day detection only (TTT-adapted)
- `ttt_auc_pr`: AUC-PR score (TTT-adapted)
- `ttt_f1_score`: F1-score (TTT-adapted)
- `ttt_accuracy`: Accuracy (TTT-adapted)
- `multi_objective`: Balanced TTT-only metrics (30% ZDR, 35% non-zero-day F1, 35% overall F1)

---

## ⚙️ **Current Configuration Values**

**From Recent Run:**
- `meta_epochs`: 100 (excessive - will be optimized)
- `transductive_steps`: 20 (will be optimized to 10-20)
- `use_teacher`: True (will be optimized True/False)
- `num_meta_tasks`: 50 (will be optimized 20-50)

---

## 📈 **Expected Results**

After optimization, you should see:
- **Optimal `meta_epochs`**: Around 25-30 (instead of 100)
- **Optimal `transductive_steps`**: Around 12-15 (instead of 20)
- **Best `use_teacher`**: True (if teacher model helps)
- **Best `num_meta_tasks`**: Around 35-45 (balance of diversity and time)

---

## 🔍 **Output Files**

After optimization completes:
1. **`best_hyperparameters_cicids.json`**: Best hyperparameters found
2. **Wandb logs**: Optimization history (offline mode)
3. **Console output**: Summary of best trial

---

## ⏱️ **Estimated Time**

- **Per trial**: ~5-15 minutes (depending on config)
- **10 trials**: ~1-2.5 hours
- **15 trials**: ~1.5-4 hours
- **20 trials**: ~2-5 hours

With FP16 enabled and current optimizations, trials should be faster.

---

## ✅ **Ready to Run**

The optimization script is ready with:
- ✅ Updated `meta_epochs` range (20-35)
- ✅ Added `transductive_steps` optimization (10-20)
- ✅ All existing parameters included
- ✅ Teacher model optimization enabled
- ✅ Balanced metric by default

**Run command:**
```bash
python optimize_hyperparameters_cicids.py --n_trials 15
```









