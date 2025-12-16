# Hyperparameter Optimization Guide

This guide explains how to use Optuna and Wandb for automated hyperparameter optimization of both meta-learning and TTT parameters.

## Overview

The optimization system uses:

- **Optuna**: For intelligent hyperparameter search using Tree-structured Parzen Estimator (TPE) sampling
- **Wandb**: For experiment tracking, visualization, and comparison

## Quick Start

### Basic Usage

```bash
# Activate your virtual environment
..\Tgnn_gpu\Scripts\activate

# Run optimization (default: 50 trials, optimize zero-day detection rate)
python optimize_hyperparameters.py

# Custom number of trials
python optimize_hyperparameters.py --n_trials 100

# Optimize different metric
python optimize_hyperparameters.py --metric ttt_auc_pr --n_trials 50

# Custom study name
python optimize_hyperparameters.py --study_name "backdoor_attack_optimization" --n_trials 30
```

### Command-Line Arguments

```bash
python optimize_hyperparameters.py \
    --n_trials 50 \                    # Number of optimization trials
    --study_name "my_optimization" \  # Name for Optuna study
    --metric ttt_zero_day_detection_rate \  # Metric to optimize
    --direction maximize                # maximize or minimize
```

**Available Metrics:**

- `ttt_zero_day_detection_rate` (default) - Zero-day attack detection rate
- `ttt_auc_pr` - Area Under Precision-Recall Curve
- `ttt_f1_score` - F1 Score
- `ttt_accuracy` - Overall Accuracy

## Optimized Hyperparameters

### Federated Learning Parameters

| Parameter         | Search Space      | Description                                                       |
| ----------------- | ----------------- | ----------------------------------------------------------------- |
| `num_clients`     | 3 to 10           | Number of federated clients                                       |
| `num_rounds`      | 5 to 15           | Number of federated learning rounds                               |
| `dirichlet_alpha` | 0.5 to 10.0 (log) | Dirichlet parameter for non-IID data (lower = more heterogeneous) |

### Meta-Learning Parameters

| Parameter                           | Search Space       | Description                                            |
| ----------------------------------- | ------------------ | ------------------------------------------------------ |
| `meta_learning_rate`                | 1e-4 to 1e-2 (log) | Learning rate for meta-training                        |
| `meta_epochs`                       | 2 to 5             | Number of meta-training epochs                         |
| `k_shot`                            | 100 to 200         | Support samples per class                              |
| `n_query`                           | 10 to 20           | Query samples per task                                 |
| `hidden_dim`                        | [256, 512, 768]    | Hidden dimension size                                  |
| `embedding_dim`                     | [128, 256, 512]    | Embedding dimension size                               |
| `enforce_equal_support_composition` | [True, False]      | Ensure balanced Normal/Attack in support set (n_way=2) |

### TCN Configuration Parameters

| Parameter           | Search Space | Description                                          |
| ------------------- | ------------ | ---------------------------------------------------- |
| `sequence_length`   | 20 to 50     | Length of temporal sequences for TCN                 |
| `sequence_stride`   | 10 to 20     | Stride for sequence creation                         |
| `tcn_kernel_size_1` | 2 to 6       | Kernel size for TCN branch 1 (fine-scale patterns)   |
| `tcn_kernel_size_2` | 2 to 6       | Kernel size for TCN branch 2 (medium-scale patterns) |
| `tcn_kernel_size_3` | 2 to 6       | Kernel size for TCN branch 3 (coarse-scale patterns) |

### TTT Parameters

| Parameter                   | Search Space       | Description                         |
| --------------------------- | ------------------ | ----------------------------------- |
| `ttt_lr`                    | 1e-4 to 2e-3 (log) | TTT learning rate                   |
| `ttt_base_steps`            | 200 to 400         | Number of TTT adaptation steps      |
| `ttt_batch_size`            | [4, 8, 16, 32]     | Batch size for TTT                  |
| `ttt_adaptation_query_size` | 1000 to 2000       | Size of adaptation query set        |
| `ttt_temperature`           | 1.0 to 2.0         | Temperature scaling for calibration |

### TENT + Pseudo-Labels Configuration

| Parameter                  | Search Space  | Description                                     |
| -------------------------- | ------------- | ----------------------------------------------- |
| `use_pseudo_labels`        | [True, False] | Enable/disable pseudo-labeling                  |
| `pseudo_weight`            | 1.5 to 3.5    | Weight for pseudo-label loss                    |
| `entropy_weight`           | 0.5 to 1.5    | Weight for entropy loss                         |
| `pseudo_threshold`         | 0.85 to 0.98  | Initial pseudo-label confidence threshold       |
| `pseudo_min_threshold`     | 0.70 to 0.85  | Minimum pseudo-label threshold (adaptive decay) |
| `use_teacher`              | [True, False] | Use EMA teacher model for stable pseudo-labels  |
| `ema_decay`                | 0.95 to 0.999 | EMA decay rate for teacher model                |
| `pseudo_label_temperature` | 0.3 to 0.8    | Temperature for sharpening pseudo-label logits  |

### Advanced TTT Techniques

| Parameter        | Search Space  | Description                                             |
| ---------------- | ------------- | ------------------------------------------------------- |
| `use_focal_loss` | [True, False] | Enable/disable focal loss for class imbalance           |
| `focal_gamma`    | 1.5 to 3.0    | Focal loss gamma (higher = more focus on hard examples) |
| `focal_alpha`    | 0.15 to 0.35  | Focal loss alpha (class balancing)                      |

### Federated Aggregation

| Parameter    | Search Space       | Description                  |
| ------------ | ------------------ | ---------------------------- |
| `fedprox_mu` | 0.001 to 0.1 (log) | FedProx proximal term weight |

## Wandb Integration

### Setup Wandb

1. **Login to Wandb** (first time only):

   ```bash
   wandb login
   ```

   Enter your API key when prompted.

2. **View Results**:
   - Open https://wandb.ai
   - Navigate to project: `zero-day-detection-optimization`
   - View real-time metrics, hyperparameter importance, and parallel coordinates plots

### Wandb Logged Metrics

**Base Model Metrics:**

- `base_accuracy`
- `base_f1_score`
- `base_auc_pr`
- `base_zero_day_detection_rate`

**TTT Model Metrics:**

- `ttt_accuracy`
- `ttt_f1_score`
- `ttt_auc_pr`
- `ttt_zero_day_detection_rate`

**Improvements:**

- `accuracy_improvement`
- `f1_improvement`
- `auc_pr_improvement`
- `zero_day_detection_improvement`

## Output Files

After optimization completes:

1. **`best_hyperparameters.json`**: Contains the best hyperparameters found

   ```json
   {
     "best_trial_number": 42,
     "best_value": 0.9234,
     "best_params": {
       "meta_learning_rate": 0.0012,
       "ttt_lr": 0.0007,
       ...
     },
     "best_user_attrs": {
       "base_accuracy": 0.8123,
       "ttt_accuracy": 0.8765,
       ...
     }
   }
   ```

2. **Optuna Study Database**: Stored in `optuna.db` (SQLite)
   - Can be loaded later for analysis
   - Use `optuna-dashboard` to visualize

## Using Best Hyperparameters

After optimization, update your `config.py` with the best hyperparameters:

```python
# Load best hyperparameters
import json
with open('best_hyperparameters.json', 'r') as f:
    best = json.load(f)

# Update config.py with best_params
config.learning_rate = best['best_params']['meta_learning_rate']
config.ttt_lr = best['best_params']['ttt_lr']
# ... etc
```

## Advanced Usage

### Resume Optimization

Optuna automatically saves progress. If interrupted, you can resume:

```python
import optuna

# Load existing study
study = optuna.load_study(
    study_name="zero_day_detection_optimization",
    storage="sqlite:///optuna.db"
)

# Continue optimization
study.optimize(objective, n_trials=50)
```

### Pruning (Early Stopping)

The optimizer uses `MedianPruner` to stop unpromising trials early:

- `n_startup_trials=5`: Run at least 5 trials before pruning
- `n_warmup_steps=10`: Wait 10 steps before pruning

### Multi-Objective Optimization

To optimize multiple metrics simultaneously:

```python
# Modify optimize_hyperparameters.py to use MultiObjectiveSampler
study = optuna.create_study(
    directions=["maximize", "maximize"],  # [ZDR, AUC-PR]
    sampler=optuna.samplers.NSGAIISampler()
)
```

## Tips for Better Results

1. **Start with fewer trials** (20-30) to test the setup
2. **Focus on one metric** at a time (e.g., zero-day detection rate)
3. **Monitor Wandb** during optimization to catch issues early
4. **Use pruning** to save time on unpromising trials
5. **Increase trials** (100+) for production optimization
6. **Fix random seeds** for reproducibility (already done in code)

## Troubleshooting

### Wandb Login Issues

```bash
# Re-login
wandb login --relogin
```

### Out of Memory

- Reduce `ttt_batch_size` search space
- Reduce `ttt_adaptation_query_size` range
- Use smaller model dimensions

### Slow Optimization

- Reduce `n_trials`
- Enable pruning (already enabled)
- Use fewer federated rounds during optimization

## Example Workflow

```bash
# 1. Quick test (10 trials)
python optimize_hyperparameters.py --n_trials 10 --study_name "quick_test"

# 2. Full optimization (100 trials)
python optimize_hyperparameters.py --n_trials 100 --study_name "production_optimization"

# 3. Check results
cat best_hyperparameters.json

# 4. Apply best hyperparameters to config.py
# (manually update config.py with best_params)

# 5. Run final evaluation
python main.py
```

## Next Steps

1. Run optimization with your desired metric
2. Review results in Wandb dashboard
3. Apply best hyperparameters to `config.py`
4. Run final evaluation with optimized parameters
5. Compare results with baseline
