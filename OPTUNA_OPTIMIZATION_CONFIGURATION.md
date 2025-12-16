# Optuna Optimization Configuration Summary

## Overview

The project uses **Optuna** with **TPE (Tree-structured Parzen Estimator) sampling** and **Wandb** integration for hyperparameter optimization. The optimization focuses on both meta-learning and Test-Time Training (TTT) parameters for zero-day attack detection.

## Quick Start

```bash
# Basic usage (default: 20 trials, balanced_base_ttt metric)
python optimize_hyperparameters.py

# Custom number of trials
python optimize_hyperparameters.py --n_trials 50

# Optimize different metric
python optimize_hyperparameters.py --metric multi_objective --n_trials 30

# Custom study name
python optimize_hyperparameters.py --study_name "kdd_optimization" --n_trials 40
```

## Command-Line Arguments

| Argument       | Default                           | Description                                |
| -------------- | --------------------------------- | ------------------------------------------ |
| `--n_trials`   | 20                                | Number of optimization trials              |
| `--study_name` | "zero_day_detection_optimization" | Name for Optuna study                      |
| `--metric`     | "balanced_base_ttt"               | Primary metric to optimize (see below)     |
| `--direction`  | "maximize"                        | Optimization direction (maximize/minimize) |

## Optimization Metrics

### 1. `balanced_base_ttt` (DEFAULT - Recommended)

**Comprehensive objective with penalties for deployment considerations**

- **Base Score**: 40% base F1 + 30% TTT ZDR + 30% TTT F1
- **FAR Penalty**: Exponential penalty for high false alarm rate
  - Formula: `0.1 * min(1.0, (FAR / 0.05)²)`
  - Heavily penalizes FAR > 5%
- **Drift Penalty**: Penalizes drop in known attack accuracy
  - Formula: `0.05 * max(0, base_non_zero_day_acc - ttt_non_zero_day_acc)`
  - Prevents TTT from degrading known attack detection
- **Final Score**: `Base Score - FAR Penalty - Drift Penalty`

**Use Case**: Best for production deployment where both zero-day detection and low false alarms are critical.

### 2. `multi_objective`

**Balanced multi-objective optimization (TTT-only metrics)**

- **Zero-day ZDR**: 30% weight
- **Non-zero-day F1**: 35% weight
- **Overall F1**: 35% weight

**Use Case**: IDS that needs to detect BOTH known and unknown attacks equally well.

### 3. `ttt_zero_day_detection_rate`

**Optimize for zero-day detection only (TTT-adapted)**

- Directly optimizes TTT zero-day detection rate
- Ignores base model and other metrics

### 4. `ttt_auc_pr`

**Optimize for AUC-PR (TTT-adapted)**

- Optimizes Area Under Precision-Recall Curve for TTT model

### 5. `ttt_f1_score`

**Optimize for F1-score (TTT-adapted)**

- Optimizes F1-score for TTT model

### 6. `ttt_accuracy`

**Optimize for accuracy (TTT-adapted)**

- Optimizes overall accuracy for TTT model

## Optimized Hyperparameters

### Meta-Learning Parameters

| Parameter            | Search Space    | Type        | Description                             |
| -------------------- | --------------- | ----------- | --------------------------------------- |
| `meta_learning_rate` | 5e-4 to 3e-3    | Log Float   | Meta-learning learning rate             |
| `meta_epochs`        | 15 to 35        | Integer     | Number of meta-training epochs          |
| `k_shot`             | 100 to 200      | Integer     | Support set size (few-shot samples)     |
| `n_query`            | 10 to 20        | Integer     | Query set size per meta-task            |
| `num_meta_tasks`     | 30 to 120       | Integer     | Number of meta-tasks per epoch          |
| `hidden_dim`         | [128, 256, 384] | Categorical | Hidden dimension for feature extractors |
| `embedding_dim`      | [64, 128, 192]  | Categorical | Embedding dimension for prototypes      |

**Fixed Parameters** (not optimized):

- `enforce_equal_support_composition = True`
- `include_all_attack_types_in_support = True`

### TCN Configuration

| Parameter                  | Search Space  | Type        | Description                        |
| -------------------------- | ------------- | ----------- | ---------------------------------- |
| `sequence_length`          | 20 to 50      | Integer     | Length of temporal sequences       |
| `sequence_stride`          | 10 to 20      | Integer     | Stride for sequence creation       |
| `tcn_kernel_size_1`        | 2 to 6        | Integer     | TCN kernel size for layer 1        |
| `tcn_kernel_size_2`        | 2 to 6        | Integer     | TCN kernel size for layer 2        |
| `tcn_kernel_size_3`        | 2 to 6        | Integer     | TCN kernel size for layer 3        |
| `use_residual_connections` | [True, False] | Categorical | Enable residual connections in TCN |

### TTT (Test-Time Training) Parameters

| Parameter                        | Search Space            | Type            | Description                               |
| -------------------------------- | ----------------------- | --------------- | ----------------------------------------- |
| `ttt_lr`                         | 1e-4 to 2e-3            | Log Float       | TTT learning rate                         |
| `ttt_base_steps`                 | 50 to 300               | Log Float → Int | Number of TTT adaptation steps            |
| `ttt_batch_size`                 | [4, 8, 16, 32, 64, 128] | Categorical     | Batch size for TTT adaptation             |
| `ttt_adaptation_query_size`      | 1000 to 2000            | Integer         | Query set size for TTT adaptation         |
| `ttt_l2_reg_weight`              | 0.001 to 0.1            | Log Float       | L2 regularization weight                  |
| `confidence_rejection_threshold` | 0.5 to 0.9              | Float           | Confidence threshold for sample rejection |

### TENT + Pseudo-Labels Configuration

| Parameter                  | Search Space  | Type        | Description                              |
| -------------------------- | ------------- | ----------- | ---------------------------------------- |
| `use_pseudo_labels`        | [True, False] | Categorical | Enable pseudo-labeling                   |
| `pseudo_weight`            | 1.5 to 3.5    | Float       | Weight for pseudo-label loss             |
| `entropy_weight`           | 0.5 to 1.5    | Float       | Weight for entropy loss                  |
| `pseudo_threshold`         | 0.85 to 0.98  | Float       | Confidence threshold for pseudo-labels   |
| `pseudo_min_threshold`     | 0.70 to 0.85  | Float       | Minimum confidence threshold             |
| `use_teacher`              | [True, False] | Categorical | Enable EMA teacher model                 |
| `ema_decay`                | 0.95 to 0.999 | Float       | EMA decay rate for teacher model         |
| `pseudo_label_temperature` | 0.3 to 0.8    | Float       | Temperature for sharpening pseudo-labels |

### Advanced TTT Techniques

| Parameter         | Search Space  | Type        | Description                 |
| ----------------- | ------------- | ----------- | --------------------------- |
| `ttt_temperature` | 1.0 to 2.0    | Float       | Temperature scaling for TTT |
| `use_focal_loss`  | [True, False] | Categorical | Enable focal loss           |
| `focal_gamma`     | 1.5 to 3.0    | Float       | Focal loss gamma parameter  |
| `focal_alpha`     | 0.15 to 0.35  | Float       | Focal loss alpha parameter  |

## Optuna Study Configuration

```python
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",  # or "minimize"
    sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune first 5 trials
        n_warmup_steps=10      # Don't prune first 10 steps
    )
)
```

## Wandb Integration

- **Project**: `zero-day-detection-optimization`
- **Mode**: Offline (prevents hanging on network issues)
- **Logged Metrics**:
  - All hyperparameters (as `hyperparam_*`)
  - Base model metrics (accuracy, F1, AUC-PR, ZDR)
  - TTT model metrics (accuracy, F1, AUC-PR, ZDR)
  - Improvements (accuracy, F1, AUC-PR, ZDR)
  - Balanced objective components (if applicable)
  - Trial number and state

## Output Files

1. **`best_hyperparameters.json`**: Best trial hyperparameters and metrics
2. **`saved_test_sets/test_set_trial_{N}.pkl`**: Test set for each trial (for reproducibility)
3. **`saved_test_sets/test_set_best_trial.pkl`**: Test set for best trial (trial 13)

## Example Usage

```bash
# Optimize for balanced base + TTT performance (recommended)
python optimize_hyperparameters.py --n_trials 50 --metric balanced_base_ttt

# Optimize for multi-objective (zero-day + known attacks)
python optimize_hyperparameters.py --n_trials 30 --metric multi_objective

# Optimize for zero-day detection only
python optimize_hyperparameters.py --n_trials 40 --metric ttt_zero_day_detection_rate
```

## Notes

- **FP16 Mixed Precision**: Automatically enabled on GPU for 40-70% faster optimization
- **Test Set Saving**: Each trial saves its test set for reproducibility
- **Pruning**: Early stopping for poor-performing trials (after 5 startup trials)
- **Seed**: Fixed seed (42) for reproducibility
- **Offline Mode**: Wandb runs in offline mode to prevent network issues

## Current Configuration Status

The optimization is configured for:

- **Default Metric**: `balanced_base_ttt` (comprehensive with penalties)
- **Default Trials**: 20
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Pruning**: Median Pruner (starts after 5 trials, 10 warmup steps)
- **Wandb**: Offline mode enabled
