"""
Dataset-Aware Configuration Loader
===================================

This module provides a factory pattern to load dataset-specific configurations.
It allows you to easily switch between different datasets with their optimized settings.

Usage:
    from config_loader import get_dataset_config
    config = get_dataset_config('KDD')  # or 'UNSW', 'CICIDS2017', 'CICIDS2023'
    
    # Or use command-line argument:
    python main.py --dataset KDD
"""

import argparse
import sys
from typing import Optional
from config import SystemConfig
from config_kdd_backup import get_kdd_config

# Dataset-specific configuration presets
DATASET_CONFIGS = {
    'KDD': {
        'input_dim': 41,
        'hidden_dim': 128,
        'embedding_dim': 256,
        'sequence_length': 22,
        'sequence_stride': 12,
        'tcn_kernel_sizes': (2, 3, 3),
        'meta_epochs': 21,
        'k_shot': 152,
        'n_query': 16,
        'learning_rate': 0.0016387494099028342,
        'confidence_rejection_threshold': 0.90,
        'data_path': "KDDTrain+.csv",
        'test_path': "KDDTest+.csv",
        'zero_day_attack': "DoS",
        'use_category_grouping': True,
    },
    'UNSW': {
        'input_dim': 43,
        'hidden_dim': 512,
        'embedding_dim': 256,
        'sequence_length': 21,
        'sequence_stride': 10,
        'tcn_kernel_sizes': (3, 3, 6),
        'meta_epochs': 40,
        'k_shot': 118,
        'n_query': 100,  # IMPROVED: Moderate increase from 20 → 100 (5× increase, balanced approach)
        # Previous attempts: n_query=304 degraded base model (74.86% → 63.59%) due to insufficient epochs + high LR
        # New strategy: Conservative increase with TENT approach (only BN params) for better generalization
        'learning_rate': 0.0009,  # Slightly reduced from 0.001096 for larger episodes with n_query=100
        'confidence_rejection_threshold': 0.70,
        'data_path': "UNSW_NB15_training-set.csv",
        'test_path': "UNSW_NB15_testing-set.csv",
        'zero_day_attack': "Generic",  # UNSW zero-day attack - Testing Generic attack type
        'use_category_grouping': False,
    },
    'CICIDS2017': {
        'input_dim': 78,  # CICIDS2017 has 78 features
        'hidden_dim': 512,  # Optimized from Optuna (5 trials)
        'embedding_dim': 128,  # Optimized from Optuna
        'sequence_length': 25,  # Optimized from Optuna
        'sequence_stride': 12,  # Optimized from Optuna (was 15)
        'tcn_kernel_sizes': (3, 4, 4),  # Optimized from Optuna (was 3, 5, 7)
        'meta_epochs': 22,  # Optimized from Optuna (was 20)
        'k_shot': 200,  # Optimized from Optuna (FIXED: was incorrectly 150)
        'n_query': 10,  # Optimized from Optuna (was 15)
        'learning_rate': 0.0015751320499779737,  # Optimized from Optuna (was 0.001)
        'confidence_rejection_threshold': 0.5682096494749166,  # Optimized from Optuna (was 0.75)
        'data_path': "UNSW_NB15_training-set.csv",  # Switched to UNSW dataset
        'test_path': "UNSW_NB15_testing-set.csv",  # Switched to UNSW dataset
        'zero_day_attack': "Backdoor",  # UNSW zero-day attack
        'use_category_grouping': False,  # UNSW uses fine-grained attack types
        # PROTOTYPE-OPTIMIZED: Loss weights for embedding quality (70% embedding, 30% classification)
        'center_loss_weight': 1.0,  # PROTOTYPE-OPTIMIZED: 0.20 * 1.0 = 20% effective weight (critical for tight clusters and low FAR)
        'contrastive_loss_weight': 1.0,  # PROTOTYPE-OPTIMIZED: 0.25 * 1.0 = 25% effective weight (inter-class separation)
        'margin_loss_weight': 1.0,  # PROTOTYPE-OPTIMIZED: 0.15 * 1.0 = 15% effective weight (minimum class separation)
        'multi_prototype_weight': 1.0,  # PROTOTYPE-OPTIMIZED: 0.10 * 1.0 = 10% effective weight (intra-class diversity)
        # =====================================================================
        # TTT Parameters - TRUE TEST-TIME TRAINING (Unsupervised Adaptation)
        # =====================================================================
        # THEORETICAL FOUNDATION (Sun et al. 2020, Wang et al. 2021):
        # TTT = Adapt feature extractor using UNSUPERVISED losses on unlabeled test data
        #
        # CORRECT PROTOCOL:
        # 1. Prototypes: FIXED from base model (validation support, NEVER updated)
        # 2. Features: Adapted via entropy minimization on test data (unlabeled)
        # 3. Pseudo-labels: Generated from FIXED prototypes (safe, no label swapping)
        # 4. Classification: adapted_features(test) + FIXED_prototypes(validation)

        'ttt_lr': 0.001,  # Conservative: 5x safer than 0.005 (gentle adaptation)
        'ttt_base_steps': 80,  # Increased from 50 to 80 for better adaptation
        'ttt_l2_reg_weight': 0.01,  # Strong: 5x more than 0.002 (stay close to base)

        'use_pseudo_labels': True,  # SAFE with FIXED prototypes (no K-means swapping)
        'pseudo_weight': 1.0,  # Balanced: equal to entropy (1:1 ratio)
        'pseudo_threshold': 0.95,  # Strict: only very confident predictions
        'pseudo_min_threshold': 0.90,  # Strict: high-confidence filtering

        'entropy_weight': 1.0,  # Primary objective: minimize prediction entropy
        'ttt_temperature': 2.0,  # Distance calibration (reasonable default)

        'ttt_batch_size': 64,  # No change (good value)
        'batch_size': 256,  # No change (optimized from Optuna)

        # =====================================================================
        # Low-Confidence-Only TTT Adaptation (NEW FEATURE)
        # =====================================================================
        # Focus TTT adaptation on LOW-CONFIDENCE samples (likely zero-day)
        # instead of adapting on ALL test samples (70% non-zero-day + 30% zero-day).
        #
        # KEY INSIGHT: Zero-day samples have HIGH uncertainty (low confidence)
        # because the model hasn't seen them before. By focusing adaptation on
        # uncertain samples, we maximize zero-day detection improvement.
        #
        # COMPARISON:
        # - All-samples TTT: Adapts on 100% of test data (70% non-zero-day dominates gradient)
        # - Low-confidence TTT: Adapts only on top 30% most uncertain samples (focuses on zero-day)
        #
        'use_low_confidence_only_ttt': True,  # Enable low-confidence-only adaptation
        'low_confidence_method': 'entropy',  # Method: 'entropy', 'probability', 'distance', 'combined'
        'low_confidence_percentile': 0.70,  # Top 30% most uncertain (0.70 quantile threshold)
        'low_confidence_min_samples': 100,  # Minimum samples for stable adaptation
        'low_confidence_max_samples': 750,  # Maximum samples (match ttt_adaptation_query_size)
        
    },
    'CICIDS2023': {
        'input_dim': 45,  # CICIoT2023 has 45 features
        'hidden_dim': 256,
        'embedding_dim': 128,
        'sequence_length': 20,
        'sequence_stride': 12,
        'tcn_kernel_sizes': (3, 4, 5),
        'meta_epochs': 19,
        'k_shot': 110,
        'n_query': 18,
        'learning_rate': 0.0012,
        'confidence_rejection_threshold': 0.72,
        'data_path': "CICIoT2023_training.csv",
        'test_path': "CICIoT2023_testing.csv",
        'zero_day_attack': "DDoS",
        'use_category_grouping': True,
    },
}


def get_dataset_config(dataset_name: Optional[str] = None) -> SystemConfig:
    """
    Get dataset-specific configuration.
    
    Args:
        dataset_name: Name of the dataset ('KDD', 'UNSW', 'CICIDS2017', 'CICIDS2023')
                     If None, tries to auto-detect from command-line args or environment.
    
    Returns:
        SystemConfig instance with dataset-specific settings
    """
    # Auto-detect from command-line arguments
    if dataset_name is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--dataset', type=str, default=None)
        args, _ = parser.parse_known_args()
        dataset_name = args.dataset
    
    # Auto-detect from environment variable
    if dataset_name is None:
        import os
        dataset_name = os.environ.get('DATASET', None)
    
    # Auto-detect from data_path in default config
    if dataset_name is None:
        default_config = SystemConfig()
        data_path = default_config.data_path.upper()
        if 'KDD' in data_path or 'NSL' in data_path:
            dataset_name = 'KDD'
        elif 'UNSW' in data_path:
            dataset_name = 'UNSW'
        elif 'CICIOT23' in data_path or 'CICIDS2023' in data_path:
            dataset_name = 'CICIDS2023'
        elif 'CICIDS2017' in data_path or 'CICIDS' in data_path:
            dataset_name = 'CICIDS2017'
    
    # Default to UNSW if still None (current default in config.py)
    if dataset_name is None:
        dataset_name = 'UNSW'
        print("⚠️  No dataset specified, defaulting to UNSW. Use --dataset KDD/UNSW/CICIDS2017/CICIDS2023")
    
    # Normalize dataset name
    dataset_name = dataset_name.upper()
    
    # Get dataset-specific config
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        )
    
    dataset_config = DATASET_CONFIGS[dataset_name]
    
    # Create base config with default values
    base_config = SystemConfig()
    
    # Override with dataset-specific values
    for key, value in dataset_config.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            print(f"⚠️  Warning: Config key '{key}' not found in SystemConfig, skipping...")
    
    print(f"✅ Loaded configuration for dataset: {dataset_name}")
    print(f"   Data path: {dataset_config['data_path']}")
    print(f"   Hidden dim: {dataset_config['hidden_dim']}, Embedding dim: {dataset_config['embedding_dim']}")
    
    return base_config


def list_available_datasets():
    """List all available dataset configurations."""
    print("Available datasets:")
    for name, config in DATASET_CONFIGS.items():
        print(f"  - {name}:")
        print(f"      Data: {config['data_path']}")
        print(f"      Input dim: {config['input_dim']}")
        print(f"      Hidden dim: {config['hidden_dim']}, Embedding dim: {config['embedding_dim']}")


if __name__ == '__main__':
    # Test the config loader
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_available_datasets()
    else:
        dataset = sys.argv[1] if len(sys.argv) > 1 else None
        config = get_dataset_config(dataset)
        print(f"\n✅ Configuration loaded for: {dataset or 'AUTO-DETECTED'}")
        print(f"   Data path: {config.data_path}")
        print(f"   Test path: {config.test_path}")
