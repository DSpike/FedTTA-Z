"""
KDD Dataset Optimized Configuration Backup
==========================================

This file contains the KDD-optimized hyperparameters that were used before switching to UNSW.
Use this as a reference to restore KDD settings or compare with UNSW settings.

⚠️ IMPORTANT: These values are from the ACTUAL config.py (git commit c619901), NOT from best_hyperparameters.json.
The Optuna file (best_hyperparameters.json) has DIFFERENT values (Trial 7), suggesting either:
- Multiple optimization runs were performed
- Config was manually adjusted after optimization
- Different optimization objectives were used

Last Updated: Before UNSW switch (git commit c619901)
Dataset: KDDTest+ / NSL-KDD
Source: Actual config.py values (not Optuna file)
"""

# === KDD-OPTIMIZED HYPERPARAMETERS ===

# Model Architecture
KDD_INPUT_DIM = 41  # KDDTest+ has 41 features (42 columns - 1 label column)
KDD_HIDDEN_DIM = 128  # From actual config.py (git c619901) - NOTE: Optuna file has 256!
KDD_EMBEDDING_DIM = 256  # From actual config.py (git c619901) - NOTE: Optuna file has 128!

# TCN Configuration
KDD_SEQUENCE_LENGTH = 22  # From actual config.py (git c619901) - NOTE: Optuna file has 37!
KDD_SEQUENCE_STRIDE = 12  # From actual config.py (git c619901) - NOTE: Optuna file has 16!
KDD_TCN_KERNEL_SIZES = (2, 3, 3)  # From actual config.py (git c619901) - NOTE: Optuna file has (4, 3, 3)!
KDD_USE_RESIDUAL_CONNECTIONS = False  # From actual config.py (git c619901) - NOTE: Optuna file has True!

# Meta-Learning Configuration
KDD_META_EPOCHS = 21  # From actual config.py (git c619901) - NOTE: Optuna file has 23!
KDD_K_SHOT = 152  # From actual config.py (git c619901) - NOTE: Optuna file has 117!
KDD_N_QUERY = 16  # From actual config.py (git c619901) - NOTE: Optuna file has 11!
KDD_LEARNING_RATE = 0.0016387494099028342  # From actual config.py (git c619901) - NOTE: Optuna file has 0.00174!

# TTT Configuration
KDD_TTT_BASE_STEPS = 70  # Current value (may have been adjusted)
KDD_TTT_ADAPTATION_QUERY_SIZE = 1198  # Optimized from Optuna Trial 1
KDD_TTT_BATCH_SIZE = 64  # Optimized from Optuna Trial 1
KDD_TTT_LR = 0.002  # Optimized from Optuna Trial 1
KDD_TTT_L2_REG_WEIGHT = 0.01  # Current value
KDD_CONFIDENCE_REJECTION_THRESHOLD = 0.90  # Optimized for KDD (strict)

# Dataset-Specific Settings
KDD_DATA_PATH = "KDDTrain+.csv"  # KDD dataset path (training)
KDD_TEST_PATH = "KDDTest+.csv"  # KDD test path (testing)
KDD_ZERO_DAY_ATTACK = "DoS"  # Category-based zero-day
KDD_USE_CATEGORY_GROUPING = True  # KDD uses category grouping
KDD_ATTACK_TYPES = {
    'Normal': 0,
    'DoS': 1,
    'Probe': 2,
    'R2L': 3,
    'U2R': 4
}

# === RESTORE FUNCTION ===
def get_kdd_config():
    """
    Returns a dictionary with KDD-optimized hyperparameters.
    Use this to restore KDD settings in config.py
    """
    return {
        'input_dim': KDD_INPUT_DIM,
        'hidden_dim': KDD_HIDDEN_DIM,
        'embedding_dim': KDD_EMBEDDING_DIM,
        'sequence_length': KDD_SEQUENCE_LENGTH,
        'sequence_stride': KDD_SEQUENCE_STRIDE,
        'tcn_kernel_sizes': KDD_TCN_KERNEL_SIZES,
        'use_residual_connections': KDD_USE_RESIDUAL_CONNECTIONS,
        'meta_epochs': KDD_META_EPOCHS,
        'k_shot': KDD_K_SHOT,
        'n_query': KDD_N_QUERY,
        'learning_rate': KDD_LEARNING_RATE,
        'ttt_base_steps': KDD_TTT_BASE_STEPS,
        'ttt_adaptation_query_size': KDD_TTT_ADAPTATION_QUERY_SIZE,
        'ttt_batch_size': KDD_TTT_BATCH_SIZE,
        'ttt_lr': KDD_TTT_LR,
        'ttt_l2_reg_weight': KDD_TTT_L2_REG_WEIGHT,
        'confidence_rejection_threshold': KDD_CONFIDENCE_REJECTION_THRESHOLD,
        'data_path': KDD_DATA_PATH,
        'test_path': KDD_TEST_PATH,
        'zero_day_attack': KDD_ZERO_DAY_ATTACK,
        'use_category_grouping': KDD_USE_CATEGORY_GROUPING,
    }

