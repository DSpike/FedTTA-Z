"""
Centralized Configuration System for Blockchain Federated Learning
This module provides a single source of truth for all system configuration.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import torch


@dataclass
class SystemConfig:
    """Centralized system configuration - single source of truth"""
    
    # === FEDERATED LEARNING CONFIGURATION ===
    num_clients: int = 10  # Optimized: 10 clients (from Multi-Objective Optuna Trial 6)
    num_rounds: int = 13  # Optimized: 13 rounds (from Multi-Objective Optuna Trial 6)
    local_epochs: int = 10  # Balanced epochs per round for better federated learning
    learning_rate: float = 0.002540786469371161  # Optimized meta-learning rate (from Multi-Objective Optuna Trial 6)
    batch_size: int = 16  # Optimal batch size (validated in hyperparameter tuning)
    dirichlet_alpha: float = 4.034673453372198  # Optimized: 4.035 (from Multi-Objective Optuna Trial 6)
                                   # CHANGED from 10.0 to 1.0: With α=10 (near IID), results are identical for 2 vs 10 clients
                                   # α = 0.5: Very high heterogeneity (extreme non-IID) - can create clients with <1% of data
                                   # α = 1.0: Moderate heterogeneity (balanced non-IID) - RECOMMENDED for 10 clients ✅
                                   # α = 10.0: Low heterogeneity (near IID) - makes all clients see same data → same results
    
    # === FEDPROX CONFIGURATION ===
    use_fedprox: bool = True  # Enable FedProx aggregation algorithm (adds proximal term for better non-IID handling)
    fedprox_mu: float = 0.004424996646298212  # Optimized: 0.0044 (from Multi-Objective Optuna Trial 6)
   
    # === QUICK VERIFICATION MODE ===
    quick_verify: bool = False  # When True, run a fast built-in self-check path
    
    # === DATA CONFIGURATION ===
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    zero_day_attack: str = "Exploits"  # UNSW-NB15 attack type (label 5)
    
    # Attack type mapping (UNSW-NB15 dataset)
    attack_types = {
        'Normal': 0,
        'Fuzzers': 1,
        'Analysis': 2,
        'Backdoor': 3,
        'DoS': 4,
        'Exploits': 5,
        'Generic': 6,
        'Reconnaissance': 7,
        'Shellcode': 8,
        'Worms': 9
    }
    
    '''
    # CICIDS dataset attack types (commented out - use UNSW-NB15 instead)
    attack_types = {
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 2,
        'DoS GoldenEye': 3,
        'DoS Hulk': 4,
        'DoS Slowhttptest': 5,
        'DoS slowloris': 6,
        'FTP-Patator': 7,
        'Heartbleed': 8,
        'Infiltration': 9,
        'PortScan': 10,
        'SSH-Patator': 11,
        'Web Attack': 12,
        'Web Attack  Brute Force': 12,
        'Web Attack  Sql Injection': 12,
        'Web Attack  XSS': 12
    }
    '''
    @property
    def zero_day_attack_label(self) -> int:
        """Get the integer label for the zero-day attack type"""
        return self.attack_types.get(self.zero_day_attack, 6)  # Default to Generic=6 (UNSW-NB15)
    
    # === MODEL CONFIGURATION ===
    input_dim: int = 43  # Updated after IGRF-RFE feature selection (43 features selected)
    hidden_dim: int = 512  # Optimized: 512 (from Multi-Objective Optuna Trial 6)
    embedding_dim: int = 256  # Optimized: 256 (from Multi-Objective Optuna Trial 6)
    num_classes: int = 2  # Binary classification (Normal vs Attack) for zero-day detection
    
    # === FEATURE SELECTION CONFIGURATION ===
    use_igrf_rfe: bool = False  # TEMPORARILY DISABLED: Enable IGRF-RFE hybrid feature selection
    feature_selection_ratio: float = 0.8  # Select top 80% of features
    
    # === TCN CONFIGURATION ===
    use_tcn: bool = True  # Enable/disable TCN feature extraction
    disable_tcn_feature_extraction: bool = False  # If True, replace TCN with simple pooling (for testing)
    sequence_length: int = 47  # Optimized: 47 (from Multi-Objective Optuna Trial 6)
    sequence_stride: int = 13  # Optimized: 13 (from Multi-Objective Optuna Trial 6)
    tcn_kernel_sizes: tuple = (3, 2, 4)  # Optimized kernel sizes for TCN branches (from Multi-Objective Optuna Trial 6)
    meta_epochs: int = 3  # Optimized: 3 (from Multi-Objective Optuna Trial 6)
    transductive_steps: int = 20
    transductive_lr: float = 0.0007
    
    # === TEST-TIME TRAINING (TTT) CONFIGURATION ===
    ttt_base_steps: int = 293  # Optimized: 293 steps (from Multi-Objective Optuna Trial 6)
    ttt_max_steps: int = 500  # Maximum TTT steps (safety limit)
    ttt_adaptation_query_size: int = 1037  # Optimized: 1037 (from Multi-Objective Optuna Trial 6)
    ttt_batch_size: int = 16  # Optimized: 16 (from Multi-Objective Optuna Trial 6)
    ttt_lr: float = 0.00011136839897653453  # Optimized: 0.000111 (from Multi-Objective Optuna Trial 6)
    
    # === ATTACK PROTOTYPE DISCOVERY TTT ===
    use_attack_prototype_ttt: bool = False  # Enable attack prototype discovery TTT (+5-8% improvement)
    ttt_prototype_clusters: int = 10  # Number of attack prototypes to discover
    ttt_prototype_weight: float = 1  # Weight for prototype alignment loss
    ttt_prototype_entropy_weight: float = 0.3  # Weight for entropy loss (complement to prototype)
    ttt_prototype_steps: int = 100  # Number of adaptation steps for prototype TTT
    ttt_lr_min: float = 4e-5  # Minimum learning rate for cosine schedule
    #ttt_gaussian_noise_std: float = 0.05  # Reduced to 5% for milder test-time augmentation
    ttt_lr_decay: float = 0.8  # (Unused now for TTT; kept for backward compatibility)
    ttt_warmup_steps: int = 20  # Learning rate warmup steps
    ttt_weight_decay: float = 1e-4  # Increased weight decay to regularize aggressive updates
    ttt_patience: int = 30  # Early stopping patience (increased for better convergence)
    ttt_timeout: int = 45  # TTT timeout in seconds (increased)
    ttt_improvement_threshold: float = 1e-5  # Minimum improvement threshold (more sensitive)
    # REMOVED: ttt_threshold, ttt_min_threshold - replaced with adaptive threshold (pseudo_threshold → pseudo_min_threshold)
    ttt_entropy_weight: float = 0.5    # Weight for entropy loss in TTT (test-time only)
    ttt_consistency_weight: float = 0.10  # Weight for prototype consistency in unsupervised TTT
    ttt_mixup_alpha: float = 0.20       # DISABLED: MixUp inappropriate for TTT with unlabeled data
    # Temperature scaling for probability calibration after TTT (improves AUC-PR calibration)
    ttt_temperature: float = 1.5406351216101064  # Optimized: 1.541 (from Multi-Objective Optuna Trial 6)
    use_semisupervised_ttt: bool = False  # When True, use new semi-supervised TTT instead of legacy TENT+pseudo

    # === TENT + PSEUDO-LABELS CONFIGURATION ===
    # Optimized hyperparameters from Multi-Objective Optuna (100% ZDR + Balanced Performance)
    # FIXED: Lowered thresholds to increase pseudo-label ratio from 1.7% to 25-40% for better loss decrease
    use_pseudo_labels: bool = True  # Optimized: True (from Multi-Objective Optuna Trial 6)
    pseudo_threshold: float = 0.75  # FIXED: Lowered from 0.950 (allows ~25% of samples vs 1.7%)
    pseudo_min_threshold: float = 0.60  # FIXED: Lowered from 0.732 (allows ~40% of samples vs 1.7%)
    pseudo_weight: float = 1.7541210253037696  # Optimized: 1.754 (from Multi-Objective Optuna Trial 6)
    entropy_weight: float = 1.0222432600548044  # Optimized: 1.022 (from Multi-Objective Optuna Trial 6)
    use_teacher: bool = True  # Optimized: True (from Multi-Objective Optuna Trial 6)
    ema_decay: float = 0.9525324043372617  # Optimized: 0.953 (from Multi-Objective Optuna Trial 6)
    
    # === ADVANCED TTT TECHNIQUES FOR SOTA PERFORMANCE ===
    use_focal_loss: bool = False  # Optimized: False (from Multi-Objective Optuna Trial 6)
    focal_gamma: float = 2.963778119193802  # Optimized: 2.964 (from Multi-Objective Optuna Trial 6)
    focal_alpha: float = 0.25326006966023906  # Optimized: 0.253 (from Multi-Objective Optuna Trial 6)
    use_mixup_ttt: bool = False  # DISABLED: Mixup inappropriate for TTT - requires labels, mixes noisy pseudo-labels, destroys network flow semantics
    mixup_alpha: float = 0.2  # Not used when use_mixup_ttt=False
    use_label_smoothing: bool = False  # DISABLED: Conflicts with focal loss (focal focuses on hard examples, smoothing softens all labels equally)
    label_smoothing: float = 0.1  # Not used when use_label_smoothing=False
    
    # === CLASS WEIGHT AND MAGIC NUMBERS ===
    pseudo_label_temperature: float = 0.565677315784074  # Optimized: 0.566 (from Multi-Objective Optuna Trial 6)
    transductive_patience: int = 8  # Early stopping patience for transductive optimization (number of steps without improvement)
    missing_class_weight_multiplier: float = 2.0  # Weight multiplier for missing classes in class weight calculation (encourages learning rare classes)
    class_weight_normalization_multiplier: float = 2.0  # Multiplier for class weight normalization (keeps weights strong after normalization)
    use_multi_scale_tta: bool = False  # DISABLED: Multi-scale TTA not applicable to network traffic data
    # Scaling features (0.9x, 1.0x, 1.1x) destroys semantic meaning for network features
    # Packet counts and network metrics don't have "scale" semantics like images do
    tta_scales: list = field(default_factory=lambda: [0.9, 1.0, 1.1])  # Unused when use_multi_scale_tta=False
    use_self_ensemble: bool = False  # DISABLED: Self-ensemble (causing errors, will re-enable after fixing)
    ensemble_checkpoints: int = 3  # Number of checkpoints to ensemble
    use_lr_warmup: bool = True  # Learning rate warmup for better initial adaptation
    warmup_steps: int = 20  # Warmup steps (already in config, but now used)
    # REMOVED: All complex threshold configs (ttt_normal_anchor_threshold, ttt_attack_conf_threshold, 
    # ttt_ambiguous_high, ttt_ambiguous_low) - replaced with single adaptive threshold
    ttt_pseudo_loss_weight: float = 1.0  # Weight for confident pseudo-label loss (not used - see pseudo_weight)
    ttt_attack_prior: float = 0.30  # Expected max attack ratio in query batches
    # Early stopping configuration
    ttt_early_stopping: bool = True  # Enable early stopping to prevent overfitting
    ttt_early_stopping_patience: int = 15  # INCREASED from 10: Allow more training before stopping (+0.5-1% accuracy)
    ttt_early_stopping_min_delta: float = 1e-4  # Minimum change to qualify as improvement
    # Pseudo-label validation configuration
    ttt_pseudo_label_validation: bool = False  # DISABLED: Redundant validation adds 3x overhead
    # Confidence-based filtering is sufficient; adding noise then checking consistency is methodologically questionable
    ttt_validation_forward_passes: int = 3  # Unused when ttt_pseudo_label_validation=False
    ttt_validation_noise_std: float = 0.05  # Unused when ttt_pseudo_label_validation=False
    # Phase 1: 95% Performance Techniques
    ttt_zero_day_focused: bool = False  # DISABLED: Removed arbitrary zero-day categorization logic
    # Low confidence ≠ zero-day. Model adapts naturally using confidence-based pseudo-labeling
    # REMOVED: ttt_zero_day_ratio, ttt_zero_day_candidate_threshold - not used in simplified approach
    ttt_bn_statistics_adaptation: bool = False  # Disabled: TENT already adapts BN via gradient descent - manual updates conflict
    ttt_bn_ema_decay: float = 0.9  # EMA decay for BN statistics update
    # REMOVED: ttt_contrastive_weight, ttt_prototype_weight - simplified loss function only uses entropy + pseudo-label
    ttt_adaptive_zdr_threshold: bool = False  # Enable adaptive threshold optimization for zero-day
    ttt_zdr_target: float = 0.85  # INCREASED from 0.80: More aggressive ZDR target (+15-25% ZDR)
    ttt_zdr_max_far: float = 0.50  # INCREASED from 0.40: Allow more false alarms for better ZDR (+10-15% ZDR)
    
    # Consistency loss configuration (ENABLED for better generalization and >96% accuracy)
    consistency_weight: float = 0.3  # ENABLED: Adds robustness through augmentation consistency
    jitter_sigma: float = 0.10  # REDUCED from 0.15 - smaller jitter for stable adaptation
    scale_sigma: float = 0.15  # REDUCED from 0.2 - smaller scale for stable adaptation
    diversity_weight: float = 0.0  # DISABLED: Using recommended approach (Entropy + Pseudo-label only)
    
    # === ENSEMBLE TTT CONFIGURATION ===
    use_ensemble_ttt: bool = True  # ENABLED: Ensemble TTT with 3 variants (pseudo-label, contrastive, self-supervised)
    # Ensemble TTT runs 3 TTT variants in parallel and combines predictions using uncertainty-weighted voting
    # Expected improvement: +3-5% accuracy over single TTT variant
    use_best_individual_if_ensemble_fails: bool = True  # FIX 5: Use best individual model if ensemble underperforms
    # When enabled, if ensemble F1 < best individual F1 by >0.01, return best individual model instead
    
    # === THRESHOLD OPTIMIZATION STRATEGY ===
    # IMPORTANT: For reproducible research, choose ONE consistent threshold optimization strategy
    # Options: 'pr_optimized' (F1-optimized using PR curve) or 'zdr_optimized' (ZDR-optimized for zero-day detection)
    # Using a single consistent strategy avoids cherry-picking concerns in research papers
    threshold_optimization_strategy: str = 'pr_optimized'  # 'pr_optimized' or 'zdr_optimized'
    # 'pr_optimized': Optimize threshold for F1-score using precision-recall curve (balanced approach)
    # 'zdr_optimized': Optimize threshold specifically for Zero-Day Detection Rate (recall-focused)
    
    # Adaptive threshold settings
    use_adaptive_threshold: bool = True  # Use data-adaptive thresholds
    threshold_adaptation_mode: str = 'combined'  # 'scheduled', 'adaptive', or 'combined'
    # Max FAR allowed when choosing a ZDR-focused threshold from ROC curve
    # Increase this to push for higher recall/ZDR at the cost of more false alarms
    max_far_for_zdr: float = 0.35  # INCREASED from 0.25 - more permissive for better ZDR/recall
    
    # === TRAINING CONFIGURATION ===
    support_weight: float = 0.5
    test_weight: float = 0.5
    validation_patience_limit: int = 10
    recent_rounds: int = 10
    
    # === EVALUATION CONFIGURATION ===
    support_size: int = 20
    num_meta_tasks: int = 35  # Optimized: 35 (from Multi-Objective Optuna Trial 6)
    
    # === FEW-SHOT LEARNING CONFIGURATION ===
    n_way: int = 2  # Number of classes per task
    k_shot: int = 129  # Optimized: 129 (from Multi-Objective Optuna Trial 6)
    n_query: int = 18  # Optimized: 18 (from Multi-Objective Optuna Trial 6)
    enforce_equal_support_composition: bool = False  # Optimized: False (from Multi-Objective Optuna Trial 6)
    include_all_attack_types_in_support: bool = True  # If True, sample attack support set from ALL attack types (uniformly distributed) instead of one random type
    
    # === VALIDATION CONFIGURATION ===
    max_val_samples: int = 1000  # Limit validation samples for memory efficiency
    overfitting_threshold: float = 0.15  # Gap threshold between training and validation accuracy
    max_overfitting_rounds: int = 5  # Stop if overfitting detected for N consecutive rounds
    recent_rounds: int = 10  # Number of recent rounds to consider for analysis
    
    # === PERFORMANCE THRESHOLDS ===
    default_threshold: float = 0.5  # Default classification threshold
    participation_excellent: float = 0.95  # Excellent participation threshold
    participation_good: float = 0.90  # Good participation threshold
    participation_fair: float = 0.80  # Fair participation threshold
    participation_poor: float = 0.70  # Poor participation threshold
    recent_participation_bonus: float = 5.0  # Bonus points for perfect recent participation
    retry_delay: float = 1.0  # Retry delay in seconds
    max_retries: int = 3  # Maximum number of retries
    
    # === DEVICE CONFIGURATION ===
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create configuration from environment variables (optional)"""
        return cls(
            num_rounds=int(os.getenv('NUM_ROUNDS', 5)),
            num_clients=int(os.getenv('NUM_CLIENTS', 5)),
            dirichlet_alpha=float(os.getenv('DIRICHLET_ALPHA', 1.0)),
            zero_day_attack=os.getenv('ZERO_DAY_ATTACK', 'Exploits'),
            use_tcn=os.getenv('USE_TCN', 'true').lower() == 'true',
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging"""
        return {
            'num_rounds': self.num_rounds,
            'num_clients': self.num_clients,
            'dirichlet_alpha': self.dirichlet_alpha,
            'zero_day_attack': self.zero_day_attack,
            'use_tcn': self.use_tcn,
            'sequence_length': self.sequence_length,
            'sequence_stride': self.sequence_stride,
            'local_epochs': self.local_epochs,
            'learning_rate': self.learning_rate,
            'ttt_base_steps': self.ttt_base_steps,
            'ttt_max_steps': self.ttt_max_steps,
            'ttt_batch_size': self.ttt_batch_size,
            'ttt_lr': self.ttt_lr,
        }
# Global configuration instance - single source of truth
config = SystemConfig()

# Convenience function to get config
def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config

# Convenience function to update config
def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

# Convenience function to reset config to defaults
def reset_config() -> None:
    """Reset configuration to defaults"""
    global config
    config = SystemConfig()
