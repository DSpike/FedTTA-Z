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
    # FIXED: Increased rounds and clients for proper training
    # CRITICAL: Only 3 rounds was insufficient for model to learn good embeddings → low base model performance (54.25%)
    # With 10 rounds, model should learn better embeddings → better prototypes → better predictions (75-85%+)
    # UPDATED: Changed from 3 rounds/3 clients to 10 rounds/5 clients based on BASE_MODEL_POOR_PERFORMANCE_DEEP_INVESTIGATION.md
    num_clients: int = 3  # Set to 3 clients for federated learning
    num_rounds: int = 3  # Set to 3 rounds for quick test
    local_epochs: int = 30  # Increased to 30 epochs per round to compensate for less data per client (3 clients = 1/3 data each)
    learning_rate: float = 0.0015751320499779737  # Optimized: 0.001575 (from CICIDS2017 Optuna Trial 0)
    batch_size: int = 16  # Optimal batch size (validated in hyperparameter tuning)
    dirichlet_alpha: float = 1.0  # Set to 1.0 for moderate heterogeneity (balanced non-IID)
                                   # α = 0.5: Very high heterogeneity (extreme non-IID) - can create clients with <1% of data
                                   # α = 1.0: Moderate heterogeneity (balanced non-IID) - RECOMMENDED for realistic federated scenario ✅
                                   # α = 4.2: Optimized value (from CICIDS2017 Optuna Trial 0)
                                   # α = 10.0: Low heterogeneity (near IID) - makes all clients see similar data distribution
    
    # === FEDPROX CONFIGURATION ===
    use_fedprox: bool = True  # Enable FedProx aggregation algorithm (adds proximal term for better non-IID handling)
    fedprox_mu: float = 0.0032927591344236173  # Optimized: 0.0033 (from CICIDS2017 Optuna Trial 0)
   
    # === QUICK VERIFICATION MODE ===
    quick_verify: bool = False  # When True, run a fast built-in self-check path
    
    # === DATA CONFIGURATION ===
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    zero_day_attack: str = "Backdoor"  # PATH 1: Set to Backdoor for overall performance improvement test
    
    # UNSW-NB15 dataset attack types
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
    
    # CICIDS dataset attack types (commented out - using UNSW-NB15 instead)
    '''
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
        return self.attack_types.get(self.zero_day_attack, 4)  # Default to DoS=4 (UNSW-NB15)
    
    # === MODEL CONFIGURATION ===
    input_dim: int = 49  # UNSW-NB15 has 49 features (CICIDS2017 has 43)
    hidden_dim: int = 512  # Optimized: 512 (from CICIDS2017 Optuna Trial 0)
    embedding_dim: int = 128  # Optimized: 128 (from CICIDS2017 Optuna Trial 0)
    num_classes: int = 2  # Binary classification (Normal vs Attack) for zero-day detection
    
    # === MULTI-PROTOTYPE CONFIGURATION ===
    use_multi_prototype: bool = True  # Enable multi-prototype approach (one prototype per attack type)
    # When enabled: Normal class uses 1 prototype, Attack class uses multiple prototypes (one per attack type)
    # Benefits: Better representation of diverse attack patterns, especially for Fuzzers and Worms
    
    # === FEATURE SELECTION CONFIGURATION ===
    use_feature_selection: bool = True  # Enable XGBoost-based feature selection (following MIX_LSTM approach)
    feature_selection_ratio: float = 0.8  # Select top 80% of features (not used with XGBoost - uses n_features_final instead)
    
    # === TCN CONFIGURATION ===
    use_tcn: bool = False  # DISABLED: Removed TCN sequence creation - using packet-level features instead
    disable_tcn_feature_extraction: bool = True  # Using simple pooling instead of TCN (packet-level features)
    sequence_length: int = 25  # Optimized: 25 (from CICIDS2017 Optuna Trial 0)
    sequence_stride: int = 12  # Optimized: 12 (from CICIDS2017 Optuna Trial 0)
    sequence_labeling_threshold: float = 0.5  # Threshold-based labeling: label as zero-day if ≥50% of sequence is zero-day
    min_sequences_per_attack_type: int = 5  # Minimum sequences per attack type (combine rare types if below)
    combine_rare_attack_types: bool = True  # Combine attack types with <min_sequences into "Other Attacks" class
    oversample_known_attacks_before_sequences: bool = True  # Oversample known attack packets before sequence creation
    known_attack_oversample_factor: float = 3.0  # Multiply known attack samples by this factor before sequences
    apply_test_sequence_filtering: bool = True  # Enforce realistic prevalence after sequence creation
    test_seq_normal_pct: float = 0.95  # Realistic prevalence: Normal
    test_seq_known_attack_pct: float = 0.04  # Realistic prevalence: Known attacks
    test_seq_zero_day_pct: float = 0.01  # Realistic prevalence: Zero-day attacks
    min_saved_test_sequences: int = 200  # Require enough sequences to trust saved test set
    tcn_kernel_sizes: tuple = (3, 4, 4)  # Optimized: (3, 4, 4) (from CICIDS2017 Optuna Trial 0)
    use_residual_connections: bool = False  # Optimized: False (from CICIDS2017 Optuna Trial 0)
    meta_epochs: int = 40  # Increased to 40 for better meta-learning convergence
    transductive_steps: int = 16  # Optimized: 16 (from CICIDS2017 Optuna Trial 0)
    transductive_lr: float = 0.0007
    
    # === CONTRASTIVE LOSS CONFIGURATION ===
    use_contrastive_loss: bool = True  # Enable contrastive loss for explicit class separation
    # UPDATED: Increased weight and margin for better embedding separation (addressing 63% embedding quality issue)
    # Target: >80% attack samples closer to Attack prototype (currently 63%)
    contrastive_loss_weight: float = 1.0  # REVERTED: Back to 1.0 (Path 1 caused catastrophic regression - 0% recall)
    contrastive_margin: float = 2.0  # REVERTED: Back to 2.0 (Path 1 caused catastrophic regression)
    contrastive_temperature: float = 0.1  # Temperature for similarity scaling
    
    # === TEST-TIME TRAINING (TTT) CONFIGURATION ===
    # FIX: Two-Phase TTT Adaptation to prevent boundary shift conflict
    # When query set is mixed (known + zero-day), single-phase TTT causes boundary shift
    # Two-phase: Phase 1 adapts on known attacks, Phase 2 fine-tunes on zero-day
    use_two_phase_ttt: bool = True  # Enable two-phase TTT to prevent known attack performance degradation
    ttt_base_steps: int = 50  # REVERTED: Back to quick test setting (300 caused regression, full training: 258)
    ttt_max_steps: int = 500  # Maximum TTT steps (safety limit)
    ttt_adaptation_query_size: int = 500  # REVERTED: Back to quick test setting (2000 may have contributed to regression, full training: 1514)
    ttt_batch_size: int = 16  # Optimized: 16 (from CICIDS2017 Optuna Trial 0)
    ttt_lr: float = 0.0001518747922672249  # Optimized: 0.000152 (from CICIDS2017 Optuna Trial 0)
    
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
    ttt_temperature: float = 1.4401524937396013  # Optimized: 1.440 (from CICIDS2017 Optuna Trial 0)
    use_semisupervised_ttt: bool = False  # When True, use new semi-supervised TTT instead of legacy TENT+pseudo

    # === TENT + PSEUDO-LABELS CONFIGURATION ===
    # Optimized hyperparameters from CICIDS2017 Optuna (Balanced Base Model + TTT Performance)
    # Optimized for balanced_base_ttt metric: 40% base F1 + 30% TTT ZDR + 30% TTT F1
    # Best Trial 9 Score: 0.6761 (Base F1: 0.5455, TTT ZDR: 1.0000, TTT F1: 0.5263)
    use_pseudo_labels: bool = False  # Optimized: False (from CICIDS2017 Optuna Trial 0)
    pseudo_threshold: float = 0.9733551198429333  # Optimized: 0.973 (from CICIDS2017 Optuna Trial 0)
    pseudo_min_threshold: float = 0.8448448049611839  # Optimized: 0.845 (from CICIDS2017 Optuna Trial 0)
    pseudo_weight: float = 1.841048247374583  # Optimized: 1.841 (from CICIDS2017 Optuna Trial 0)
    entropy_weight: float = 0.5650515929852795  # Optimized: 0.565 (from CICIDS2017 Optuna Trial 0)
    use_teacher: bool = True  # Optimized: True (from CICIDS2017 Optuna Trial 0)
    ema_decay: float = 0.9547859335863128  # Optimized: 0.955 (from CICIDS2017 Optuna Trial 0)
    
    # === ADVANCED TTT TECHNIQUES FOR SOTA PERFORMANCE ===
    # UPDATED: Enabled focal loss with optimized parameters to address class imbalance
    # Target: Better handling of rare attack types (addressing severe class imbalance)
    # FIX: Focal loss enabled for training but disabled for TTT (focal loss interferes with TTT adaptation)
    use_focal_loss: bool = True  # Enabled for training to handle class imbalance (was False)
    use_focal_loss_ttt: bool = False  # NEW: Disabled for TTT (focal loss makes TTT adaptation harder)
    focal_gamma: float = 2.0  # Increased from 1.552 to 2.0 (focus more on hard examples)
    focal_alpha: float = 0.4  # Increased from 0.332 to 0.4 (favor rare classes more)
    use_mixup_ttt: bool = False  # DISABLED: Mixup inappropriate for TTT - requires labels, mixes noisy pseudo-labels, destroys network flow semantics
    mixup_alpha: float = 0.2  # Not used when use_mixup_ttt=False
    use_label_smoothing: bool = False  # DISABLED: Conflicts with focal loss (focal focuses on hard examples, smoothing softens all labels equally)
    label_smoothing: float = 0.1  # Not used when use_label_smoothing=False
    
    # === CLASS WEIGHT AND MAGIC NUMBERS ===
    # Note: pseudo_label_temperature is defined in TENT + PSEUDO-LABELS CONFIGURATION section above
    transductive_patience: int = 8  # Early stopping patience for transductive optimization (number of steps without improvement)
    missing_class_weight_multiplier: float = 2.0  # REVERTED: Back to 2.0 (Path 1 caused regression)
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
    diversity_weight: float = 0.15  # FIXED: Re-enabled to prevent overfitting (was 0.0, now 0.15)
    
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
    # UPDATED: Increased weight decay to reduce overfitting (training: 84.95%, test: 56.34% = 28% gap)
    # Higher weight decay → better generalization → smaller train/test gap
    meta_learning_weight_decay: float = 5e-4  # Increased from 1e-4 to 5e-4 for better regularization
    
    # === EVALUATION CONFIGURATION ===
    support_size: int = 20
    # FIXED: Increased tasks for better meta-learning diversity (with k_shot=5, need more tasks)
    num_meta_tasks: int = 100  # Increased to 100 for better task diversity and generalization
    # UPDATED: Increased support set size to reduce overfitting and improve generalization
    # Larger support sets → more representative prototypes → better test performance
    support_set_size_per_class: int = 300  # Increased from 150 to 300 per class (600 total) for better prototype quality
    
    # === FEW-SHOT LEARNING CONFIGURATION ===
    n_way: int = 2  # Number of classes per task
    # FIXED: Reduced from 41 to TRUE few-shot learning with multi-attack support
    # k_shot=41 was "many-shot" learning, not few-shot. True few-shot uses 1-5 shots.
    # IMPORTANT: k_shot is distributed across 3-5 attack types in support set (Balanced Multi-Attack Support Sets)
    # With k_shot=10: 3-5 attack types × 2-3 samples each = ensures all attack types are represented
    # This prevents hiding attack types in binary classification while maintaining few-shot learning
    k_shot: int = 10  # FIXED: True few-shot learning (was 41) - distributed across 3-5 attack types (2-3 samples each)
    n_query: int = 20  # FIXED: Increased for better task evaluation (was 10 - too small)
    enforce_equal_support_composition: bool = False  # Optimized: False (from Multi-Objective Optuna Trial 6)
    # IMPORTANT: Balanced Multi-Attack Support Sets automatically uses 3-5 attack types per task
    # k_shot samples are distributed across these attack types (ensures all types are represented)
    # This prevents hiding attack types in binary classification while maintaining few-shot learning
    include_all_attack_types_in_support: bool = False  # DEPRECATED: Balanced Multi-Attack Support Sets handles this automatically (uses 3-5 types per task)
    
    # === VALIDATION CONFIGURATION ===
    max_val_samples: int = 500  # Quick test: Reduced from 1000 for faster testing (normal: 1000)
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
            'apply_test_sequence_filtering': self.apply_test_sequence_filtering,
            'test_seq_normal_pct': self.test_seq_normal_pct,
            'test_seq_known_attack_pct': self.test_seq_known_attack_pct,
            'test_seq_zero_day_pct': self.test_seq_zero_day_pct,
            'min_saved_test_sequences': self.min_saved_test_sequences,
            'local_epochs': self.local_epochs,
            'learning_rate': self.learning_rate,
            'ttt_base_steps': self.ttt_base_steps,
            'ttt_max_steps': self.ttt_max_steps,
            'ttt_batch_size': self.ttt_batch_size,
            'ttt_lr': self.ttt_lr,
        }
# Global configuration instance - single source of truth
config = SystemConfig()
# FORCE num_rounds to 20 for better convergence with 3 clients (each client has less data)
config.num_rounds = 20
# FORCE num_clients to 3 for proper federated learning
config.num_clients = 3
# FORCE local_epochs to 30 to compensate for less data per client (3 clients = 1/3 data each)
config.local_epochs = 30
# FORCE k_shot to 50 for 10-shot per attack type (10 samples per type with 5 attack types)
config.k_shot = 50
# Verify it's set correctly
assert config.num_rounds == 20, f"Config num_rounds is {config.num_rounds}, expected 20!"
assert config.num_clients == 3, f"Config num_clients is {config.num_clients}, expected 3!"
assert config.local_epochs == 30, f"Config local_epochs is {config.local_epochs}, expected 30!"
assert config.k_shot == 50, f"Config k_shot is {config.k_shot}, expected 50!"

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
