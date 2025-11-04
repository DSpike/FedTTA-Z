"""
Centralized Configuration System for Blockchain Federated Learning
This module provides a single source of truth for all system configuration.
"""

from dataclasses import dataclass
from typing import Optional
import os
import torch


@dataclass
class SystemConfig:
    """Centralized system configuration - single source of truth"""
    
    # === FEDERATED LEARNING CONFIGURATION ===
    num_clients: int = 5   # Reduced for quick test
    num_rounds: int = 15  # Reduced for quick test
    local_epochs: int = 50  # Balanced epochs per round for better federated learning
    learning_rate: float = 0.001
    batch_size: int = 32
    dirichlet_alpha: float = 0.5 # Dirichlet distribution parameter for non-IID data splitting
                                  # α = 0.1: Very high heterogeneity (extreme non-IID)
                                  # α = 1.0: Moderate heterogeneity (balanced non-IID) - RECOMMENDED
                                  # α = 10.0: Low heterogeneity (near IID)
    
    # === QUICK VERIFICATION MODE ===
    quick_verify: bool = False  # When True, run a fast built-in self-check path
    
    # === DATA CONFIGURATION ===
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    zero_day_attack: str = "Analysis"  # Single place to control attack type
    
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
    
    @property
    def zero_day_attack_label(self) -> int:
        """Get the integer label for the zero-day attack type"""
        return self.attack_types.get(self.zero_day_attack, 5)  # Default to Exploits=5
    
    # === MODEL CONFIGURATION ===
    input_dim: int = 43  # Updated after IGRF-RFE feature selection (43 features selected)
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_classes: int = 2  # Binary classification (Normal vs Attack) for zero-day detection
    
    # === FEATURE SELECTION CONFIGURATION ===
    use_igrf_rfe: bool = True  # Enable IGRF-RFE hybrid feature selection
    feature_selection_ratio: float = 0.8  # Select top 80% of features
    
    # === TCN CONFIGURATION ===
    use_tcn: bool = True
    sequence_length: int = 30
    sequence_stride: int = 15
    meta_epochs: int = 3
    transductive_steps: int = 5
    transductive_lr: float = 0.0005
    
    # === TEST-TIME TRAINING (TTT) CONFIGURATION ===
    ttt_base_steps: int = 50  # Base number of TTT adaptation steps (INCREASED from 10 to 25 for better loss convergence)
    ttt_max_steps: int = 300  # Maximum TTT steps (safety limit)
    ttt_adaptation_query_size: int = 750  # TTT adaptation query set size (CRITICAL: increased from 200 to 500-1000 range for better performance)
    ttt_batch_size: int = 32  # TTT batch size (Testing with 16 to find optimal batch size)
    ttt_lr: float = 3e-5  # TTT learning rate (INCREASED from 1e-5 to 3e-5 for faster adaptation while maintaining stability)
    ttt_lr_min: float = 1e-6  # Minimum learning rate
    ttt_lr_decay: float = 0.8  # Learning rate decay factor
    ttt_warmup_steps: int = 10  # Learning rate warmup steps
    ttt_weight_decay: float = 1e-5  # TTT weight decay (reduced for stability)
    ttt_patience: int = 20  # Early stopping patience (increased for better convergence)
    ttt_timeout: int = 45  # TTT timeout in seconds (increased)
    ttt_improvement_threshold: float = 1e-5  # Minimum improvement threshold (more sensitive)
    ttt_diversity_weight: float = 0.15  # Diversity loss weight (INCREASED from 0.1 to 0.15 to better balance entropy vs diversity)
                                          # This helps entropy and diversity losses decrease together
                                          # Range: 0.05-0.2 (higher = more emphasis on maintaining class diversity)

    # === TENT + PSEUDO-LABELS CONFIGURATION ===
    # New configuration for improved TTT performance (+8-12% vs pure TENT)
    use_pseudo_labels: bool = True  # Enable TENT + Pseudo-labels (RECOMMENDED)
    pseudo_threshold: float = 0.9  # Initial confidence threshold for pseudo-labels (start strict)
    pseudo_min_threshold: float = 0.7  # Minimum threshold - curriculum learning (end relaxed)
    pseudo_weight: float = 1.0  # Weight for pseudo-label loss (supervised signal)
    entropy_weight: float = 0.0  # Weight for entropy loss (lower than pseudo-labels)
    use_teacher: bool = True  # Use EMA teacher model for more stable pseudo-labels
    ema_decay: float = 0.999  # EMA decay rate for teacher model (very smooth)
    
    # Adaptive threshold settings
    use_adaptive_threshold: bool = True  # Use data-adaptive thresholds
    threshold_adaptation_mode: str = 'combined'  # 'scheduled', 'adaptive', or 'combined'
    
    # === TRAINING CONFIGURATION ===
    support_weight: float = 0.5
    test_weight: float = 0.5
    validation_patience_limit: int = 10
    recent_rounds: int = 10
    
    # === EVALUATION CONFIGURATION ===
    support_size: int = 20
    num_meta_tasks: int = 20
    
    # === FEW-SHOT LEARNING CONFIGURATION ===
    n_way: int = 2  # Number of classes per task
    k_shot: int = 50  # Number of support samples per class (increased for better performance)
    n_query: int = 100  # Number of query samples per class
    
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
