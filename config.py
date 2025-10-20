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
    num_clients: int = 3
    num_rounds: int = 3 # Increased rounds for better federated learning convergence
    local_epochs: int = 50  # Balanced epochs per round for better federated learning
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # === DATA CONFIGURATION ===
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    zero_day_attack: str = "Exploits"  # Single place to control attack type
    
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
    ttt_base_steps: int = 100  # Base number of TTT adaptation steps
    ttt_max_steps: int = 300  # Maximum TTT steps (safety limit)
    ttt_lr: float = 0.0005  # TTT learning rate (optimized for stability)
    ttt_lr_min: float = 1e-6  # Minimum learning rate
    ttt_lr_decay: float = 0.8  # Learning rate decay factor
    ttt_warmup_steps: int = 10  # Learning rate warmup steps
    ttt_weight_decay: float = 1e-5  # TTT weight decay (reduced for stability)
    ttt_patience: int = 20  # Early stopping patience (increased for better convergence)
    ttt_timeout: int = 45  # TTT timeout in seconds (increased)
    ttt_improvement_threshold: float = 1e-5  # Minimum improvement threshold (more sensitive)
    
    # === BLOCKCHAIN CONFIGURATION ===
    ethereum_rpc_url: str = "http://localhost:8545"
    contract_address: str = "0x74f2D28CEC2c97186dE1A02C1Bae84D19A7E8BC8"
    incentive_contract_address: str = "0x02090bbB57546b0bb224880a3b93D2Ffb0dde144"
    private_key: str = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"
    aggregator_address: str = "0x4565f36D8E3cBC1c7187ea39Eb613E484411e075"
    ipfs_url: str = "http://localhost:5001"
    enable_incentives: bool = True
    base_reward: int = 100
    max_reward: int = 1000
    fully_decentralized: bool = False
    min_reputation: int = 100
    
    # === TRAINING CONFIGURATION ===
    support_weight: float = 0.3
    test_weight: float = 0.7
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
    
    # === BLOCKCHAIN ADDRESSES ===
    ganache_addresses: list = None  # Will be set to default addresses if None
    
    # === DEVICE CONFIGURATION ===
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_default_ganache_addresses(self) -> list:
        """Get default Ganache addresses for testing"""
        if self.ganache_addresses is None:
            return [
                "0xCD3a95b26EA98a04934CCf6C766f9406496CA986",
                "0x32cE285CF96cf83226552A9c3427Bd58c0A9AccD", 
                "0x8EbA3b47c80a5E31b4Ea6fED4d5De8ebc93B8d6f",
                "0x4565f36D8E3cBC1c7187ea39Eb613E484411e075",
                "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
                "0x74f2D28CEC2c97186dE1A02C1Bae84D19A7E8BC8",
                "0x02090bbB57546b0bb224880a3b93D2Ffb0dde144",
                "0x1234567890123456789012345678901234567890",
                "0x2345678901234567890123456789012345678901",
                "0x3456789012345678901234567890123456789012"
            ]
        return self.ganache_addresses
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create configuration from environment variables (optional)"""
        return cls(
            num_rounds=int(os.getenv('NUM_ROUNDS', 5)),
            num_clients=int(os.getenv('NUM_CLIENTS', 3)),
            zero_day_attack=os.getenv('ZERO_DAY_ATTACK', 'Exploits'),
            use_tcn=os.getenv('USE_TCN', 'true').lower() == 'true',
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging"""
        return {
            'num_rounds': self.num_rounds,
            'num_clients': self.num_clients,
            'zero_day_attack': self.zero_day_attack,
            'use_tcn': self.use_tcn,
            'sequence_length': self.sequence_length,
            'sequence_stride': self.sequence_stride,
            'local_epochs': self.local_epochs,
            'learning_rate': self.learning_rate,
            'ttt_base_steps': self.ttt_base_steps,
            'ttt_max_steps': self.ttt_max_steps,
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
