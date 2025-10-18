#!/usr/bin/env python3
"""
Fix indentation for the EnhancedSystemConfig class
"""

import re

def fix_class_indentation():
    """Fix indentation for the EnhancedSystemConfig class"""
    
    print("🔧 Fixing class indentation...")
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the EnhancedSystemConfig class specifically
    pattern = r'@dataclass\nclass EnhancedSystemConfig:.*?(?=\nclass|\n\nclass|\Z)'
    
    replacement = '''@dataclass
class EnhancedSystemConfig:
    """Enhanced system configuration with incentive mechanisms"""
    # Data configuration
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    zero_day_attack: str = "DoS"
    
    # Model configuration (restored to best performing)
    input_dim: int = 30
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    num_classes: int = 2
    
    # Training configuration
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    rounds: int = 6
    num_clients: int = 3
    local_epochs: int = 5
    
    # Meta-learning configuration
    meta_learning_rate: float = 0.001
    meta_batch_size: int = 16
    meta_epochs: int = 5
    n_way: int = 2
    k_shot: int = 5
    n_query: int = 15
    
    # TTT configuration
    ttt_steps: int = 21
    ttt_learning_rate: float = 0.0001
    ttt_batch_size: int = 32
    
    # Blockchain configuration
    ethereum_rpc_url: str = "http://127.0.0.1:8545"
    private_key: str = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"
    contract_address: str = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    
    # Incentive configuration
    enable_incentives: bool = True
    base_reward: int = 1000
    performance_multiplier: float = 100.0
    
    # System configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: str = "INFO"
    random_seed: int = 42
    output_dir: str = "performance_plots"
    
    # Data distribution configuration
    alpha: float = 0.5  # Dirichlet distribution parameter for non-IID data
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Performance configuration
    early_stopping_patience: int = 3
    min_delta: float = 0.001
    save_best_model: bool = True
    
    # Visualization configuration
    plot_training_history: bool = True
    plot_performance_comparison: bool = True
    plot_token_distribution: bool = True
    plot_client_performance: bool = True
    
    # Debug configuration
    debug_mode: bool = False
    verbose: bool = True
    save_intermediate_results: bool = False
'''
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed class indentation!")

if __name__ == "__main__":
    fix_class_indentation()







