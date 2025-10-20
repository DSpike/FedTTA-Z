class EnhancedSystemConfig:
    """Enhanced system configuration with incentive mechanisms"""
    # Data configuration
    data_path: str = "UNSW_NB15_training-set.csv"
    test_path: str = "UNSW_NB15_testing-set.csv"
    # Default value, will be overridden by centralized config
    zero_day_attack: str = "DoS"
    
    # Model configuration (aligned with SystemConfig)
    # Updated after IGRF-RFE feature selection (43 features selected)
    input_dim: int = 43
    hidden_dim: int = 128
    embedding_dim: int = 64
    
    # TCN configuration
    use_tcn: bool = True  # Use TCN-based model instead of linear-based
    # Length of sequences for TCN processing (optimized)
    sequence_length: int = 30
    sequence_stride: int = 15  # Stride for sequence creation
    meta_epochs: int = 3# Reduced epochs for TCN stability
    
    # Prototype update weights (configurable for different data distributions)
    support_weight: float = 0.3  # Weight for support set contribution
    test_weight: float = 0.7     # Weight for test set contribution
    
    # Federated learning configuration (aligned with SystemConfig)
    num_clients: int = 15
    num_rounds: int = 15  # Increased rounds for better federated learning convergence
    local_epochs: int = 50  # Increased for better performance
    learning_rate: float = 0.001
    
    # Blockchain configuration - Using REAL deployed contracts
    ethereum_rpc_url: str = "http://localhost:8545"
    # Deployed FederatedLearning contract
    contract_address: str = "0x74f2D28CEC2c97186dE1A02C1Bae84D19A7E8BC8"
    # Deployed Incentive contract
    incentive_contract_address: str = "0x02090bbB57546b0bb224880a3b93D2Ffb0dde144"
    # Will use first account with ETH
    private_key: str = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"
    # First Ganache account with 100 ETH
    aggregator_address: str = "0x4565f36D8E3cBC1c7187ea39Eb613E484411e075"
    
    # IPFS configuration
    ipfs_url: str = "http://localhost:5001"
    
    # Incentive configuration
    enable_incentives: bool = True
    base_reward: int = 100
    max_reward: int = 1000
    
    # TTT configuration (matching SystemConfig)
    ttt_base_steps: int = 50  # Base number of TTT adaptation steps
    ttt_max_steps: int = 200  # Maximum TTT steps (safety limit)
    ttt_lr: float = 0.001  # TTT learning rate
    ttt_weight_decay: float = 1e-4  # TTT weight decay
    ttt_patience: int = 15  # Early stopping patience
    ttt_timeout: int = 30  # TTT timeout in seconds
    ttt_improvement_threshold: float = 1e-4  # Minimum improvement threshold
    min_reputation: int = 100
    
    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Decentralization configuration
    fully_decentralized: bool = False  # Set to True for 100% decentralized system
    
    # Few-shot learning configuration (aligned with SystemConfig)
    n_way: int = 2  # Number of 