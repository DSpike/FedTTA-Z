"""
Test best hyperparameters on Backdoor as zero-day attack
"""
import json
import sys
from config import SystemConfig
from main import BlockchainFederatedIncentiveSystem

def load_and_apply_best_hyperparameters(config: SystemConfig):
    """Load and apply best hyperparameters from optimization"""
    with open('best_hyperparameters.json', 'r') as f:
        best = json.load(f)
    
    params = best['best_params']
    
    print("=" * 80)
    print("📊 APPLYING BEST HYPERPARAMETERS FROM OPTIMIZATION")
    print("=" * 80)
    print(f"Best Trial: {best['best_trial_number']}")
    print(f"Best Zero-Day Detection Rate: {best['best_value']:.4f}")
    print(f"Zero-Day Attack (original): Exploits")
    print(f"Zero-Day Attack (testing): Backdoor")
    print()
    
    # Apply hyperparameters
    config.num_clients = params['num_clients']
    config.num_rounds = params['num_rounds']
    config.dirichlet_alpha = params['dirichlet_alpha']
    config.learning_rate = params['meta_learning_rate']
    config.meta_epochs = params['meta_epochs']
    config.k_shot = params['k_shot']
    config.n_query = params['n_query']
    config.num_meta_tasks = params['num_meta_tasks']
    config.hidden_dim = params['hidden_dim']
    config.embedding_dim = params['embedding_dim']
    config.sequence_length = params['sequence_length']
    config.sequence_stride = params['sequence_stride']
    config.tcn_kernel_sizes = (params['tcn_kernel_size_1'], 
                               params['tcn_kernel_size_2'], 
                               params['tcn_kernel_size_3'])
    config.use_residual_connections = params['use_residual_connections']
    
    # TTT parameters
    config.ttt_lr = params['ttt_lr']
    config.ttt_base_steps = params['ttt_base_steps']
    config.ttt_batch_size = params['ttt_batch_size']
    config.ttt_adaptation_query_size = params['ttt_adaptation_query_size']
    
    # TENT + Pseudo-Labels
    config.use_pseudo_labels = params['use_pseudo_labels']
    config.pseudo_weight = params['pseudo_weight']
    config.entropy_weight = params['entropy_weight']
    config.pseudo_threshold = params['pseudo_threshold']
    config.pseudo_min_threshold = params['pseudo_min_threshold']
    config.use_teacher = params['use_teacher']
    config.ema_decay = params['ema_decay']
    config.pseudo_label_temperature = params['pseudo_label_temperature']
    config.ttt_temperature = params['ttt_temperature']
    
    # Advanced TTT
    config.use_focal_loss = params['use_focal_loss']
    config.focal_gamma = params['focal_gamma']
    config.focal_alpha = params['focal_alpha']
    
    # FedProx
    config.fedprox_mu = params['fedprox_mu']
    
    # Set zero-day attack to Backdoor
    config.zero_day_attack = "Backdoor"
    
    print("✅ Applied all hyperparameters")
    print(f"   Zero-Day Attack: {config.zero_day_attack}")
    print(f"   Meta Epochs: {config.meta_epochs}")
    print(f"   K-shot: {config.k_shot}")
    print(f"   Use Residual: {config.use_residual_connections}")
    print(f"   Use Teacher: {config.use_teacher}")
    print()
    
    return config

def main():
    """Run test with Backdoor as zero-day attack"""
    print("=" * 80)
    print("🧪 TESTING BEST HYPERPARAMETERS ON BACKDOOR ATTACK")
    print("=" * 80)
    print()
    
    # Create config and apply best hyperparameters
    config = SystemConfig()
    config = load_and_apply_best_hyperparameters(config)
    
    # Ensure equal distribution is enabled
    config.enforce_equal_support_composition = True
    config.include_all_attack_types_in_support = True
    
    print("=" * 80)
    print("🚀 STARTING SYSTEM EXECUTION")
    print("=" * 80)
    print()
    
    # Initialize and run system
    system = BlockchainFederatedIncentiveSystem(config)
    
    if not system.initialize_system():
        print("❌ System initialization failed")
        sys.exit(1)
    
    if not system.preprocess_data():
        print("❌ Data preprocessing failed")
        sys.exit(1)
    
    if not system.setup_federated_learning():
        print("❌ Federated learning setup failed")
        sys.exit(1)
    
    # Run federated training
    system.training_history = []
    for round_num in range(1, config.num_rounds + 1):
        system.run_federated_round(round_num)
    
    # Run evaluation
    print()
    print("=" * 80)
    print("📊 RUNNING EVALUATION")
    print("=" * 80)
    print()
    
    system.evaluate_zero_day_detection()
    
    print()
    print("=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()










