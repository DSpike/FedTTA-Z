"""Run CICIDS2017 optimization and save results properly"""
from optimize_hyperparameters_cicids import HyperparameterOptimizerCICIDS

if __name__ == "__main__":
    # Create optimizer
    optimizer = HyperparameterOptimizerCICIDS(
        study_name="cicids_zero_day_detection_optimization",
        n_trials=5,  # Run 5 trials
        direction="maximize",
        metric="balanced_base_ttt",  # Balanced metric
        zero_day_attack="PortScan"
    )
    
    # Run optimization
    best_trial = optimizer.optimize()
    
    print(f"\n{'='*80}")
    print("✅ Optimization Complete!")
    print(f"Best Trial: {best_trial.number}")
    print(f"Best Value: {best_trial.value}")
    print(f"{'='*80}")
    
    # Print TCN-specific values
    print("\nTCN Configuration to use in config_loader.py:")
    print(f"  'tcn_kernel_sizes': ({best_trial.params['tcn_kernel_size_1']}, "
          f"{best_trial.params['tcn_kernel_size_2']}, "
          f"{best_trial.params['tcn_kernel_size_3']}),")
    print(f"  'sequence_length': {best_trial.params['sequence_length']},")
    print(f"  'sequence_stride': {best_trial.params['sequence_stride']},")
    print(f"  'hidden_dim': {best_trial.params['hidden_dim']},")
    print(f"  'meta_epochs': {best_trial.params['meta_epochs']},")
    print(f"  'k_shot': {best_trial.params['k_shot']},")
    print(f"  'n_query': {best_trial.params['n_query']},")
    print(f"  'learning_rate': {best_trial.params['meta_learning_rate']},")




