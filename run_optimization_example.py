"""
Example script showing how to run hyperparameter optimization
"""

from optimize_hyperparameters import HyperparameterOptimizer

def main():
    """Example: Optimize for zero-day detection rate"""
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name="zero_day_optimization_example",
        n_trials=20,  # Start with 20 trials for testing
        direction="maximize",
        metric="ttt_zero_day_detection_rate"  # Optimize zero-day detection
    )
    
    # Run optimization
    print("🚀 Starting optimization...")
    best_trial = optimizer.optimize()
    
    print(f"\n✅ Optimization complete!")
    print(f"Best trial: {best_trial.number}")
    print(f"Best zero-day detection rate: {best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    return best_trial

if __name__ == "__main__":
    main()










