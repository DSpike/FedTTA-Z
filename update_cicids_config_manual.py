"""Manually update CICIDS2017 config with optimization results"""
import json
from config_loader import DATASET_CONFIGS

# If you have the best trial results from console output, update these values:
BEST_TRIAL_PARAMS = {
    # TCN Configuration
    'tcn_kernel_size_1': 3,  # Update with your best trial value
    'tcn_kernel_size_2': 4,  # Update with your best trial value
    'tcn_kernel_size_3': 4,  # Update with your best trial value
    'sequence_length': 25,   # Update with your best trial value
    'sequence_stride': 12,    # Update with your best trial value
    
    # Model Configuration
    'hidden_dim': 512,        # Update with your best trial value
    'embedding_dim': 128,     # Update with your best trial value
    'meta_epochs': 22,        # Update with your best trial value
    'k_shot': 41,             # Update with your best trial value
    'n_query': 10,            # Update with your best trial value
    'meta_learning_rate': 0.0015751320499779737,  # Update with your best trial value
}

print("="*80)
print("CICIDS2017 Configuration Update")
print("="*80)
print("\nPlease update BEST_TRIAL_PARAMS in this script with your best trial results,")
print("then run: python update_cicids_config_manual.py")
print("\nCurrent values in config_loader.py:")
print(f"  tcn_kernel_sizes: {DATASET_CONFIGS['CICIDS2017']['tcn_kernel_sizes']}")
print(f"  sequence_stride: {DATASET_CONFIGS['CICIDS2017']['sequence_stride']}")
print(f"  hidden_dim: {DATASET_CONFIGS['CICIDS2017']['hidden_dim']}")
print(f"  meta_epochs: {DATASET_CONFIGS['CICIDS2017']['meta_epochs']}")
print(f"  k_shot: {DATASET_CONFIGS['CICIDS2017']['k_shot']}")
print(f"  n_query: {DATASET_CONFIGS['CICIDS2017']['n_query']}")
print(f"  learning_rate: {DATASET_CONFIGS['CICIDS2017']['learning_rate']}")

# Uncomment and update the section below with your actual best trial results
"""
# Update config_loader.py with these values:
DATASET_CONFIGS['CICIDS2017'].update({
    'tcn_kernel_sizes': (BEST_TRIAL_PARAMS['tcn_kernel_size_1'], 
                         BEST_TRIAL_PARAMS['tcn_kernel_size_2'], 
                         BEST_TRIAL_PARAMS['tcn_kernel_size_3']),
    'sequence_stride': BEST_TRIAL_PARAMS['sequence_stride'],
    'hidden_dim': BEST_TRIAL_PARAMS['hidden_dim'],
    'meta_epochs': BEST_TRIAL_PARAMS['meta_epochs'],
    'k_shot': BEST_TRIAL_PARAMS['k_shot'],
    'n_query': BEST_TRIAL_PARAMS['n_query'],
    'learning_rate': BEST_TRIAL_PARAMS['meta_learning_rate'],
})

print("\n✅ Configuration updated!")
print("\nUpdated values:")
print(f"  tcn_kernel_sizes: {DATASET_CONFIGS['CICIDS2017']['tcn_kernel_sizes']}")
print(f"  sequence_stride: {DATASET_CONFIGS['CICIDS2017']['sequence_stride']}")
print(f"  hidden_dim: {DATASET_CONFIGS['CICIDS2017']['hidden_dim']}")
print(f"  meta_epochs: {DATASET_CONFIGS['CICIDS2017']['meta_epochs']}")
print(f"  k_shot: {DATASET_CONFIGS['CICIDS2017']['k_shot']}")
print(f"  n_query: {DATASET_CONFIGS['CICIDS2017']['n_query']}")
print(f"  learning_rate: {DATASET_CONFIGS['CICIDS2017']['learning_rate']}")
"""




