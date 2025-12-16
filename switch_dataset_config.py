"""
Dataset Configuration Switcher
==============================

Helper script to switch between KDD and UNSW-NB15 configurations.
This script modifies config.py to use the appropriate settings for each dataset.

Usage:
    python switch_dataset_config.py --dataset KDD
    python switch_dataset_config.py --dataset UNSW
"""

import argparse
import re
from pathlib import Path
from config_kdd_backup import get_kdd_config

# UNSW-optimized configuration
UNSW_CONFIG = {
    'input_dim': 43,
    'hidden_dim': 256,
    'embedding_dim': 128,
    'sequence_length': 21,
    'sequence_stride': 13,
    'tcn_kernel_sizes': (3, 3, 6),
    'meta_epochs': 18,
    'k_shot': 118,
    'n_query': 20,
    'learning_rate': 0.001096821720752952,
    'confidence_rejection_threshold': 0.70,
    'data_path': "UNSW_NB15_training-set.csv",
    'test_path': "UNSW_NB15_testing-set.csv",
    'zero_day_attack': "DoS",
    'use_category_grouping': False,
}

# KDD-optimized configuration
KDD_CONFIG = get_kdd_config()

# Mapping of parameter names to their patterns in config.py
PARAM_PATTERNS = {
    'input_dim': r'input_dim:\s*int\s*=\s*\d+',
    'hidden_dim': r'hidden_dim:\s*int\s*=\s*\d+',
    'embedding_dim': r'embedding_dim:\s*int\s*=\s*\d+',
    'sequence_length': r'sequence_length:\s*int\s*=\s*\d+',
    'sequence_stride': r'sequence_stride:\s*int\s*=\s*\d+',
    'tcn_kernel_sizes': r'tcn_kernel_sizes:\s*tuple\s*=\s*\([^)]+\)',
    'meta_epochs': r'meta_epochs:\s*int\s*=\s*\d+',
    'k_shot': r'k_shot:\s*int\s*=\s*\d+',
    'n_query': r'n_query:\s*int\s*=\s*\d+',
    'learning_rate': r'learning_rate:\s*float\s*=\s*[\d.]+',
    'confidence_rejection_threshold': r'confidence_rejection_threshold:\s*float\s*=\s*[\d.]+',
    'data_path': r'data_path:\s*str\s*=\s*"[^"]*"',
    'test_path': r'test_path:\s*str\s*=\s*"[^"]*"',
    'zero_day_attack': r'zero_day_attack:\s*str\s*=\s*"[^"]*"',
    'use_category_grouping': r'use_category_grouping:\s*bool\s*=\s*(True|False)',
}

def format_value(value):
    """Format a value for insertion into config.py"""
    if isinstance(value, tuple):
        return str(value)
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, float):
        return str(value)
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)

def update_config_file(dataset: str):
    """Update config.py with the specified dataset configuration"""
    config_file = Path('config.py')
    
    if not config_file.exists():
        print(f"❌ Error: {config_file} not found!")
        return False
    
    # Select configuration
    if dataset.upper() == 'KDD':
        config = KDD_CONFIG
        print("📝 Switching to KDD-optimized configuration...")
    elif dataset.upper() == 'UNSW':
        config = UNSW_CONFIG
        print("📝 Switching to UNSW-NB15-optimized configuration...")
    else:
        print(f"❌ Error: Unknown dataset '{dataset}'. Use 'KDD' or 'UNSW'.")
        return False
    
    # Read config file
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update each parameter
    updated_count = 0
    for param_name, param_value in config.items():
        if param_name in PARAM_PATTERNS:
            pattern = PARAM_PATTERNS[param_name]
            replacement = f'{param_name}: {type(param_value).__name__} = {format_value(param_value)}'
            
            # Find and replace
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                updated_count += 1
                print(f"  ✅ Updated {param_name} = {format_value(param_value)}")
            else:
                print(f"  ⚠️  Could not find pattern for {param_name}")
    
    # Write updated content
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Successfully updated {updated_count} parameters in config.py")
    print(f"📋 Dataset: {dataset.upper()}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Switch between KDD and UNSW-NB15 configurations'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['KDD', 'UNSW', 'kdd', 'unsw'],
        help='Dataset to switch to (KDD or UNSW)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Configuration Switcher")
    print("=" * 60)
    print()
    
    success = update_config_file(args.dataset)
    
    if success:
        print()
        print("=" * 60)
        print("✅ Configuration switch completed!")
        print("=" * 60)
        print()
        print("⚠️  IMPORTANT: Review config.py to ensure all changes are correct.")
        print("⚠️  Some parameters may need manual adjustment (e.g., comments).")
    else:
        print()
        print("=" * 60)
        print("❌ Configuration switch failed!")
        print("=" * 60)

if __name__ == '__main__':
    main()




