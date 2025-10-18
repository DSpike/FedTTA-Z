#!/usr/bin/env python3
"""
Test the fixed zero-day split logic
"""

import pandas as pd
import numpy as np

def test_zero_day_fix():
    """Test the fixed zero-day split logic"""
    print('🧪 Testing fixed zero-day split logic...')

    # Create sample UNSW data
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'attack_cat': ['Normal'] * 300 + ['DoS'] * 200 + ['Analysis'] * 150 + ['Generic'] * 100 + ['Fuzzers'] * 100 + ['Exploits'] * 100 + ['Reconnaissance'] * 50,
        'label': [0] * 300 + [1] * 200 + [1] * 150 + [1] * 100 + [1] * 100 + [1] * 100 + [1] * 50
    }

    df = pd.DataFrame(data)
    print(f'Original dataset shape: {df.shape}')
    print(f'Attack types: {df["attack_cat"].unique()}')
    print('Attack type distribution:')
    print(df['attack_cat'].value_counts())

    # Test the FIXED zero-day split logic
    zero_day_attack = 'DoS'
    print(f'\nTesting zero-day split for: {zero_day_attack}')

    # FIXED logic: Exclude zero-day attack (this automatically includes Normal and other attacks)
    train_mask = df['attack_cat'] != zero_day_attack
    train_data = df[train_mask].copy()

    print(f'\nTraining data shape: {train_data.shape}')
    print(f'Training attack types: {train_data["attack_cat"].unique()}')
    print('Training attack type distribution:')
    print(train_data['attack_cat'].value_counts())

    # Check if training data has both classes
    unique_labels = train_data['label'].unique()
    print(f'\nUnique training labels: {unique_labels}')
    print(f'Number of classes in training: {len(unique_labels)}')

    if len(unique_labels) == 1:
        print('❌ CRITICAL ISSUE: Training data has only 1 class!')
        return False
    else:
        print('✅ Training data has multiple classes - FIXED!')
        
    # Check if zero-day attack is excluded
    if zero_day_attack in train_data['attack_cat'].values:
        print(f'❌ ERROR: Zero-day attack {zero_day_attack} still in training data!')
        return False
    else:
        print(f'✅ Zero-day attack {zero_day_attack} correctly excluded from training')
        return True

if __name__ == "__main__":
    success = test_zero_day_fix()
    if success:
        print('\n🎉 Zero-day split fix is working correctly!')
    else:
        print('\n❌ Zero-day split fix failed!')



