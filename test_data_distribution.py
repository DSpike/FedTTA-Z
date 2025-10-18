#!/usr/bin/env python3
"""
Test script to check data distribution issue
"""

import pandas as pd
import numpy as np

def test_zero_day_split():
    """Test the zero-day split logic"""
    print('🔍 Testing zero-day split logic...')

    # Create a sample dataset to test the logic
    data = {
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'Attack_type': ['Normal'] * 400 + ['DoS'] * 300 + ['Analysis'] * 200 + ['Generic'] * 100,
        'Attack_label': [0] * 400 + [1] * 300 + [1] * 200 + [1] * 100
    }

    df = pd.DataFrame(data)
    print(f'Original dataset shape: {df.shape}')
    print(f'Attack types: {df["Attack_type"].unique()}')
    print('Attack type distribution:')
    print(df['Attack_type'].value_counts())
    print('Attack label distribution:')
    print(df['Attack_label'].value_counts())

    # Test the zero-day split logic
    zero_day_attack = 'DoS'
    print(f'\nTesting zero-day split for: {zero_day_attack}')

    # Create training data (exclude zero-day attack)
    train_df = df[df['Attack_type'] != zero_day_attack].copy()
    print(f'\nTraining data shape: {train_df.shape}')
    print(f'Training attack types: {train_df["Attack_type"].unique()}')
    print('Training attack type distribution:')
    print(train_df['Attack_type'].value_counts())
    print('Training attack label distribution:')
    print(train_df['Attack_label'].value_counts())

    # Check if training data has both classes
    unique_labels = train_df['Attack_label'].unique()
    print(f'\nUnique training labels: {unique_labels}')
    print(f'Number of classes in training: {len(unique_labels)}')

    if len(unique_labels) == 1:
        print('❌ CRITICAL ISSUE: Training data has only 1 class!')
        print('This is the root cause of the zero evaluation values.')
        return False
    else:
        print('✅ Training data has multiple classes')
        return True

if __name__ == "__main__":
    test_zero_day_split()



