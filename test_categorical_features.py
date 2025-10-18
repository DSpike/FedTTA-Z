#!/usr/bin/env python3
"""
Test script to analyze categorical features in UNSW-NB15 dataset
"""

import pandas as pd
import numpy as np

def analyze_categorical_features():
    """Analyze the cardinality of categorical features in UNSW-NB15"""
    
    print("🔍 Analyzing UNSW-NB15 Categorical Features")
    print("=" * 50)
    
    try:
        # Load the dataset
        train_df = pd.read_csv("UNSW_NB15_training-set.csv")
        print(f"✅ Loaded training data: {train_df.shape}")
        
        # Analyze categorical columns
        categorical_cols = ['proto', 'service', 'state']
        
        print("\n📊 Categorical Feature Analysis:")
        print("-" * 40)
        
        total_original_features = len(train_df.columns)
        print(f"Original features: {total_original_features}")
        
        # After feature engineering (45 → 49)
        print(f"After feature engineering: 49 features")
        
        # Analyze each categorical column
        for col in categorical_cols:
            if col in train_df.columns:
                unique_values = train_df[col].unique()
                unique_count = len(unique_values)
                
                print(f"\n🔹 {col.upper()}:")
                print(f"   Unique values: {unique_count}")
                print(f"   Values: {sorted(unique_values)}")
                
                if unique_count > 10:
                    print(f"   → HIGH CARDINALITY → Target Encoding (1 feature)")
                else:
                    print(f"   → LOW CARDINALITY → One-Hot Encoding ({unique_count} features)")
        
        # Calculate expected feature count after encoding
        print(f"\n📈 Expected Feature Count After Encoding:")
        print("-" * 40)
        
        # Start with 49 features (after feature engineering)
        features_after_encoding = 49
        
        for col in categorical_cols:
            if col in train_df.columns:
                unique_count = train_df[col].nunique()
                
                if unique_count > 10:
                    # Target encoding: 1 new feature, remove original
                    features_after_encoding = features_after_encoding - 1 + 1  # No change
                    print(f"   {col}: {unique_count} values → Target encoding → +0 features")
                else:
                    # One-hot encoding: N new features, remove original
                    features_after_encoding = features_after_encoding - 1 + unique_count
                    print(f"   {col}: {unique_count} values → One-hot encoding → +{unique_count-1} features")
        
        print(f"\n🎯 Final expected features: {features_after_encoding}")
        print(f"   Increase: {features_after_encoding - 49} features")
        
    except FileNotFoundError:
        print("❌ UNSW_NB15_training-set.csv not found")
        print("   Using estimated values based on UNSW-NB15 documentation:")
        print("\n📊 Estimated Categorical Feature Analysis:")
        print("-" * 40)
        print("🔹 PROTO: ~133 unique values → Target encoding → +0 features")
        print("🔹 SERVICE: ~13 unique values → Target encoding → +0 features") 
        print("🔹 STATE: ~9 unique values → One-hot encoding → +8 features")
        print("\n🎯 Estimated final features: 49 + 8 = 57 features")
        print("   Note: Actual count may vary based on dataset version")

if __name__ == "__main__":
    analyze_categorical_features()





