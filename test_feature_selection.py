#!/usr/bin/env python3
"""
Test script to verify feature selection changes work correctly after splitting
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from config import get_config

def test_feature_selection_after_splitting():
    """Test the modified preprocessing pipeline with feature selection after splitting"""
    print("Testing feature selection after splitting...")
    
    try:
        # Get configuration
        config = get_config()
        print(f"Using zero-day attack from config: {config.zero_day_attack}")
        
        # Initialize preprocessor
        preprocessor = UNSWPreprocessor()
        
        # Run preprocessing using config
        data = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)
        
        print("✅ Preprocessing completed successfully!")
        print(f"Training data shape: {data['X_train'].shape}")
        print(f"Validation data shape: {data['X_val'].shape}")
        print(f"Test data shape: {data['X_test'].shape}")
        print(f"Feature count: {len(data['feature_names'])}")
        
        # Check if selected features were stored
        if hasattr(preprocessor, 'selected_features') and preprocessor.selected_features:
            print(f"✅ Selected features: {len(preprocessor.selected_features)} features")
            print(f"Selected features: {preprocessor.selected_features[:10]}{'...' if len(preprocessor.selected_features) > 10 else ''}")
        else:
            print("⚠️  No selected features found")
        
        # Verify all datasets have the same number of features
        train_features = data['X_train'].shape[1]
        val_features = data['X_val'].shape[1]
        test_features = data['X_test'].shape[1]
        
        if train_features == val_features == test_features:
            print(f"✅ All datasets have consistent feature count: {train_features}")
        else:
            print(f"❌ Feature count mismatch - Train: {train_features}, Val: {val_features}, Test: {test_features}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_selection_after_splitting()
    sys.exit(0 if success else 1)
