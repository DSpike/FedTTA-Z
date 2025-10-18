#!/usr/bin/env python3
"""
Compare Performance With and Without IGRF-RFE Feature Selection
"""

import pandas as pd
import numpy as np
import torch
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_feature_selection():
    """Compare performance with and without IGRF-RFE feature selection"""
    
    logger.info("🔬 Comparing Performance With and Without IGRF-RFE Feature Selection")
    logger.info("=" * 80)
    
    # Test 1: With IGRF-RFE feature selection (current system)
    logger.info("\n📊 TEST 1: WITH IGRF-RFE Feature Selection")
    logger.info("-" * 50)
    
    preprocessor_with_fs = UNSWPreprocessor()
    data_with_fs = preprocessor_with_fs.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    X_train_fs = data_with_fs['X_train'].numpy()
    y_train_fs = data_with_fs['y_train'].numpy()
    X_test_fs = data_with_fs['X_test'].numpy()
    y_test_fs = data_with_fs['y_test'].numpy()
    
    logger.info(f"Features with IGRF-RFE: {X_train_fs.shape[1]}")
    logger.info(f"Training samples: {X_train_fs.shape[0]}")
    logger.info(f"Test samples: {X_test_fs.shape[0]}")
    
    # Train Random Forest with feature selection
    rf_with_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_with_fs.fit(X_train_fs, y_train_fs)
    
    y_pred_fs = rf_with_fs.predict(X_test_fs)
    
    acc_fs = accuracy_score(y_test_fs, y_pred_fs)
    f1_fs = f1_score(y_test_fs, y_pred_fs)
    prec_fs = precision_score(y_test_fs, y_pred_fs)
    rec_fs = recall_score(y_test_fs, y_pred_fs)
    
    logger.info(f"Results WITH IGRF-RFE:")
    logger.info(f"  Accuracy:  {acc_fs:.4f}")
    logger.info(f"  F1-Score:  {f1_fs:.4f}")
    logger.info(f"  Precision: {prec_fs:.4f}")
    logger.info(f"  Recall:    {rec_fs:.4f}")
    
    # Test 2: Without feature selection (all features)
    logger.info("\n📊 TEST 2: WITHOUT Feature Selection (All Features)")
    logger.info("-" * 50)
    
    # Create a modified preprocessor that skips feature selection
    class UNSWPreprocessorNoFS(UNSWPreprocessor):
        def preprocess_unsw_dataset(self, zero_day_attack: str = 'DoS'):
            """Complete preprocessing pipeline WITHOUT feature selection"""
            logger.info("Starting UNSW-NB15 preprocessing pipeline (NO FEATURE SELECTION)")
            logger.info("=" * 60)
            
            # Load datasets
            logger.info("Loading UNSW-NB15 datasets...")
            train_df = pd.read_csv(self.data_path)
            test_df = pd.read_csv(self.test_path)
            
            logger.info(f"Training data: {train_df.shape}")
            logger.info(f"Testing data: {test_df.shape}")
            
            # Process training data
            logger.info("\nProcessing training data...")
            train_quality = self.step1_data_quality_assessment(train_df)
            train_df = self.step2_feature_engineering(train_df)
            train_df = self.step3_data_cleaning(train_df)
            train_df = self.step4_categorical_encoding(train_df)
            # SKIP feature selection step
            
            # Process test data
            logger.info("\nProcessing test data...")
            test_quality = self.step1_data_quality_assessment(test_df)
            test_df = self.step2_feature_engineering(test_df)
            test_df = self.step3_data_cleaning(test_df)
            test_df = self.step4_categorical_encoding(test_df)
            # SKIP feature selection step
            
            # Align features between train and test data
            logger.info("Aligning features between train and test data...")
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            
            # Find missing columns in each dataset
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            
            # Add missing columns with zeros
            for col in missing_in_test:
                test_df[col] = 0
                logger.info(f"  Added missing column to test data: {col}")
            
            for col in missing_in_train:
                train_df[col] = 0
                logger.info(f"  Added missing column to train data: {col}")
            
            # Ensure same column order
            common_cols = sorted(list(train_cols.union(test_cols)))
            train_df = train_df[common_cols]
            test_df = test_df[common_cols]
            
            logger.info(f"  Final feature count - Train: {len(train_df.columns)}, Test: {len(test_df.columns)}")
            
            # Create zero-day split
            split_data = self.create_zero_day_split(train_df, test_df, zero_day_attack)
            
            # Apply feature scaling
            train_scaled, val_scaled, test_scaled = self.step6_feature_scaling(
                split_data['train'], split_data['val'], split_data['test']
            )
            
            # Convert to PyTorch tensors
            feature_cols = [col for col in train_scaled.columns if col not in ['label', 'binary_label', 'attack_cat']]
            
            X_train = torch.FloatTensor(train_scaled[feature_cols].values)
            y_train = torch.LongTensor(train_scaled['binary_label'].values)
            
            X_val = torch.FloatTensor(val_scaled[feature_cols].values)
            y_val = torch.LongTensor(val_scaled['binary_label'].values)
            
            X_test = torch.FloatTensor(test_scaled[feature_cols].values)
            y_test = torch.LongTensor(test_scaled['binary_label'].values)
            
            # Create zero-day indices (indices where label != 0, i.e., attack samples)
            zero_day_indices = torch.where(y_test != 0)[0].tolist()
            
            logger.info("\nPreprocessing completed successfully!")
            logger.info(f"Final feature count: {len(feature_cols)}")
            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Validation samples: {len(X_val)}")
            logger.info(f"Test samples: {len(X_test)}")
            logger.info(f"Zero-day samples: {len(zero_day_indices)}")
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'zero_day_indices': zero_day_indices,
                'feature_names': feature_cols,
                'scaler': self.scaler,
                'target_encoders': self.target_encoders,
                'zero_day_attack': zero_day_attack,
                'attack_types': self.attack_types,
                'quality_reports': {
                    'train': train_quality,
                    'test': test_quality
                }
            }
    
    preprocessor_no_fs = UNSWPreprocessorNoFS()
    data_no_fs = preprocessor_no_fs.preprocess_unsw_dataset(zero_day_attack='DoS')
    
    X_train_no_fs = data_no_fs['X_train'].numpy()
    y_train_no_fs = data_no_fs['y_train'].numpy()
    X_test_no_fs = data_no_fs['X_test'].numpy()
    y_test_no_fs = data_no_fs['y_test'].numpy()
    
    logger.info(f"Features without IGRF-RFE: {X_train_no_fs.shape[1]}")
    logger.info(f"Training samples: {X_train_no_fs.shape[0]}")
    logger.info(f"Test samples: {X_test_no_fs.shape[0]}")
    
    # Train Random Forest without feature selection
    rf_no_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_no_fs.fit(X_train_no_fs, y_train_no_fs)
    
    y_pred_no_fs = rf_no_fs.predict(X_test_no_fs)
    
    acc_no_fs = accuracy_score(y_test_no_fs, y_pred_no_fs)
    f1_no_fs = f1_score(y_test_no_fs, y_pred_no_fs)
    prec_no_fs = precision_score(y_test_no_fs, y_pred_no_fs)
    rec_no_fs = recall_score(y_test_no_fs, y_pred_no_fs)
    
    logger.info(f"Results WITHOUT IGRF-RFE:")
    logger.info(f"  Accuracy:  {acc_no_fs:.4f}")
    logger.info(f"  F1-Score:  {f1_no_fs:.4f}")
    logger.info(f"  Precision: {prec_no_fs:.4f}")
    logger.info(f"  Recall:    {rec_no_fs:.4f}")
    
    # Comparison
    logger.info("\n📈 PERFORMANCE COMPARISON")
    logger.info("=" * 50)
    logger.info(f"Feature Count: {X_train_fs.shape[1]} vs {X_train_no_fs.shape[1]} (reduction: {X_train_no_fs.shape[1] - X_train_fs.shape[1]})")
    logger.info(f"Accuracy:     {acc_fs:.4f} vs {acc_no_fs:.4f} (diff: {acc_fs - acc_no_fs:+.4f})")
    logger.info(f"F1-Score:     {f1_fs:.4f} vs {f1_no_fs:.4f} (diff: {f1_fs - f1_no_fs:+.4f})")
    logger.info(f"Precision:    {prec_fs:.4f} vs {prec_no_fs:.4f} (diff: {prec_fs - prec_no_fs:+.4f})")
    logger.info(f"Recall:       {rec_fs:.4f} vs {rec_no_fs:.4f} (diff: {rec_fs - rec_no_fs:+.4f})")
    
    # Analysis
    logger.info("\n🔍 ANALYSIS")
    logger.info("=" * 50)
    
    if acc_fs > acc_no_fs:
        logger.info("✅ IGRF-RFE IMPROVED accuracy")
    elif acc_fs < acc_no_fs:
        logger.info("❌ IGRF-RFE HURT accuracy")
    else:
        logger.info("➖ IGRF-RFE had NO EFFECT on accuracy")
    
    if f1_fs > f1_no_fs:
        logger.info("✅ IGRF-RFE IMPROVED F1-score")
    elif f1_fs < f1_no_fs:
        logger.info("❌ IGRF-RFE HURT F1-score")
    else:
        logger.info("➖ IGRF-RFE had NO EFFECT on F1-score")
    
    # Feature importance comparison
    logger.info(f"\n🎯 FEATURE IMPORTANCE COMPARISON")
    logger.info("=" * 50)
    
    # Top features from both models
    feature_importance_fs = pd.DataFrame({
        'feature': data_with_fs['feature_names'],
        'importance': rf_with_fs.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_no_fs = pd.DataFrame({
        'feature': data_no_fs['feature_names'],
        'importance': rf_no_fs.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 features WITH IGRF-RFE:")
    for idx, row in feature_importance_fs.head(10).iterrows():
        logger.info(f"  {row['feature']:25} | {row['importance']:.4f}")
    
    logger.info("\nTop 10 features WITHOUT IGRF-RFE:")
    for idx, row in feature_importance_no_fs.head(10).iterrows():
        logger.info(f"  {row['feature']:25} | {row['importance']:.4f}")
    
    return {
        'with_fs': {
            'accuracy': acc_fs,
            'f1': f1_fs,
            'precision': prec_fs,
            'recall': rec_fs,
            'n_features': X_train_fs.shape[1]
        },
        'without_fs': {
            'accuracy': acc_no_fs,
            'f1': f1_no_fs,
            'precision': prec_no_fs,
            'recall': rec_no_fs,
            'n_features': X_train_no_fs.shape[1]
        }
    }

if __name__ == "__main__":
    results = compare_feature_selection()





