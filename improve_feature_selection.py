#!/usr/bin/env python3
"""
Improved Feature Selection Strategies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_feature_selection():
    """Test different feature selection strategies"""
    
    logger.info("🔧 Testing Improved Feature Selection Strategies")
    logger.info("=" * 60)
    
    # Load preprocessed data (without feature selection)
    from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
    
    class ImprovedUNSWPreprocessor(UNSWPreprocessor):
        def step5_improved_feature_selection(self, df: pd.DataFrame, target_col: str = 'label', 
                                           method: str = 'conservative') -> pd.DataFrame:
            """
            Improved feature selection with different strategies
            
            Args:
                df: Input dataframe
                target_col: Target column name
                method: 'conservative', 'moderate', 'aggressive', or 'none'
                
            Returns:
                df: Dataframe with selected features
            """
            logger.info(f"Step 5: Improved Feature Selection ({method})")
            
            # Separate features and target
            exclude_cols = [target_col, 'attack_cat', 'binary_label']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols]
            y = df[target_col]
            
            logger.info(f"  Input features: {len(feature_cols)}")
            
            if method == 'none':
                logger.info("  Skipping feature selection - using all features")
                selected_features = feature_cols
            else:
                # Calculate Information Gain scores
                logger.info("  Computing Information Gain scores...")
                ig_scores = mutual_info_classif(X, y, random_state=42)
                
                # Train Random Forest and get feature importances
                logger.info("  Training Random Forest for feature importance...")
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                rf_importances = rf.feature_importances_
                
                # Different strategies
                if method == 'conservative':
                    # Keep 95% of features (very conservative)
                    n_features = max(10, int(len(feature_cols) * 0.95))
                    logger.info(f"  Conservative: Keeping {n_features} features (95%)")
                    
                elif method == 'moderate':
                    # Keep 90% of features (moderate)
                    n_features = max(10, int(len(feature_cols) * 0.90))
                    logger.info(f"  Moderate: Keeping {n_features} features (90%)")
                    
                elif method == 'aggressive':
                    # Keep 80% of features (original)
                    n_features = max(10, int(len(feature_cols) * 0.80))
                    logger.info(f"  Aggressive: Keeping {n_features} features (80%)")
                
                # Use only Random Forest importance (more stable than hybrid)
                selector = SelectKBest(score_func=lambda X, y: rf_importances, k=n_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            logger.info(f"  Selected {len(selected_features)} features out of {len(feature_cols)}")
            logger.info(f"  Top 10 selected features: {selected_features[:10]}")
            
            # Return dataframe with selected features + target + attack_cat
            selected_features.append(target_col)
            if 'attack_cat' in df.columns:
                selected_features.append('attack_cat')
            df_selected = df[selected_features].copy()
            
            # Store for later use
            feature_only_cols = [col for col in selected_features if col not in [target_col, 'attack_cat']]
            self.selected_features = feature_only_cols
            
            logger.info(f"  Final shape: {df_selected.shape}")
            return df_selected
        
        def preprocess_unsw_dataset(self, zero_day_attack: str = 'DoS', feature_selection_method: str = 'conservative'):
            """Complete preprocessing pipeline with improved feature selection"""
            logger.info("Starting UNSW-NB15 preprocessing pipeline (IMPROVED FEATURE SELECTION)")
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
            train_df = self.step5_improved_feature_selection(train_df, method=feature_selection_method)
            
            # Process test data
            logger.info("\nProcessing test data...")
            test_quality = self.step1_data_quality_assessment(test_df)
            test_df = self.step2_feature_engineering(test_df)
            test_df = self.step3_data_cleaning(test_df)
            test_df = self.step4_categorical_encoding(test_df)
            test_df = self.step5_improved_feature_selection(test_df, method=feature_selection_method)
            
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
            
            # Create zero-day indices
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
    
    # Test different feature selection methods
    methods = ['none', 'conservative', 'moderate', 'aggressive']
    results = {}
    
    for method in methods:
        logger.info(f"\n🧪 Testing Method: {method.upper()}")
        logger.info("-" * 40)
        
        preprocessor = ImprovedUNSWPreprocessor()
        data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS', feature_selection_method=method)
        
        X_train = data['X_train'].numpy()
        y_train = data['y_train'].numpy()
        X_test = data['X_test'].numpy()
        y_test = data['y_test'].numpy()
        
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[method] = {
            'accuracy': acc,
            'f1': f1,
            'n_features': X_train.shape[1]
        }
        
        logger.info(f"Results: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    # Summary
    logger.info(f"\n📊 FEATURE SELECTION COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Method':<12} {'Features':<8} {'Accuracy':<8} {'F1-Score':<8} {'Improvement':<12}")
    logger.info("-" * 60)
    
    baseline_acc = results['none']['accuracy']
    baseline_f1 = results['none']['f1']
    
    for method in methods:
        r = results[method]
        acc_improvement = r['accuracy'] - baseline_acc
        f1_improvement = r['f1'] - baseline_f1
        improvement = f"+{acc_improvement:.3f}" if acc_improvement > 0 else f"{acc_improvement:.3f}"
        
        logger.info(f"{method:<12} {r['n_features']:<8} {r['accuracy']:<8.4f} {r['f1']:<8.4f} {improvement:<12}")
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_acc = results[best_method]['accuracy']
    best_f1 = results[best_method]['f1']
    best_features = results[best_method]['n_features']
    
    logger.info(f"\n🏆 BEST METHOD: {best_method.upper()}")
    logger.info(f"   Accuracy: {best_acc:.4f}")
    logger.info(f"   F1-Score: {best_f1:.4f}")
    logger.info(f"   Features: {best_features}")
    
    return results

if __name__ == "__main__":
    results = test_improved_feature_selection()





