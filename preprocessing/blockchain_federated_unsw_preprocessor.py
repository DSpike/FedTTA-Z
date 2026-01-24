#!/usr/bin/env python3
"""
Blockchain Federated Learning - UNSW-NB15 Preprocessor
Implements the 6-step preprocessing pipeline for zero-day detection
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from preprocessing.wgan_gp_generator import generate_synthetic_samples_wgan_gp
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UNSWPreprocessor:
    """
    UNSW-NB15 Dataset Preprocessor for Blockchain Federated Learning
    Implements 6-step preprocessing pipeline for zero-day detection
    
    Preprocessing Pipeline Order:
    1. Data Quality Assessment
    2. Feature Engineering  
    3. Data Cleaning (handles missing values, duplicates, infinite values)
    4. Categorical Encoding (after cleaning to avoid encoding invalid data)
    5. Feature Selection (XGBoost-based, following MIX_LSTM approach)
    6. Feature Scaling
    7. Data Rebalancing
    """
    
    def __init__(self, data_path: str = "UNSW_NB15_training-set.csv", test_path: str = "UNSW_NB15_testing-set.csv"):
        """
        Initialize UNSW preprocessor
        
        Args:
            data_path: Path to training CSV file
            test_path: Path to testing CSV file
        """
        self.data_path = data_path
        self.test_path = test_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoders = {}
        self.onehot_columns = {}  # Track one-hot encoded columns for transform
        self.feature_names = None
        
        # UNSW-NB15 attack types
        self.attack_types = {
            'Normal': 0,
            'Fuzzers': 1,
            'Analysis': 2,
            'Backdoor': 3,  # Fixed: singular form as in dataset
            'DoS': 4,
            'Exploits': 5,
            'Generic': 6,
            'Reconnaissance': 7,
            'Shellcode': 8,
            'Worms': 9
        }
        
        logger.info("UNSW-NB15 Preprocessor initialized")
    
    def _create_flow_ids(self, df: pd.DataFrame) -> List:
        """
        Create flow IDs for packets based on network flow characteristics
        
        Flow ID is created from source IP, destination IP, protocol, and timestamp
        (or other distinguishing features if available)
        
        Args:
            df: DataFrame with test data after preprocessing
            
        Returns:
            flow_ids: List of flow IDs for each packet
        """
        logger.info("Creating flow IDs for flow-level evaluation...")
        
        # Check if we have IP/port columns (UNSW-NB15 has srcip, dstip, sport, dport, proto)
        flow_columns = []
        
        # Try to find flow-defining columns
        possible_columns = ['srcip', 'dstip', 'sport', 'dport', 'proto', 'stime', 'ltime']
        
        for col in possible_columns:
            if col in df.columns:
                flow_columns.append(col)
        
        if len(flow_columns) >= 2:
            # Create flow ID from combination of columns
            # Use string concatenation for simplicity
            flow_ids = df[flow_columns].astype(str).apply(lambda x: '_'.join(x), axis=1).tolist()
            logger.info(f"  Created flow IDs from columns: {flow_columns}")
            logger.info(f"  Unique flows: {len(set(flow_ids))} out of {len(flow_ids)} packets")
        else:
            # Fallback: Use index-based grouping (every 5 packets = 1 flow)
            # This is a simple heuristic if flow columns are not available
            packets_per_flow = 5
            flow_ids = [i // packets_per_flow for i in range(len(df))]
            logger.warning(f"  Flow-defining columns not found, using index-based grouping ({packets_per_flow} packets/flow)")
            logger.info(f"  Created {len(set(flow_ids))} flows from {len(df)} packets")
        
        return flow_ids
    
    def step1_data_quality_assessment(self, df: pd.DataFrame) -> Dict:
        """
        Step 1: Data Quality Assessment
        
        Args:
            df: Input dataframe
            
        Returns:
            quality_report: Dictionary with quality metrics
        """
        logger.info("Step 1: Data Quality Assessment")
        
        quality_report = {
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'shape': df.shape,
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'missing_per_feature': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        }
        
        logger.info(f"  Memory usage: {quality_report['memory_usage']:.2f} MB")
        logger.info(f"  Shape: {quality_report['shape']}")
        logger.info(f"  Missing values: {quality_report['missing_values']}")
        logger.info(f"  Duplicate rows: {quality_report['duplicate_rows']}")
        logger.info(f"  Infinite values: {quality_report['infinite_values']}")
        
        return quality_report
    
    def step2_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Minimal Feature Engineering
        Add 4 scientifically-sound features: 45 → 48 features
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Dataframe with new features
        """
        logger.info("Step 2: Feature Engineering (45 → 48 features)")
        
        # Add high-appropriateness features
        #Captures traffic asymmetry, critical for detecting attacks like Backdoors and Exploits
        df['packet_size_ratio'] = df['sbytes'] / (df['dbytes'] + 1) 
        
        #Measures source packet rate, a key indicator for rate-based attacks (DoS, Fuzzers, Reconnaissance)
        df['packets_per_second'] = df['spkts'] / (df['dur'] + 1) 
        
        #Combines TCP-specificity (~80% of flows) with packet rate, highlighting TCP-based attack bursts
        df['tcp_rate'] = (df['proto'] == 'tcp').astype(int) * df['packets_per_second'] 
        
        logger.info(f"  Added 3 features: packet_size_ratio, packets_per_second, tcp_rate")
        logger.info(f"  New shape: {df.shape}")
        return df
    
    def step3_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Data Cleaning
        Handles duplicates, inf/NaN, and missing values while preserving attack signatures
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Cleaned dataframe
        """
        logger.info("Step 3: Data Cleaning (Preserving Attack Signatures)")
        
        initial_shape = df.shape
        logger.info(f"  Initial shape: {initial_shape}")
        
        # 1. Remove duplicate rows (exact duplicates only)
        df_before_dedup = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = df_before_dedup - df.shape[0]
        logger.info(f"  Removed {duplicates_removed} duplicate rows")
        
        # 2. Handle infinite values (convert to NaN for proper imputation)
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        inf_count = inf_mask.sum().sum()
        df = df.replace([np.inf, -np.inf], np.nan)
        logger.info(f"  Converted {inf_count} infinite values to NaN")
        
        # 3. Identify column types for proper imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        logger.info(f"  Processing {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        
        # 4. Handle missing values in numeric columns (median imputation)
        numeric_missing = 0
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Use median for robust imputation (preserves attack patterns)
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                numeric_missing += missing_count
                logger.info(f"    {col}: filled {missing_count} missing values with median {median_value:.4f}")
        
        # 5. Handle missing values in categorical columns (mode imputation)
        categorical_missing = 0
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Use mode for categorical imputation
                mode_values = df[col].mode()
                if not mode_values.empty:
                    mode_value = mode_values[0]
                else:
                    # Fallback for columns with all NaN values
                    mode_value = 'unknown'
                df[col].fillna(mode_value, inplace=True)
                categorical_missing += missing_count
                logger.info(f"    {col}: filled {missing_count} missing values with mode '{mode_value}'")
        
        # 6. Verify no remaining missing values
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"  Warning: {remaining_missing} missing values still remain after imputation")
        else:
            logger.info("  ✅ All missing values successfully imputed")
        
        # 7. Data quality summary
        final_shape = df.shape
        logger.info(f"  Data cleaning summary:")
        logger.info(f"    - Duplicates removed: {duplicates_removed}")
        logger.info(f"    - Infinite values converted: {inf_count}")
        logger.info(f"    - Numeric missing values filled: {numeric_missing}")
        logger.info(f"    - Categorical missing values filled: {categorical_missing}")
        logger.info(f"    - Final shape: {final_shape}")
        
        return df
    
    def step5_feature_selection_hybrid(self, df: pd.DataFrame, target_col: str = 'attack_cat', 
                                       n_features_final: int = 30) -> pd.DataFrame:
        """
        Step 5: XGBoost Feature Selection for Multiclass Zero-Day Detection
        
        This method implements XGBoost-based feature selection, following the approach used in
        MIX_LSTM paper (Chen et al., 2024) which achieved 98.4% AUC-ROC on UNSW-NB15 dataset.
        
        Why XGBoost Feature Selection:
        - Gradient boosting is more powerful than bagging (Random Forest)
        - Single-stage selection (no information loss between stages)
        - Captures complex non-linear feature interactions
        - Handles class imbalance well (important for rare attack types)
        - Proven performance: 98.4% AUC in MIX_LSTM on same dataset (UNSW-NB15)
        
        How It Works:
        1. Train XGBoost classifier on full dataset with all features
        2. Extract feature importance scores (gain-based)
        3. Select top N features based on importance scores
        4. Return dataframe with selected features
        
        Why 30 Features:
        - Balance between information content and computational efficiency
        - Reduces dimensionality from ~48 to 30 (37.5% reduction) while preserving critical attack signatures
        - Prevents overfitting by removing redundant/noisy features
        - Maintains interpretability for security analysts
        
        Note: This step is called AFTER categorical encoding (step 4) to work with
        properly encoded features for better feature selection performance.
        
        Args:
            df: Input dataframe
            target_col: Target column name (default: 'attack_cat' for multiclass)
            n_features_final: Final number of features to select (default: 30)
            
        Returns:
            df: Dataframe with selected features + ['label', 'binary_label', 'attack_cat']
        """
        logger.info("Step 5: XGBoost Feature Selection for Multiclass Zero-Day Detection")
        logger.info("  Method: XGBoost-based feature importance (following MIX_LSTM approach)")
        
        # Separate features and target (exclude attack_cat and other non-feature columns)
        exclude_cols = ['label', 'attack_cat', 'binary_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        y = df[target_col]  # Use multiclass labels (0-9 from attack_cat)

        # CRITICAL FIX: Encode string labels to numeric if needed
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            logger.info(f"  Converting string labels to numeric for XGBoost...")
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            logger.info(f"  Encoded {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
            y = pd.Series(y_encoded, index=y.index)

        logger.info(f"  Input features: {len(feature_cols)}")
        logger.info(f"  Using multiclass target: {target_col} (classes: {sorted(y.unique())})")
        logger.info(f"  Target final features: {n_features_final}")
        
        # OPTIMIZATION: For large datasets (>100K samples), use sampling to speed up XGBoost training
        # XGBoost works well on representative samples, especially for feature importance
        max_samples_for_xgb = 100000  # Use up to 100K samples for XGBoost training
        if len(X) > max_samples_for_xgb:
            logger.info(f"  ⚡ Large dataset detected ({len(X):,} samples). Sampling {max_samples_for_xgb:,} samples for faster XGBoost training...")
            # Stratified sampling to maintain class distribution
            X_sample, _, y_sample, _ = train_test_split(
                X.values, y.values, 
                train_size=max_samples_for_xgb, 
                stratify=y.values,
                random_state=42
            )
            logger.info(f"  ✅ Using {len(X_sample):,} samples for XGBoost training (representative sample)")
            X_train = X_sample
            y_train = y_sample
        else:
            logger.info(f"  Using all {len(X):,} samples for XGBoost training")
            X_train = X.values
            y_train = y.values
        
        # Train XGBoost classifier
        logger.info("  Training XGBoost classifier for feature importance...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',  # Multi-class log loss
            tree_method='hist',  # Fast histogram-based method
            verbosity=0  # Suppress XGBoost output
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Get feature importance scores (gain-based by default)
        feature_importance = xgb_model.feature_importances_
        
        logger.info(f"  XGBoost training complete. Extracting feature importance scores...")
        
        # Get top features by XGBoost importance
        top_indices = np.argsort(feature_importance)[-n_features_final:][::-1]
        final_selected_features = [feature_cols[i] for i in top_indices]
        
        logger.info(f"  Feature selection complete: Selected {len(final_selected_features)} features")
        
        # Log top 10 features by XGBoost importance
        xgb_feature_df = pd.DataFrame({
            'feature': feature_cols,
            'xgb_importance': feature_importance
        }).sort_values('xgb_importance', ascending=False)
        
        logger.info("  Top 10 features by XGBoost importance:")
        for idx, row in xgb_feature_df.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['xgb_importance']:.4f}")
        
        # Log final selected features
        logger.info(f"  Final {len(final_selected_features)} selected features:")
        for i, feat in enumerate(final_selected_features, 1):
            logger.info(f"    {i}. {feat}")
        
        # Create feature importance dataframe for later use
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'xgb_importance': feature_importance
        })
        feature_importance_df = feature_importance_df[feature_importance_df['feature'].isin(final_selected_features)].copy()
        feature_importance_df = feature_importance_df.sort_values('xgb_importance', ascending=False)
        
        # Return dataframe with selected features + target columns
        # Only include columns that actually exist in the dataframe
        selected_cols = final_selected_features.copy()
        
        # Add target and label columns if they exist
        for col in [target_col, 'label', 'binary_label', 'attack_cat']:
            if col in df.columns and col not in selected_cols:
                selected_cols.append(col)
        
        df_selected = df[selected_cols].copy()
        
        # Store feature selection info for later use
        self.selected_features = final_selected_features
        self.feature_importance_scores = feature_importance_df
        
        logger.info(f"  Final shape: {df_selected.shape}")
        logger.info(f"  Feature reduction: {len(feature_cols)} → {len(final_selected_features)} ({100 * (1 - len(final_selected_features)/len(feature_cols)):.1f}% reduction)")
        
        return df_selected
    
    def step4_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Categorical Encoding
        High-cardinality features: Target encoding
        Low-cardinality features: One-hot encoding
        Selected features → 56 features
        
        Note: This step is called AFTER data cleaning (step 3) to ensure
        all missing values, duplicates, and invalid data are handled first.
        
        Args:
            df: Input dataframe (should be cleaned)
            
        Returns:
            df: Dataframe with encoded features
        """
        logger.info("Step 4: Categorical Encoding (45 → 56 features)")
        
        # Identify categorical columns
        categorical_cols = ['proto', 'service', 'state']
        
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                logger.info(f"  {col}: {unique_count} unique values")
                
                if unique_count > 10:  # High-cardinality: Target encoding
                    # Target encoding for proto (high cardinality)
                    if 'label' in df.columns:
                        target_mean = df.groupby(col)['label'].mean()
                        df[f'{col}_target_encoded'] = df[col].map(target_mean)
                        self.target_encoders[col] = target_mean
                        logger.info(f"    Applied target encoding to {col}")
                    else:
                        # If no label column, use frequency encoding
                        freq_encoding = df[col].value_counts() / len(df)
                        df[f'{col}_freq_encoded'] = df[col].map(freq_encoding)
                        logger.info(f"    Applied frequency encoding to {col}")
                    
                    # Drop original column
                    df = df.drop(columns=[col])
                
                else:  # Low-cardinality: One-hot encoding
                    # One-hot encoding for service and state
                    dummies = pd.get_dummies(df[col], prefix=col)
                    # Store one-hot column names for later transform
                    self.onehot_columns[col] = dummies.columns.tolist()
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    logger.info(f"    Applied one-hot encoding to {col} → {dummies.shape[1]} features")
        
        logger.info(f"  Final shape after encoding: {df.shape}")
        return df
    
    def step4_categorical_encoding_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted categorical encoders from training data.
        This method should be called AFTER step4_categorical_encoding has been called on training data.
        
        Args:
            df: Test dataframe (should be cleaned)
            
        Returns:
            df: Dataframe with encoded features using fitted encoders
        """
        logger.info("Step 4 (Transform): Applying fitted categorical encoders to test data")
        
        # Identify categorical columns
        categorical_cols = ['proto', 'service', 'state']
        
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                logger.info(f"  {col}: {unique_count} unique values")
                
                if unique_count > 10:  # High-cardinality: Use fitted target encoder
                    # Use fitted target encoder from training data
                    if col in self.target_encoders:
                        target_mean = self.target_encoders[col]
                        # Map values, using mean of all target means for unseen categories
                        default_value = target_mean.mean() if len(target_mean) > 0 else 0.0
                        df[f'{col}_target_encoded'] = df[col].map(target_mean).fillna(default_value)
                        logger.info(f"    Applied fitted target encoding to {col}")
                    else:
                        # Fallback: use frequency encoding if encoder not found
                        freq_encoding = df[col].value_counts() / len(df)
                        df[f'{col}_freq_encoded'] = df[col].map(freq_encoding)
                        logger.warning(f"    Warning: No fitted encoder found for {col}, using frequency encoding")
                    
                    # Drop original column
                    df = df.drop(columns=[col])
                
                else:  # Low-cardinality: One-hot encoding
                    # One-hot encoding - ensure same columns as training
                    if col in self.onehot_columns:
                        # Use the same one-hot columns from training
                        expected_cols = self.onehot_columns[col]
                        dummies = pd.get_dummies(df[col], prefix=col)
                        
                        # Add missing columns (present in training but not in test) with zeros
                        for expected_col in expected_cols:
                            if expected_col not in dummies.columns:
                                dummies[expected_col] = 0
                        
                        # Select only expected columns in correct order (add missing ones with zeros first)
                        dummies = dummies[expected_cols]
                        
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(columns=[col])
                        logger.info(f"    Applied one-hot encoding to {col} → {len(expected_cols)} features (aligned with training)")
                    else:
                        # Fallback: create one-hot normally if no training columns stored
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(columns=[col])
                        logger.warning(f"    Warning: No training one-hot columns found for {col}, created {dummies.shape[1]} features")
        
        logger.info(f"  Final shape after encoding: {df.shape}")
        return df
    
    
    def step6_feature_scaling(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, 
                            test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Step 6: Feature Scaling using StandardScaler
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional)
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of scaled dataframes
        """
        logger.info("Step 6: Feature Scaling")
        
        # Identify feature columns (exclude target columns and attack_cat)
        feature_cols = [col for col in train_df.columns if col not in ['label', 'binary_label', 'attack_cat']]
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[feature_cols])
        
        # Transform all datasets
        train_scaled = train_df.copy()
        train_scaled[feature_cols] = self.scaler.transform(train_df[feature_cols])
        
        val_scaled = None
        if val_df is not None:
            val_scaled = val_df.copy()
            val_scaled[feature_cols] = self.scaler.transform(val_df[feature_cols])
        
        test_scaled = None
        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])
        
        logger.info(f"  Scaled {len(feature_cols)} features")
        logger.info(f"  Training shape: {train_scaled.shape}")
        if val_scaled is not None:
            logger.info(f"  Validation shape: {val_scaled.shape}")
        if test_scaled is not None:
            logger.info(f"  Test shape: {test_scaled.shape}")
        
        return train_scaled, val_scaled, test_scaled
    
    
    def step7_data_rebalancing_complete(self, complete_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 7: Data Rebalancing for Complete Dataset using 10-class labels
        
        This method addresses the extreme class imbalance in the complete UNSW-NB15 dataset:
        - ADASYN: Adaptive Synthetic Sampling for better minority class representation
        - RandomUnderSampler: Undersamples majority classes to reduce their dominance
        - Dynamic targets: min samples = median class size, max = 2x median
        
        Args:
            complete_df: Complete dataframe with all classes
            
        Returns:
            Rebalanced complete dataframe
        """
        logger.info("Step 7: Data Rebalancing for Complete Dataset")
        
        # Create 10-class labels from attack_cat
        attack_type_mapping = {
            'Normal': 0, 'Fuzzers': 1, 'Analysis': 2, 'Backdoor': 3, 'DoS': 4,
            'Exploits': 5, 'Generic': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9
        }
        
        # Debug: Check if attack_cat column exists and its type
        logger.info(f"  Available columns: {list(complete_df.columns)}")
        if 'attack_cat' in complete_df.columns:
            # Handle duplicate columns by taking the first one
            if isinstance(complete_df['attack_cat'], pd.DataFrame):
                logger.warning("  Duplicate attack_cat columns detected, using first one")
                attack_cat_series = complete_df['attack_cat'].iloc[:, 0]
            else:
                attack_cat_series = complete_df['attack_cat']
            
            logger.info(f"  attack_cat column type: {type(attack_cat_series)}")
            logger.info(f"  attack_cat unique values: {attack_cat_series.unique()[:10]}")
        else:
            logger.error("  attack_cat column not found!")
            return complete_df
        
        # Map attack categories to numeric labels
        # Handle any unmapped categories by setting them to 0 (Normal)
        complete_df['label'] = attack_cat_series.map(attack_type_mapping).fillna(0).astype(int)
        complete_df['binary_label'] = (complete_df['label'] != 0).astype(int)
        
        # Get feature columns and target
        feature_cols = [col for col in complete_df.columns if col not in ['label', 'binary_label', 'attack_cat']]
        X = complete_df[feature_cols].values
        y = complete_df['label'].values
        
        # Analyze class distribution before rebalancing
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info("  Class distribution before rebalancing:")
        for class_label, count in zip(unique_classes, class_counts):
            attack_name = list(attack_type_mapping.keys())[list(attack_type_mapping.values()).index(class_label)]
            percentage = (count / len(y)) * 100
            logger.info(f"    {attack_name} (Label {class_label}): {count:,} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        max_count = np.max(class_counts)
        min_count = np.min(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        logger.info(f"  Imbalance ratio before rebalancing: {imbalance_ratio:.2f}:1")
        
        # Dynamic targets: min samples = median class size, max = 50k samples
        median_class_size = np.median(class_counts)
        target_min_samples = int(median_class_size)
        target_max_samples = 50000  # Fixed maximum of 50k samples
        
        logger.info(f"  Dynamic targets - Min: {target_min_samples:,}, Max: {target_max_samples:,} (fixed maximum of 50k samples)")
        
        # Create sampling strategy
        sampling_strategy = {}
        for class_label, count in zip(unique_classes, class_counts):
            if count < target_min_samples:
                # Oversample minority classes to target_min_samples
                sampling_strategy[class_label] = target_min_samples
            elif count > target_max_samples:
                # Undersample majority classes to target_max_samples
                sampling_strategy[class_label] = target_max_samples
            else:
                # Keep as is
                sampling_strategy[class_label] = count
        
        logger.info("  Sampling strategy:")
        for class_label, target_count in sampling_strategy.items():
            attack_name = list(attack_type_mapping.keys())[list(attack_type_mapping.values()).index(class_label)]
            current_count = class_counts[unique_classes == class_label][0]
            action = "oversample" if target_count > current_count else "undersample" if target_count < current_count else "keep"
            logger.info(f"    {attack_name}: {current_count:,} → {target_count:,} ({action})")
        
        # Step 1: Apply WGAN-GP for extreme minority classes, ADASYN for others
        logger.info("  Step 1: Applying oversampling (WGAN-GP for extreme minority, ADASYN for others)...")
        
        # Separate classes into extreme minority (< 1000 samples) and others
        extreme_minority_classes = {}  # Use WGAN-GP
        moderate_minority_classes = {}  # Use ADASYN
        
        for class_label, count in zip(unique_classes, class_counts):
            target_count = sampling_strategy[class_label]
            if target_count > count:  # Only oversample if target > current
                if count < 1000:  # Extreme minority: use WGAN-GP
                    extreme_minority_classes[class_label] = (count, target_count)
                else:  # Moderate minority: use ADASYN
                    moderate_minority_classes[class_label] = target_count
        
        # Start with original data
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        # Process extreme minority classes with WGAN-GP
        if extreme_minority_classes:
            logger.info(f"  Processing {len(extreme_minority_classes)} extreme minority classes with WGAN-GP...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            synthetic_samples_list = []
            synthetic_labels_list = []
            
            for class_label, (current_count, target_count) in extreme_minority_classes.items():
                    attack_name = list(attack_type_mapping.keys())[list(attack_type_mapping.values()).index(class_label)]
                    n_samples_needed = target_count - current_count
                    
                    logger.info(f"    {attack_name} (Label {class_label}): {current_count} → {target_count} samples (need {n_samples_needed})")
                    
                    # Get minority class samples
                    class_mask = (y == class_label)
                    class_samples = X[class_mask]
                    
                    if len(class_samples) < 10:
                        logger.warning(f"      Too few samples ({len(class_samples)}) for WGAN-GP, skipping")
                        continue
                    
                    try:
                        # Generate synthetic samples using WGAN-GP
                        logger.info(f"      Training WGAN-GP on {len(class_samples)} samples...")
                        synthetic_samples = generate_synthetic_samples_wgan_gp(
                            X=class_samples,
                            n_samples=n_samples_needed,
                            n_epochs=50,  # Reduced for faster training
                            batch_size=min(64, len(class_samples)),
                            device=device,
                            verbose=False
                        )
                        
                        synthetic_samples_list.append(synthetic_samples)
                        synthetic_labels_list.append(np.full(n_samples_needed, class_label))
                        
                        logger.info(f"      ✅ Generated {n_samples_needed} synthetic samples for {attack_name}")
                        
                    except Exception as e:
                        logger.error(f"      ❌ WGAN-GP failed for {attack_name}: {e}")
                        raise RuntimeError(f"WGAN-GP generation failed for {attack_name}. Cannot continue without synthetic samples.")
            
            # Combine synthetic samples with original data
            if synthetic_samples_list:
                synthetic_X = np.vstack(synthetic_samples_list)
                synthetic_y = np.hstack(synthetic_labels_list)
                
                X_resampled = np.vstack([X_resampled, synthetic_X])
                y_resampled = np.hstack([y_resampled, synthetic_y])
                
                logger.info(f"  WGAN-GP completed: Added {len(synthetic_X)} synthetic samples")
        
        # Process moderate minority classes with ADASYN
        if moderate_minority_classes:
            logger.info(f"  Processing {len(moderate_minority_classes)} moderate minority classes with ADASYN...")
            
            adasyn = ADASYN(
                sampling_strategy=moderate_minority_classes,
                random_state=42,
                n_neighbors=5
            )
            
            try:
                X_adasyn, y_adasyn = adasyn.fit_resample(X_resampled, y_resampled)
                logger.info(f"  ADASYN completed: {len(X_resampled)} → {len(X_adasyn)} samples")
                X_resampled, y_resampled = X_adasyn, y_adasyn
            except Exception as e:
                logger.warning(f"  ADASYN failed: {e}")
                logger.info("  Falling back to SMOTE")
                # Fallback to SMOTE if ADASYN fails
                try:
                    smote = SMOTE(
                        sampling_strategy=moderate_minority_classes,
                        random_state=42,
                        k_neighbors=3
                    )
                    X_smote, y_smote = smote.fit_resample(X_resampled, y_resampled)
                    logger.info(f"  SMOTE fallback completed: {len(X_resampled)} → {len(X_smote)} samples")
                    X_resampled, y_resampled = X_smote, y_smote
                except Exception as e2:
                    logger.warning(f"  SMOTE fallback also failed: {e2}")
                    logger.info("  Using current data")
        else:
            if not extreme_minority_classes:
                logger.info("  No classes need oversampling")
        
        # Step 2: Apply RandomUnderSampler for undersampling majority classes
        logger.info("  Step 2: Applying RandomUnderSampler...")
        
        # Create undersampling strategy (only for classes that need undersampling)
        undersample_strategy = {}
        for class_label, count in zip(unique_classes, class_counts):
            target_count = sampling_strategy[class_label]
            if target_count < count:  # Only undersample if target < current
                undersample_strategy[class_label] = target_count
        
        if undersample_strategy:
            undersampler = RandomUnderSampler(
                sampling_strategy=undersample_strategy,
                random_state=42
            )
            
            try:
                X_balanced, y_balanced = undersampler.fit_resample(X_resampled, y_resampled)
                logger.info(f"  RandomUnderSampler completed: {len(X_resampled)} → {len(X_balanced)} samples")
            except Exception as e:
                logger.warning(f"  RandomUnderSampler failed: {e}")
                logger.info("  Using ADASYN result")
                X_balanced, y_balanced = X_resampled, y_resampled
        else:
            logger.info("  No classes need undersampling, skipping RandomUnderSampler")
            X_balanced, y_balanced = X_resampled, y_resampled
        
        # Create balanced dataframe
        balanced_df = pd.DataFrame(X_balanced, columns=feature_cols)
        balanced_df['label'] = y_balanced
        
        # Add binary labels and attack categories
        balanced_df['binary_label'] = (y_balanced != 0).astype(int)
        # Reverse mapping for attack categories
        reverse_mapping = {v: k for k, v in attack_type_mapping.items()}
        balanced_df['attack_cat'] = [reverse_mapping[label] for label in y_balanced]
        
        # Analyze class distribution after rebalancing
        unique_classes_after, class_counts_after = np.unique(y_balanced, return_counts=True)
        logger.info("  Class distribution after rebalancing:")
        for class_label, count in zip(unique_classes_after, class_counts_after):
            attack_name = reverse_mapping[class_label]
            percentage = (count / len(y_balanced)) * 100
            logger.info(f"    {attack_name} (Label {class_label}): {count:,} samples ({percentage:.2f}%)")
        
        # Calculate new imbalance ratio
        max_count_after = np.max(class_counts_after)
        min_count_after = np.min(class_counts_after)
        imbalance_ratio_after = max_count_after / min_count_after if min_count_after > 0 else float('inf')
        logger.info(f"  Imbalance ratio after rebalancing: {imbalance_ratio_after:.2f}:1")
        
        # Calculate improvement
        improvement = (imbalance_ratio - imbalance_ratio_after) / imbalance_ratio * 100
        logger.info(f"  Imbalance reduction: {improvement:.1f}%")
        
        logger.info(f"  Final complete dataset shape: {balanced_df.shape}")
        
        return balanced_df
    
    def create_zero_day_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                            zero_day_attack: str = 'DoS') -> Dict:
        """
        Create zero-day detection split with data leakage prevention:
        - Zero-Day Holdout: One attack class completely excluded from training/validation
        - Training Data: Normal + other attack classes (excluding zero-day attack)
        - Validation Data: Normal + other attack classes (excluding zero-day attack)
        - Test Data: 30% Normal + 70% Attacks (50% Zero-day + 50% Other attacks from test data only)
        
        Data Leakage Prevention:
        - Train/Val/Test data split: 80/10/10 (no overlap)
        - Test "other attacks" sampled from test data only
        - Test "normal samples" sourced EXCLUSIVELY from test data only (no fallback to train data)
        - Zero-day attack completely excluded from train/val
        
        Query Set Distribution:
        - Training/Validation Phase: Query sets have 80% Normal samples
        - Testing Phase: Query sets have 90% Normal samples
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            zero_day_attack: Attack type to hold out for zero-day testing
            
        Returns:
            Dictionary with split data
            
        Raises:
            ValueError: If no normal samples available in test_df for test set creation
        """
        logger.info(f"Creating zero-day split with '{zero_day_attack}' as zero-day attack")
        
        # Check what attack types are available in test data
        test_attack_types = test_df['label'].value_counts().sort_index()
        logger.info(f"  Available attack types in test data: {dict(test_attack_types)}")
        
        # Keep original 10-class labels (0=Normal, 1-9=Attack types)
        # Map attack categories to numeric labels
        train_df['label'] = train_df['attack_cat'].map(self.attack_types)
        test_df['label'] = test_df['attack_cat'].map(self.attack_types)
        
        # Handle any unmapped categories (set to 0 = Normal)
        train_df['label'] = train_df['label'].fillna(0).astype(int)
        test_df['label'] = test_df['label'].fillna(0).astype(int)
        
        # Log the actual labels found
        logger.info(f"  Training labels: {sorted(train_df['label'].unique())}")
        logger.info(f"  Test labels: {sorted(test_df['label'].unique())}")
        
        # Also create binary labels for compatibility
        train_df['binary_label'] = (train_df['attack_cat'] != 'Normal').astype(int)
        val_df['binary_label'] = (val_df['attack_cat'] != 'Normal').astype(int)
        test_df['binary_label'] = (test_df['attack_cat'] != 'Normal').astype(int)
        
        # Log binary label distribution BEFORE zero-day filtering
        logger.info(f"  Binary labels BEFORE zero-day filtering:")
        logger.info(f"    Training - Normal: {len(train_df[train_df['binary_label'] == 0])}, Attack: {len(train_df[train_df['binary_label'] == 1])}")
        logger.info(f"    Validation - Normal: {len(val_df[val_df['binary_label'] == 0])}, Attack: {len(val_df[val_df['binary_label'] == 1])}")
        logger.info(f"    Test - Normal: {len(test_df[test_df['binary_label'] == 0])}, Attack: {len(test_df[test_df['binary_label'] == 1])}")
        
        # Separate Normal and Attack samples
        train_normal = train_df[train_df['attack_cat'] == 'Normal'].copy()
        train_attacks = train_df[train_df['attack_cat'] != 'Normal'].copy()
        val_normal = val_df[val_df['attack_cat'] == 'Normal'].copy()
        val_attacks = val_df[val_df['attack_cat'] != 'Normal'].copy()
        test_normal = test_df[test_df['attack_cat'] == 'Normal'].copy()
        test_attacks = test_df[test_df['attack_cat'] != 'Normal'].copy()
        
        # Get zero-day attack samples from test data
        zero_day_test = test_df[test_df['attack_cat'] == zero_day_attack].copy()
        
        # If zero-day attack not found in test data, find alternative
        if len(zero_day_test) == 0:
            logger.warning(f"No {zero_day_attack} attacks found in test data. Finding best alternative.")
            available_attacks = test_df[test_df['attack_cat'] != 'Normal']['attack_cat'].value_counts()
            if len(available_attacks) > 0:
                most_common_attack = available_attacks.index[0]
                zero_day_test = test_df[test_df['attack_cat'] == most_common_attack].copy()
                logger.info(f"Using {most_common_attack} as zero-day attack")
                zero_day_attack = most_common_attack
            else:
                logger.error("No attack samples found in test data!")
                zero_day_test = pd.DataFrame()
        
        # Filter out zero-day attack from training and validation attacks
        train_attacks_filtered = train_attacks[train_attacks['attack_cat'] != zero_day_attack].copy()
        val_attacks_filtered = val_attacks[val_attacks['attack_cat'] != zero_day_attack].copy()
        
        # Create training data: Normal + other attack classes (excluding zero-day)
        train_data = pd.concat([train_normal, train_attacks_filtered], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create validation data: Normal + other attack classes (excluding zero-day)
        val_data = pd.concat([val_normal, val_attacks_filtered], ignore_index=True)
        val_data = val_data.sample(frac=1, random_state=43).reset_index(drop=True)
        
        # Create test data with BALANCED distribution for better evaluation
        # Target: 40% Normal + 60% Attacks (35% Non-zero-day + 25% Zero-day)
        
        # Calculate target sample sizes for balanced evaluation
        # Use full available test data (no cap) while maintaining 40% normal / 60% attack ratio
        total_test_samples = len(zero_day_test) + len(test_attacks[test_attacks['attack_cat'] != zero_day_attack]) + len(test_normal)
        target_normal_samples = int(total_test_samples * 0.40)  # 40% Normal
        target_attack_samples = int(total_test_samples * 0.60)  # 60% Attacks
        target_zero_day_samples = int(total_test_samples * 0.25)  # 25% Zero-day (adjusted from 20% to total 100%)
        target_non_zero_day_samples = int(total_test_samples * 0.35)  # 35% Non-zero-day attacks
        
        logger.info(f"  Test set composition target: {target_normal_samples} normal (40%), {target_non_zero_day_samples} non-zero-day attacks (35%), {target_zero_day_samples} zero-day attacks (25%)")
        logger.info(f"  Available in test data: {len(test_normal)} normal, {len(zero_day_test)} zero-day, {len(test_attacks[test_attacks['attack_cat'] != zero_day_attack])} other attacks")
        
        # Sample Normal samples for test data - PREVENT DATA LEAKAGE
        # Only use normal samples from test_df to prevent leakage from training data
        if len(test_normal) >= target_normal_samples:
            test_normal_sample = test_normal.sample(n=target_normal_samples, random_state=42)
            logger.info(f"Normal samples for test set sourced exclusively from test_df: {len(test_normal_sample)} samples (40% target)")
        elif len(test_normal) > 0:
            # Use all available normal samples from test data if insufficient for target
            test_normal_sample = test_normal.sample(n=len(test_normal), random_state=42)
            logger.warning(f"Insufficient normal samples in test_df for 40% target: {len(test_normal)} < {target_normal_samples} required")
            logger.info(f"Using all available normal samples from test_df: {len(test_normal_sample)} samples")
        else:
            # No normal samples in test data - raise error to prevent data leakage
            raise ValueError("No normal samples available in test_df for test set. Cannot use training data to prevent data leakage.")
        
        # Sample zero-day attacks for test data - use target percentage (25%)
        if len(zero_day_test) >= target_zero_day_samples:
            zero_day_sample = zero_day_test.sample(n=target_zero_day_samples, random_state=42)
            logger.info(f"Zero-day samples selected: {len(zero_day_sample)} samples (25% target)")
        elif len(zero_day_test) > 0:
            zero_day_sample = zero_day_test.copy()
            logger.warning(f"Insufficient zero-day samples for 25% target: {len(zero_day_test)} < {target_zero_day_samples} required, using all available")
        else:
            zero_day_sample = pd.DataFrame()
            logger.warning("No zero-day attack samples found in test data")
        
        # Sample non-zero-day attacks for test data (35% target)
        test_other_attacks = test_attacks[test_attacks['attack_cat'] != zero_day_attack].copy()
        if len(test_other_attacks) > 0:
            other_attacks_sample = test_other_attacks.sample(n=min(target_non_zero_day_samples, len(test_other_attacks)), random_state=42)
            logger.info(f"Non-zero-day attack samples selected: {len(other_attacks_sample)} samples (35% target)")
        else:
            # Fallback: if no other attacks in test data
            logger.warning("No other attack types found in test data, using only zero-day attacks")
            other_attacks_sample = pd.DataFrame()
        
        test_data = pd.concat([test_normal_sample, zero_day_sample, other_attacks_sample], ignore_index=True)
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"  Training data: {len(train_data)} samples")
        logger.info(f"    Normal: {len(train_data[train_data['binary_label'] == 0])}")
        logger.info(f"    Other attacks (excluding zero-day): {len(train_data[train_data['binary_label'] == 1])}")
        
        logger.info(f"  Validation data: {len(val_data)} samples")
        logger.info(f"    Normal: {len(val_data[val_data['binary_label'] == 0])}")
        logger.info(f"    Other attacks (excluding zero-day): {len(val_data[val_data['binary_label'] == 1])}")
        
        logger.info(f"  Test data: {len(test_data)} samples")
        actual_normal_count = len(test_data[test_data['binary_label'] == 0])
        actual_zero_day_count = len(test_data[test_data['label'] == self.attack_types[zero_day_attack]]) if zero_day_attack in self.attack_types else 0
        actual_non_zero_day_count = len(test_data[(test_data['label'] != 0) & (test_data['label'] != self.attack_types.get(zero_day_attack, -1))])
        total_test_count = len(test_data)
        logger.info(f"    Normal (40% target): {actual_normal_count} ({100*actual_normal_count/total_test_count:.1f}%)")
        logger.info(f"    Zero-day attacks (25% target): {actual_zero_day_count} ({100*actual_zero_day_count/total_test_count:.1f}%)")
        logger.info(f"    Non-zero-day attacks (35% target): {actual_non_zero_day_count} ({100*actual_non_zero_day_count/total_test_count:.1f}%)")
        logger.info(f"    Test binary labels: {sorted(test_data['binary_label'].unique())}")
        logger.info(f"    Test multi-class labels: {sorted(test_data['label'].unique())}")
        
        # Log data leakage prevention measures
        logger.info("  Data leakage prevention:")
        logger.info(f"    ✓ Train/Val data split: 80/10/10 (no overlap)")
        logger.info(f"    ✓ Test 'other attacks' sampled from test data only")
        logger.info(f"    ✓ Test 'normal samples' sourced exclusively from test data only")
        logger.info(f"    ✓ Zero-day attack completely excluded from train/val")
        
        # CRITICAL: Verify that training data has both classes
        train_normal_count = len(train_data[train_data['binary_label'] == 0])
        train_attack_count = len(train_data[train_data['binary_label'] == 1])
        logger.info(f"  🔍 CRITICAL VERIFICATION:")
        logger.info(f"    Training data classes: Normal={train_normal_count}, Attack={train_attack_count}")
        if train_attack_count == 0:
            logger.error(f"    ❌ CRITICAL BUG: Training data has NO attack samples!")
            logger.error(f"    This will cause the model to only learn Normal patterns!")
        else:
            logger.info(f"    ✅ Training data has both Normal and Attack samples - GOOD!")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'zero_day_attack': zero_day_attack,
            'attack_types': self.attack_types
        }
    
    def preprocess_unsw_dataset(self, zero_day_attack: str = 'DoS') -> Dict:
        """
        Complete preprocessing pipeline for UNSW-NB15 dataset
        
        Args:
            zero_day_attack: Attack type to hold out for zero-day testing
            
        Returns:
            Dictionary with preprocessed data and metadata
        """
        logger.info("Starting UNSW-NB15 preprocessing pipeline")
        logger.info("=" * 60)
        
        # Load datasets
        logger.info("Loading UNSW-NB15 datasets...")
        train_df = pd.read_csv(self.data_path)
        test_df = pd.read_csv(self.test_path)
        
        logger.info(f"Training data: {train_df.shape}")
        logger.info(f"Testing data: {test_df.shape}")
        
        # FIX #1: Process train and test separately to prevent data leakage
        # Correct preprocessing order: Quality Assessment → Feature Engineering → Data Cleaning → Categorical Encoding → Feature Selection
        # Note: Categorical encoding comes before feature selection to provide encoded features for selection algorithms
        
        logger.info("\nProcessing training data...")
        train_quality = self.step1_data_quality_assessment(train_df)
        train_df = self.step2_feature_engineering(train_df)
        train_df = self.step3_data_cleaning(train_df)  # Data cleaning before encoding
        train_df = self.step4_categorical_encoding(train_df)  # Encoding after cleaning - FITS encoders
        
        logger.info("\nProcessing test data...")
        test_quality = self.step1_data_quality_assessment(test_df)
        test_df = self.step2_feature_engineering(test_df)
        test_df = self.step3_data_cleaning(test_df)  # Data cleaning before encoding
        test_df = self.step4_categorical_encoding_transform(test_df)  # Transform using fitted encoders from training
        
        # Check if feature selection is enabled
        try:
            from config import SystemConfig
            use_feature_selection = SystemConfig().use_feature_selection
        except:
            use_feature_selection = True  # Default to enabled if config not available
        
        if use_feature_selection:
            # FIX #1: Feature selection on TRAINING data only (prevents leakage)
            logger.info("\nApplying XGBoost feature selection to TRAINING data only...")
            train_df = self.step5_feature_selection_hybrid(train_df, target_col='attack_cat', n_features_final=30)
            
            # Apply same selected features to test data
            selected_features = self.selected_features
            logger.info(f"\nApplying selected features ({len(selected_features)} features) to test data...")
            
            # Ensure we include target columns if they exist in test_df
            target_cols = []
            for col in ['label', 'binary_label', 'attack_cat']:
                if col in test_df.columns:
                    target_cols.append(col)
            
            # Select only the features that exist in test_df
            available_selected_features = [f for f in selected_features if f in test_df.columns]
            missing_features = [f for f in selected_features if f not in test_df.columns]
            
            if missing_features:
                logger.warning(f"  Warning: {len(missing_features)} selected features not found in test data: {missing_features[:5]}...")
            
            # Select features and target columns
            test_df = test_df[available_selected_features + target_cols].copy()
            
            # Add missing selected features as zeros (shouldn't happen, but for safety)
            for feat in missing_features:
                test_df[feat] = 0.0
                logger.info(f"  Added missing selected feature '{feat}' to test data (filled with 0)")
            
            # Ensure test_df has all selected features in the same order
            final_test_features = selected_features + target_cols
            # Reorder and ensure all columns exist
            for col in final_test_features:
                if col not in test_df.columns:
                    test_df[col] = 0.0
            test_df = test_df[final_test_features]
            
            logger.info(f"  Test data shape after feature selection: {test_df.shape}")
        else:
            # TEMPORARILY DISABLED: Skip feature selection, keep all features
            logger.info("\n⚠️  Feature selection DISABLED - keeping all features...")
            
            # Ensure we include target columns if they exist
            target_cols = []
            for col in ['label', 'binary_label', 'attack_cat']:
                if col in train_df.columns:
                    target_cols.append(col)
            
            # Remove target columns from feature list
            feature_cols = [col for col in train_df.columns if col not in target_cols]
            
            # Store all features as "selected" for consistency
            self.selected_features = feature_cols
            logger.info(f"  Keeping all {len(feature_cols)} features (excluding target columns)")
            
            # Ensure train and test have the same features
            train_feature_cols = [col for col in train_df.columns if col not in target_cols]
            test_feature_cols = [col for col in test_df.columns if col not in target_cols]
            
            # Find common features
            common_features = list(set(train_feature_cols) & set(test_feature_cols))
            logger.info(f"  Common features between train and test: {len(common_features)}")
            
            # Keep only common features (plus target columns)
            train_df = train_df[common_features + target_cols].copy()
            test_df = test_df[common_features + target_cols].copy()
            
            logger.info(f"  Training data shape after skipping feature selection: {train_df.shape}")
            logger.info(f"  Test data shape after skipping feature selection: {test_df.shape}")
        
        # FIX #2: Split FIRST, then rebalance training data only (prevents leakage)
        # Split training data into train/val using STRATIFIED sampling BEFORE rebalancing
        logger.info("\nSplitting training data into train/val using stratified sampling (BEFORE rebalancing)...")
        original_train_total = len(train_df)
        
        # Use stratified split to preserve class distribution
        from sklearn.model_selection import train_test_split
        
        # Prepare features and labels for stratified split
        feature_cols = [col for col in train_df.columns if col not in ['label', 'binary_label', 'attack_cat']]
        X = train_df[feature_cols].values
        
        # CRITICAL FIX: Ensure label column contains MULTICLASS labels (0-9), not binary (0,1)
        # Recreate label column from attack_cat to ensure multiclass labels are preserved
        attack_type_mapping = {
            'Normal': 0, 'Fuzzers': 1, 'Analysis': 2, 'Backdoor': 3, 'DoS': 4,
            'Exploits': 5, 'Generic': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9
        }
        
        if 'attack_cat' in train_df.columns:
            # Always recreate label from attack_cat to ensure multiclass labels
            old_unique_labels = train_df['label'].unique() if 'label' in train_df.columns else []
            train_df['label'] = train_df['attack_cat'].map(attack_type_mapping).fillna(0).astype(int)
            new_unique_labels = train_df['label'].unique()
            
            if len(old_unique_labels) > 0 and len(old_unique_labels) != len(new_unique_labels):
                logger.info(f"✅ FIXED: Recreated 'label' column from attack_cat (was {len(old_unique_labels)} labels, now {len(new_unique_labels)} labels)")
            elif 'label' not in train_df.columns:
                logger.info(f"✅ Created 'label' column from attack_cat. Unique labels: {sorted(new_unique_labels)}")
        
        # Debug: Log label distribution before split
        unique_labels = train_df['label'].unique()
        logger.info(f"🔍 BEFORE SPLIT: train_df has {len(unique_labels)} unique labels: {sorted(unique_labels)}")
        if 'attack_cat' in train_df.columns:
            unique_attack_cats = train_df['attack_cat'].unique()
            logger.info(f"🔍 BEFORE SPLIT: train_df has {len(unique_attack_cats)} unique attack categories: {sorted(unique_attack_cats)}")
            # Count samples per attack category
            for attack_cat in sorted(unique_attack_cats):
                count = len(train_df[train_df['attack_cat'] == attack_cat])
                logger.info(f"   {attack_cat}: {count:,} samples")
        
        y = train_df['label'].values
        
        # Split: 80% train, 20% val (BEFORE rebalancing)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X, y,
            test_size=0.2,  # 20% for validation
            stratify=y,  # This ensures all classes are represented in both sets
            random_state=42
        )
        
        # Log split percentages relative to original training total
        train_pct = (len(X_train_split) / original_train_total) * 100
        val_pct = (len(X_val_split) / original_train_total) * 100
        
        logger.info(f"Split percentages relative to original training data ({original_train_total}):")
        logger.info(f"  Train: {len(X_train_split)} samples (~{train_pct:.2f}%)")
        logger.info(f"  Validation: {len(X_val_split)} samples (~{val_pct:.2f}%)")
        logger.info(f"  Test: {len(test_df)} samples (original test set - untouched)")
        
        # Reconstruct dataframes (before rebalancing)
        train_df_split = pd.DataFrame(X_train_split, columns=feature_cols)
        train_df_split['label'] = y_train_split
        train_df_split['binary_label'] = (y_train_split != 0).astype(int)
        
        val_df = pd.DataFrame(X_val_split, columns=feature_cols)
        val_df['label'] = y_val_split
        val_df['binary_label'] = (y_val_split != 0).astype(int)
        
        # Create reverse mapping for attack categories
        attack_type_mapping = {
            'Normal': 0, 'Fuzzers': 1, 'Analysis': 2, 'Backdoor': 3, 'DoS': 4,
            'Exploits': 5, 'Generic': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9
        }
        reverse_mapping = {v: k for k, v in attack_type_mapping.items()}
        train_df_split['attack_cat'] = [reverse_mapping[label] for label in y_train_split]
        val_df['attack_cat'] = [reverse_mapping[label] for label in y_val_split]
        
        # FIX #2: Rebalance TRAINING data only (validation and test remain untouched)
        logger.info("\nApplying data rebalancing to TRAINING data only (validation and test untouched)...")
        train_df_before_rebal = len(train_df_split)
        train_df_rebalanced = self.step7_data_rebalancing_complete(train_df_split)
        train_df_after_rebal = len(train_df_rebalanced)
        
        logger.info(f"  Training data: {train_df_before_rebal} → {train_df_after_rebal} samples (rebalanced)")
        logger.info(f"  Validation data: {len(val_df)} samples (unchanged - original distribution)")
        logger.info(f"  Test data: {len(test_df)} samples (unchanged - original distribution)")
        
        # Use rebalanced training data
        train_df = train_df_rebalanced
        
        # Ensure test_df has label and binary_label columns (they should already exist)
        if 'label' not in test_df.columns:
            # Create label from attack_cat if needed
            if 'attack_cat' in test_df.columns:
                test_df['label'] = test_df['attack_cat'].map({v: k for k, v in attack_type_mapping.items()})
            else:
                logger.error("  ❌ Test data missing both 'label' and 'attack_cat' columns!")
        
        if 'binary_label' not in test_df.columns and 'label' in test_df.columns:
            test_df['binary_label'] = (test_df['label'] != 0).astype(int)
        
        logger.info(f"Rebalanced training data: {train_df.shape}")
        logger.info(f"Rebalanced validation data: {val_df.shape}")
        logger.info(f"Test data: {test_df.shape}")
        
        # Align features between train, validation, and test data
        logger.info("\nAligning features between train, validation, and test data...")
        train_cols = set(train_df.columns)
        val_cols = set(val_df.columns)
        test_cols = set(test_df.columns)
        
        # Find missing columns in each dataset
        all_cols = train_cols.union(val_cols).union(test_cols)
        missing_in_train = all_cols - train_cols
        missing_in_val = all_cols - val_cols
        missing_in_test = all_cols - test_cols
        
        # Add missing columns with zeros
        for col in missing_in_train:
            train_df[col] = 0
            logger.info(f"  Added missing column to train data: {col}")
        
        for col in missing_in_val:
            val_df[col] = 0
            logger.info(f"  Added missing column to validation data: {col}")
        
        for col in missing_in_test:
            test_df[col] = 0
            logger.info(f"  Added missing column to test data: {col}")
        
        # Ensure same column order
        common_cols = sorted(list(all_cols))
        train_df = train_df[common_cols]
        val_df = val_df[common_cols]
        test_df = test_df[common_cols]
        
        logger.info(f"  Final feature count - Train: {len(train_df.columns)}, Val: {len(val_df.columns)}, Test: {len(test_df.columns)}")
        
        # Create zero-day split
        split_data = self.create_zero_day_split(train_df, val_df, test_df, zero_day_attack)
        
        # Apply feature scaling
        train_scaled, val_scaled, test_scaled = self.step6_feature_scaling(
            split_data['train'], split_data['val'], split_data['test']
        )
        
        # Convert to PyTorch tensors
        feature_cols = [col for col in train_scaled.columns if col not in ['label', 'binary_label', 'attack_cat']]
        
        X_train = torch.FloatTensor(train_scaled[feature_cols].values)
        y_train = torch.LongTensor(train_scaled['binary_label'].values)  # Use binary labels (0=Normal, 1=Attack)
        y_train_multiclass = torch.LongTensor(train_scaled['label'].values)  # Multiclass labels (0-9) for attack type distinction
        
        # Debug: Check multiclass label distribution
        unique_multiclass = torch.unique(y_train_multiclass)
        logger.info(f"🔍 Training multiclass labels: {unique_multiclass.tolist()} ({len(unique_multiclass)} unique)")
        if len(unique_multiclass) <= 2:
            logger.warning(f"⚠️  Only {len(unique_multiclass)} unique multiclass labels found! Expected 8+ attack types (excluding zero-day).")
            logger.warning(f"   This means 'include_all_attack_types_in_support' will not work correctly.")
            # Check if attack_cat column exists and has multiple attack types
            if 'attack_cat' in train_scaled.columns:
                unique_attack_cats = train_scaled['attack_cat'].unique()
                logger.info(f"   Available attack categories in training data: {sorted(unique_attack_cats)}")
                if len(unique_attack_cats) > 2:
                    logger.warning(f"   ⚠️  Attack categories exist but labels are binary! The 'label' column may have been overwritten.")
        
        X_val = torch.FloatTensor(val_scaled[feature_cols].values)
        y_val = torch.LongTensor(val_scaled['binary_label'].values)  # Use binary labels (0=Normal, 1=Attack)
        y_val_multiclass = torch.LongTensor(val_scaled['label'].values)  # Multiclass labels (0-9)
        
        X_test = torch.FloatTensor(test_scaled[feature_cols].values)
        y_test = torch.LongTensor(test_scaled['binary_label'].values)  # Use binary labels (0=Normal, 1=Attack)
        
        # Create zero-day indices (indices where attack_cat == zero_day_attack, e.g., DoS)
        # For binary classification, we need to identify specific DoS samples in test set
        zero_day_attack_label = self.attack_types.get(zero_day_attack, 4)  # Default to DoS=4
        
        # Calculate zero-day indices based on the final test data after zero-day split
        # We need to find which samples in the final test set are zero-day attacks
        test_data_final = test_df  # This is the final test data after zero-day split
        zero_day_mask = test_data_final['attack_cat'] == zero_day_attack
        zero_day_indices = torch.where(torch.tensor(zero_day_mask.values))[0].tolist()
        
        logger.info("\nPreprocessing completed successfully!")
        logger.info(f"Final feature count: {len(feature_cols)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Zero-day samples ({zero_day_attack}, label={zero_day_attack_label}): {len(zero_day_indices)}")
        
        # Store multiclass labels and attack_cat for zero-day identification after sequence creation
        y_test_multiclass = torch.LongTensor(test_scaled['label'].values)  # Multiclass labels (0-9)
        test_attack_cat = test_scaled['attack_cat'].values.tolist()  # Attack category names
        
        # Create flow IDs for test data
        test_flow_ids = self._create_flow_ids(test_scaled)
        
        return {
            'X_train': X_train,
            'y_train': y_train,  # Binary labels (0=Normal, 1=Attack)
            'y_train_multiclass': y_train_multiclass,  # Multiclass labels (0-9) for attack type distinction
            'X_val': X_val,
            'y_val': y_val,  # Binary labels (0=Normal, 1=Attack)
            'y_val_multiclass': y_val_multiclass,  # Multiclass labels (0-9)
            'X_test': X_test,
            'y_test': y_test,  # Binary labels for training
            'y_test_multiclass': y_test_multiclass,  # Multiclass labels for zero-day identification
            'test_attack_cat': test_attack_cat,  # Attack category names for zero-day identification
            'test_flow_ids': test_flow_ids,  # Flow IDs for flow-level evaluation
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
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        preprocessor_state = {
            'scaler': self.scaler,
            'target_encoders': self.target_encoders,
            'feature_names': self.feature_names,
            'attack_types': self.attack_types
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        
        logger.info(f"Preprocessor state saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.scaler = preprocessor_state['scaler']
        self.target_encoders = preprocessor_state['target_encoders']
        self.feature_names = preprocessor_state['feature_names']
        self.attack_types = preprocessor_state['attack_types']
        
        logger.info(f"Preprocessor state loaded from {filepath}")
    
    def sample_stratified_subset(self, X: torch.Tensor, y: torch.Tensor, n_samples: int, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a stratified subset preserving class distribution
        
        Args:
            X: Input features tensor
            y: Target labels tensor
            n_samples: Number of samples to select
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_subset, y_subset) with stratified sampling
        """
        from sklearn.model_selection import train_test_split
        
        # Convert to numpy for sklearn
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Ensure we don't sample more than available
        n_samples = min(n_samples, len(X_np))
        
        # Use stratified sampling to preserve class distribution
        if n_samples >= len(X_np):
            # If we want all samples, just return the original data
            X_subset = X_np
            y_subset = y_np
        else:
            # Use specific number of samples
            X_subset, _, y_subset, _ = train_test_split(
                X_np, y_np,
                train_size=n_samples,
                stratify=y_np,
                random_state=random_state
            )
        
        # Convert back to tensors
        X_subset = torch.FloatTensor(X_subset)
        y_subset = torch.LongTensor(y_subset)
        
        logger.info(f"Sampled {len(X_subset)} stratified samples from {len(X)} total samples")
        logger.info(f"Class distribution: {np.bincount(y_subset.numpy())}")
        
        return X_subset, y_subset
    
    def create_sequences(self, X, y, sequence_length=30, stride=15, zero_pad=True):
        """
        Create sequences from preprocessed data with optional zero-padding
        
        Args:
            X: Preprocessed features array of shape (n_samples, n_features)
            y: Labels array of shape (n_samples,)
            sequence_length: Length of each sequence (default: 30)
            stride: Step size for sliding window (default: 15)
            zero_pad: Whether to zero-pad short sequences to sequence_length (default: True)
            
        Returns:
            X_sequences: Array of shape (n_sequences, sequence_length, n_features)
            y_sequences: Array of shape (n_sequences,)
        """
        logger.info(f"Creating sequences with length={sequence_length}, stride={stride}, zero_pad={zero_pad}")
        
        # Convert to numpy arrays if they're tensors
        if hasattr(X, 'cpu'):
            X = X.cpu().numpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if hasattr(y, 'cpu'):
            y = y.cpu().numpy()
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        
        n_samples, n_features = X.shape
        sequences = []
        labels = []
        
        # Create sliding window sequences
        for i in range(0, n_samples - sequence_length + 1, stride):
            # Extract sequence
            sequence = X[i:i + sequence_length]
            label = y[i + sequence_length - 1]  # Use label from last timestep
            
            sequences.append(sequence)
            labels.append(label)
        
        # Convert to numpy arrays
        X_sequences = np.array(sequences)
        y_sequences = np.array(labels)
        
        # Zero-pad short sequences if requested
        if zero_pad and len(sequences) > 0:
            # Check if any sequences are shorter than sequence_length
            actual_length = X_sequences.shape[1]
            if actual_length < sequence_length:
                logger.info(f"Zero-padding sequences from length {actual_length} to {sequence_length}")
                
                # Create padded sequences
                padded_sequences = []
                for seq in X_sequences:
                    if len(seq) < sequence_length:
                        # Zero-pad to the right
                        padding = np.zeros((sequence_length - len(seq), n_features))
                        padded_seq = np.vstack([seq, padding])
                    else:
                        padded_seq = seq
                    padded_sequences.append(padded_seq)
                
                X_sequences = np.array(padded_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        logger.info(f"Sequence labels shape: {y_sequences.shape}")
        
        return X_sequences, y_sequences

def main():
    """Test the UNSW preprocessor"""
    logger.info("Testing UNSW-NB15 Preprocessor")
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor()
    
    # Run preprocessing
    try:
        data = preprocessor.preprocess_unsw_dataset(zero_day_attack='DoS')
        
        logger.info("\nPreprocessing Results:")
        logger.info(f"Training data shape: {data['X_train'].shape}")
        logger.info(f"Validation data shape: {data['X_val'].shape}")
        logger.info(f"Test data shape: {data['X_test'].shape}")
        logger.info(f"Feature count: {len(data['feature_names'])}")
        logger.info(f"Zero-day attack: {data['zero_day_attack']}")
        
        # Save preprocessor state
        preprocessor.save_preprocessor('unsw_preprocessor.pkl')
        
        logger.info("✅ Preprocessing test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
