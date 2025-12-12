#!/usr/bin/env python3
"""
Centralized NIDS - KDDTest+ Preprocessor
Customized for Zero-Day Attack Detection
"""
import pandas as pd
import numpy as np
import logging
import warnings
import re
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor  # Inherit basics

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KDDPreprocessor(UNSWPreprocessor):
    """
    Customized Preprocessor for KDDTest+ Dataset (NSL-KDD)
    """
    
    def __init__(self, data_path: str = "KDDTrain+.csv", test_path: str = "KDDTest+.csv"):
        super().__init__(data_path, test_path)
        
        # KDDTest+ Attack Types (NSL-KDD)
        # Based on actual labels found in dataset
        self.attack_types = {
            'normal': 0,
            # DoS attacks
            'back': 1,
            'land': 2,
            'neptune': 3,
            'pod': 4,
            'smurf': 5,
            'teardrop': 6,
            # Probe attacks
            'ipsweep': 7,
            'nmap': 8,
            'portsweep': 9,
            'satan': 10,
            # R2L attacks
            'guess_passwd': 11,
            'ftp_write': 12,
            'imap': 13,
            'multihop': 14,
            'phf': 15,
            'spy': 16,
            'warezclient': 17,
            'warezmaster': 18,
            # U2R attacks
            'buffer_overflow': 19,
            'loadmodule': 20,
            'perl': 21,
            'rootkit': 22,
            # Additional attacks that may appear in test set
            'mailbomb': 23,
            'apache2': 24,
            'processtable': 25,
            'udpstorm': 26,
            'mscan': 27,
            'saint': 28,
            'xlock': 29,
            'xsnoop': 30,
            'snmpguess': 31,
            'snmpgetattack': 32,
            'httptunnel': 33,
            'sendmail': 34,
            'named': 35,
            'ps': 36,
            'sqlattack': 37,
            'xterm': 38,
            'worm': 39,
        }
        logger.info("KDDTest+ Preprocessor initialized")
    
    def load_and_clean_columns(self, path):
        """Load KDD CSV file"""
        logger.info(f"Loading KDD CSV file: {path}")
        try:
            # KDD files can be large, use chunking for memory efficiency
            chunk_list = []
            chunk_size = 100000
            
            for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                # Strip whitespace from column names
                chunk.columns = chunk.columns.str.strip()
                chunk_list.append(chunk)
            
            df = pd.concat(chunk_list, ignore_index=True)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            raise
    
    def step2_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        KDD dataset doesn't need feature engineering like UNSW
        (no sbytes/dbytes columns to create)
        """
        # KDD already has all features, just return as-is
        return df
    
    def step4_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override categorical encoding for KDD dataset
        KDD has: protocol_type, service, flag (not proto, service, state)
        """
        logger.info("Step 4: Categorical Encoding for KDD dataset")
        
        # KDD categorical columns
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Convert label to numeric first for target encoding
        if 'label' in df.columns:
            # Map label to numeric using attack_types
            df['label_numeric'] = df['label'].map(self.attack_types).fillna(-1).astype(int)
        else:
            df['label_numeric'] = 0
        
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                logger.info(f"  {col}: {unique_count} unique values")
                
                if unique_count > 10:  # High-cardinality: Target encoding
                    # Target encoding using numeric label
                    target_mean = df.groupby(col)['label_numeric'].mean()
                    df[f'{col}_target_encoded'] = df[col].map(target_mean)
                    self.target_encoders[col] = target_mean
                    logger.info(f"    Applied target encoding to {col}")
                    # Drop original column
                    df = df.drop(columns=[col])
                else:  # Low-cardinality: One-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col)
                    self.onehot_columns[col] = dummies.columns.tolist()
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    logger.info(f"    Applied one-hot encoding to {col} → {dummies.shape[1]} features")
        
        # Remove temporary label_numeric column
        if 'label_numeric' in df.columns:
            df = df.drop(columns=['label_numeric'])
        
        logger.info(f"  Final shape after encoding: {df.shape}")
        return df
    
    def step4_categorical_encoding_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted categorical encoders for KDD dataset
        """
        logger.info("Step 4 (Transform): Applying fitted categorical encoders to KDD test data")
        
        # KDD categorical columns
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                logger.info(f"  {col}: {unique_count} unique values")
                
                if unique_count > 10:  # High-cardinality: Use fitted target encoder
                    if col in self.target_encoders:
                        target_mean = self.target_encoders[col]
                        default_value = target_mean.mean() if len(target_mean) > 0 else 0.0
                        df[f'{col}_target_encoded'] = df[col].map(target_mean).fillna(default_value)
                        logger.info(f"    Applied fitted target encoding to {col}")
                    else:
                        # Fallback: frequency encoding
                        freq_encoding = df[col].value_counts() / len(df)
                        df[f'{col}_freq_encoded'] = df[col].map(freq_encoding)
                        logger.warning(f"    Warning: No fitted encoder for {col}, using frequency encoding")
                    df = df.drop(columns=[col])
                else:  # Low-cardinality: One-hot encoding
                    if col in self.onehot_columns:
                        expected_cols = self.onehot_columns[col]
                        dummies = pd.get_dummies(df[col], prefix=col)
                        # Add missing columns
                        for expected_col in expected_cols:
                            if expected_col not in dummies.columns:
                                dummies[expected_col] = 0
                        dummies = dummies[expected_cols]
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(columns=[col])
                        logger.info(f"    Applied one-hot encoding to {col} → {len(expected_cols)} features")
                    else:
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(columns=[col])
                        logger.warning(f"    Warning: No training one-hot columns for {col}, created {dummies.shape[1]} features")
        
        logger.info(f"  Final shape after encoding: {df.shape}")
        return df
    
    def preprocess_unsw_dataset(self, zero_day_attack: str = 'neptune', 
                                source_data_path: str = None, 
                                target_test_path: str = None) -> dict:
        """
        Complete preprocessing pipeline for KDDTest+ dataset
        
        Args:
            zero_day_attack: Attack type to treat as zero-day (e.g., 'neptune', 'portsweep')
            source_data_path: Optional source dataset path for cross-dataset evaluation (if None, uses self.data_path)
            target_test_path: Optional target test dataset path for cross-dataset evaluation (if None, uses self.test_path)
        
        Returns:
            Dictionary with preprocessed data and metadata
        """
        logger.info("🚀 Starting KDDTest+ preprocessing pipeline...")
        logger.info(f"🎯 Zero-day attack: {zero_day_attack}")
        
        # Determine dataset paths (support cross-dataset evaluation)
        train_path = source_data_path if source_data_path else self.data_path
        test_path = target_test_path if target_test_path else self.test_path
        
        if source_data_path or target_test_path:
            logger.info("📊 CROSS-DATASET EVALUATION MODE")
            logger.info(f"   Training on: {train_path}")
            logger.info(f"   Testing on: {test_path}")
        
        # Load datasets
        logger.info("Loading datasets...")
        train_df = self.load_and_clean_columns(train_path)
        test_df = self.load_and_clean_columns(test_path)
        
        logger.info(f"Training data: {train_df.shape}")
        logger.info(f"Testing data: {test_df.shape}")
        
        # Check label column
        label_column = 'label'
        if label_column not in train_df.columns:
            logger.error(f"Label column '{label_column}' not found in dataset!")
            raise ValueError(f"Label column '{label_column}' not found")
        
        # Normalize label values (lowercase, strip whitespace)
        train_df[label_column] = train_df[label_column].str.lower().str.strip()
        test_df[label_column] = test_df[label_column].str.lower().str.strip()
        
        # Log label distribution
        logger.info(f"📊 Training label distribution:")
        train_label_counts = train_df[label_column].value_counts()
        logger.info(f"   {train_label_counts.to_dict()}")
        
        logger.info(f"📊 Test label distribution:")
        test_label_counts = test_df[label_column].value_counts()
        logger.info(f"   {test_label_counts.to_dict()}")
        
        # Process training data
        logger.info("\n🔧 Processing training data...")
        train_df = self.step2_feature_engineering(train_df)
        train_df = self.step3_data_cleaning(train_df)
        train_df = self.step4_categorical_encoding(train_df)
        
        # Process test data
        logger.info("\n🔧 Processing test data...")
        test_df = self.step2_feature_engineering(test_df)
        test_df = self.step3_data_cleaning(test_df)
        test_df = self.step4_categorical_encoding_transform(test_df)
        
        # Get feature columns (exclude label and difficulty)
        train_feature_cols = [col for col in train_df.columns if col not in [label_column, 'difficulty']]
        test_feature_cols = [col for col in test_df.columns if col not in [label_column, 'difficulty']]
        
        # Feature alignment for cross-dataset evaluation (BEFORE feature selection)
        if source_data_path or target_test_path:
            logger.info("\n🔗 Aligning features for cross-dataset evaluation...")
            common_features = list(set(train_feature_cols) & set(test_feature_cols))
            train_only = set(train_feature_cols) - set(test_feature_cols)
            test_only = set(test_feature_cols) - set(train_feature_cols)
            
            if train_only:
                logger.warning(f"   ⚠️  {len(train_only)} features in training data but not in test: {list(train_only)[:5]}...")
            if test_only:
                logger.warning(f"   ⚠️  {len(test_only)} features in test data but not in training: {list(test_only)[:5]}...")
            
            logger.info(f"   ✅ Using {len(common_features)} common features")
            feature_cols = sorted(common_features)  # Sort for consistency
            
            # Align dataframes to common features only
            train_df = train_df[[label_column] + feature_cols + (['difficulty'] if 'difficulty' in train_df.columns else [])]
            test_df = test_df[[label_column] + feature_cols + (['difficulty'] if 'difficulty' in test_df.columns else [])]
        else:
            # Same dataset: use all features from training
            feature_cols = train_feature_cols
            # Ensure test has same features (pad with zeros if missing)
            missing_in_test = set(feature_cols) - set(test_feature_cols)
            if missing_in_test:
                for col in missing_in_test:
                    test_df[col] = 0
                    logger.info(f"   Added missing column to test data: {col}")
        
        # Feature selection (if enabled and config available) - runs on aligned features
        if hasattr(self, 'config') and hasattr(self.config, 'use_igrf_rfe') and self.config.use_igrf_rfe:
            logger.info("\n🔍 Running IGRF-RFE feature selection...")
            X_train_features = train_df[feature_cols]
            y_train_multiclass = train_df[label_column].map(self.attack_types).fillna(-1).astype(int)
            
            # Run feature selection
            selected_features = self._run_igrf_rfe(X_train_features, y_train_multiclass)
            logger.info(f"✅ Selected {len(selected_features)} features")
            
            # Apply to both train and test
            train_df = train_df[[label_column, 'difficulty'] + selected_features]
            test_df = test_df[[label_column, 'difficulty'] + selected_features]
            feature_cols = selected_features
        
        # Create zero-day split
        logger.info(f"\n🎯 Creating zero-day split with '{zero_day_attack}' as zero-day attack...")
        
        # Check if zero_day_attack is a category name (when grouping is enabled)
        # Category names: 'DoS', 'Probe', 'R2L', 'U2R', 'Normal'
        category_mapping = {
            'DoS': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'apache2', 'processtable', 'udpstorm'],
            'Probe': ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'],
            'R2L': ['guess_passwd', 'ftp_write', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named', 'worm'],
            'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'sqlattack', 'xterm', 'ps'],
            'Normal': ['normal']
        }
        
        # Determine if zero_day_attack is a category or specific attack
        zero_day_attacks_to_filter = []
        if zero_day_attack in category_mapping:
            # It's a category - filter out all attacks in this category
            zero_day_attacks_to_filter = category_mapping[zero_day_attack]
            logger.info(f"   Category-based zero-day: Filtering out {len(zero_day_attacks_to_filter)} attack types from category '{zero_day_attack}'")
            logger.info(f"   Attacks to exclude: {zero_day_attacks_to_filter}")
        else:
            # It's a specific attack - filter out just this one
            zero_day_attacks_to_filter = [zero_day_attack.lower()]
            logger.info(f"   Specific attack zero-day: Filtering out '{zero_day_attack}'")
        
        # Filter training data to exclude zero-day attack(s)
        train_mask = ~train_df[label_column].isin(zero_day_attacks_to_filter)
        train_df_filtered = train_df[train_mask].copy()
        
        # Test data includes zero-day attack
        test_df_filtered = test_df.copy()
        
        logger.info(f"   Training data after filtering: {len(train_df_filtered)} samples")
        logger.info(f"   Test data: {len(test_df_filtered)} samples")
        
        # Split training into train/val
        from sklearn.model_selection import train_test_split
        
        # Get features and labels
        X_train_full = train_df_filtered[feature_cols].values
        y_train_full = train_df_filtered[label_column].map(self.attack_types).fillna(-1).astype(int)
        
        # Binary labels: 0 = normal, 1 = attack
        y_train_binary = (y_train_full != 0).astype(int)
        
        # Split train/val
        X_train, X_val, y_train, y_val, y_train_binary_split, y_val_binary = train_test_split(
            X_train_full, y_train_full, y_train_binary,
            test_size=0.2, random_state=42, stratify=y_train_binary
        )
        
        # Prepare test data
        X_test = test_df_filtered[feature_cols].values
        y_test_multiclass = test_df_filtered[label_column].map(self.attack_types).fillna(-1).astype(int)
        y_test_binary = (y_test_multiclass != 0).astype(int)
        
        # Feature scaling
        logger.info("\n📏 Scaling features...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Store scaler
        self.scaler = scaler
        
        # Log statistics
        logger.info(f"\n✅ Preprocessing completed!")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Validation samples: {len(X_val)}")
        logger.info(f"   Test samples: {len(X_test)}")
        logger.info(f"   Features: {len(feature_cols)}")
        
        # Count zero-day samples in test set
        if zero_day_attack in category_mapping:
            # Count all attacks in the category
            zero_day_mask = test_df_filtered[label_column].isin(zero_day_attacks_to_filter)
            zero_day_count = zero_day_mask.sum()
            logger.info(f"   Zero-day category: '{zero_day_attack}' ({len(zero_day_attacks_to_filter)} attack types)")
            logger.info(f"   Zero-day samples in test: {zero_day_count}")
        else:
            # Count specific attack
            zero_day_label = self.attack_types.get(zero_day_attack.lower(), -1)
            if zero_day_label >= 0:
                zero_day_count = (y_test_multiclass == zero_day_label).sum()
                logger.info(f"   Zero-day attack: {zero_day_attack} (label: {zero_day_label})")
                logger.info(f"   Zero-day samples in test: {zero_day_count}")
            else:
                logger.warning(f"   Zero-day attack '{zero_day_attack}' not found in attack_types!")
        
        # Return preprocessed data
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_binary_split,
            'y_val': y_val_binary,
            'y_test': y_test_binary,
            'y_train_multiclass': y_train,
            'y_val_multiclass': y_val,
            'y_test_multiclass': y_test_multiclass,
            'test_attack_cat': test_df_filtered[label_column].values,
            'zero_day_attack': zero_day_attack,
            'attack_types': self.attack_types,
            'feature_names': feature_cols,
            'scaler': scaler
        }
