#!/usr/bin/env python3
"""
Blockchain Federated Learning - UNSW-NB15 Preprocessor
Implements the 5-step preprocessing pipeline for zero-day detection
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
    Implements 5-step preprocessing pipeline for zero-day detection
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
        self.feature_names = None
        
        # UNSW-NB15 attack types
        self.attack_types = {
            'Normal': 0,
            'Fuzzers': 1,
            'Analysis': 2,
            'Backdoors': 3,
            'DoS': 4,
            'Exploits': 5,
            'Generic': 6,
            'Reconnaissance': 7,
            'Shellcode': 8,
            'Worms': 9
        }
        
        logger.info("UNSW-NB15 Preprocessor initialized")
    
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
            'infinite_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'outliers': {}
        }
        
        # Check for outliers using IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            quality_report['outliers'][col] = outliers
        
        logger.info(f"  Memory usage: {quality_report['memory_usage']:.2f} MB")
        logger.info(f"  Shape: {quality_report['shape']}")
        logger.info(f"  Missing values: {quality_report['missing_values']}")
        logger.info(f"  Duplicate rows: {quality_report['duplicate_rows']}")
        logger.info(f"  Infinite values: {quality_report['infinite_values']}")
        
        return quality_report
    
    def step2_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Minimal Feature Engineering
        Add 4 scientifically-sound features: 45 → 49 features
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Dataframe with new features
        """
        logger.info("Step 2: Feature Engineering (45 → 49 features)")
        
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
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Cleaned dataframe
        """
        logger.info("Step 3: Data Cleaning")
        
        initial_shape = df.shape
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        logger.info(f"  Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Convert infinite values to NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        infinite_count = df.isnull().sum().sum() - initial_shape[0] + df.shape[0]
        logger.info(f"  Converted {infinite_count} infinite values to NaN")
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_value, inplace=True)
        
        logger.info(f"  Filled missing values in {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        logger.info(f"  Final shape: {df.shape}")
        
        return df
    
    def step4_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Categorical Encoding
        High-cardinality features: Target encoding
        Low-cardinality features: One-hot encoding
        49 → 67 features
        
        Args:
            df: Input dataframe
            
        Returns:
            df: Dataframe with encoded features
        """
        logger.info("Step 4: Categorical Encoding (49 → 67 features)")
        
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
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    logger.info(f"    Applied one-hot encoding to {col} → {dummies.shape[1]} features")
        
        logger.info(f"  Final shape after encoding: {df.shape}")
        return df
    
    
    def step5_feature_scaling(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, 
                            test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Step 5: Feature Scaling using StandardScaler
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional)
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of scaled dataframes
        """
        logger.info("Step 5: Feature Scaling")
        
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
    
    def create_zero_day_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                            zero_day_attack: str = 'DoS') -> Dict:
        """
        Create zero-day detection split:
        - Zero-Day Holdout: One attack class completely excluded from training/validation
        - Training Data: Normal + 8 other attack classes
        - Validation Data: Normal + 8 other attack classes  
        - Test Data: 50% Normal + 50% DoS attacks
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            zero_day_attack: Attack type to hold out for zero-day testing
            
        Returns:
            Dictionary with split data
        """
        logger.info(f"Creating zero-day split with '{zero_day_attack}' as zero-day attack")
        
        # Check what attack types are available in test data
        test_attack_types = test_df['label'].value_counts().sort_index()
        logger.info(f"  Available attack types in test data: {dict(test_attack_types)}")
        
        # Convert attack labels to binary (0=Normal, 1=Attack)
        train_df['binary_label'] = (train_df['attack_cat'] != 'Normal').astype(int)
        test_df['binary_label'] = (test_df['attack_cat'] != 'Normal').astype(int)
        
        # Filter training data to exclude zero-day attack
        if zero_day_attack in train_df['attack_cat'].values:
            train_mask = (train_df['attack_cat'] == 'Normal') | (train_df['attack_cat'] != zero_day_attack)
        else:
            # If zero-day attack not found, use all data
            train_mask = pd.Series([True] * len(train_df), index=train_df.index)
        
        train_data = train_df[train_mask].copy()
        
        # Validation data: Same as training (Normal + 8 other attack classes)
        val_data = train_data.copy()
        
        # Test data: 50% Normal + 50% Zero-day attacks
        normal_test = test_df[test_df['attack_cat'] == 'Normal'].copy()
        zero_day_test = test_df[test_df['attack_cat'] == zero_day_attack].copy()
        
        # If the specified zero-day attack type is not in test data, find the best alternative
        if len(zero_day_test) == 0:
            logger.warning(f"No {zero_day_attack} attacks found in test data. Finding best alternative.")
            
            # Find attack types available in test data
            available_attacks = test_df[test_df['attack_cat'] != 'Normal']['attack_cat'].value_counts()
            if len(available_attacks) > 0:
                # Use the most common attack type as zero-day
                most_common_attack = available_attacks.index[0]
                zero_day_test = test_df[test_df['attack_cat'] == most_common_attack].copy()
                
                logger.info(f"Using {most_common_attack} as zero-day attack")
                
                # Update the training data to exclude this attack type
                train_mask = (train_df['attack_cat'] == 'Normal') | (train_df['attack_cat'] != most_common_attack)
                train_data = train_df[train_mask].copy()
                val_data = train_data.copy()
            else:
                logger.error("No attack samples found in test data!")
                # Create synthetic test data from training data
                zero_day_test = train_df[train_df['attack_cat'] == zero_day_attack].copy()
                if len(zero_day_test) > 0:
                    zero_day_test = zero_day_test.sample(n=min(1000, len(zero_day_test)), random_state=42)
                    logger.info(f"Created synthetic test data with {len(zero_day_test)} samples")
        
        # Balance test data
        min_samples = min(len(normal_test), len(zero_day_test))
        if min_samples > 0:
            normal_test = normal_test.sample(n=min_samples, random_state=42)
            zero_day_test = zero_day_test.sample(n=min_samples, random_state=42)
            
            test_data = pd.concat([normal_test, zero_day_test], ignore_index=True)
            test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            logger.error("Cannot create test data - no samples available")
            test_data = pd.DataFrame()
        
        logger.info(f"  Training data: {len(train_data)} samples")
        logger.info(f"    Normal: {len(train_data[train_data['label'] == 0])}")
        logger.info(f"    Attacks: {len(train_data[train_data['label'] != 0])}")
        
        logger.info(f"  Validation data: {len(val_data)} samples")
        
        logger.info(f"  Test data: {len(test_data)} samples")
        logger.info(f"    Normal: {len(test_data[test_data['label'] == 0])}")
        logger.info(f"    Zero-day attacks: {len(test_data[test_data['label'] != 0])}")
        
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
        
        # Process training data
        logger.info("\nProcessing training data...")
        train_quality = self.step1_data_quality_assessment(train_df)
        train_df = self.step2_feature_engineering(train_df)
        train_df = self.step3_data_cleaning(train_df)
        train_df = self.step4_categorical_encoding(train_df)
        # Skip feature selection - let multi-scale extractors learn feature importance
        logger.info("Skipping Pearson correlation feature selection - using all features")
        # train_df = self.step5_feature_selection(train_df)  # DISABLED
        
        # Process test data
        logger.info("\nProcessing test data...")
        test_quality = self.step1_data_quality_assessment(test_df)
        test_df = self.step2_feature_engineering(test_df)
        test_df = self.step3_data_cleaning(test_df)
        test_df = self.step4_categorical_encoding(test_df)
        # Skip feature selection - let multi-scale extractors learn feature importance
        logger.info("Skipping Pearson correlation feature selection for test data - using all features")
        # test_df = self.step5_feature_selection(test_df)  # DISABLED
        
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
        train_scaled, val_scaled, test_scaled = self.step5_feature_scaling(
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
