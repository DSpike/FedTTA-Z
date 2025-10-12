#!/usr/bin/env python3
"""
Edge-IIoTset Dataset Preprocessor for Federated Learning
Handles the Edge-IIoTset dataset with 2.2M+ samples and 14 attack types
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EdgeIIoTPreprocessor:
    """
    Preprocessor for Edge-IIoTset dataset
    Handles 2.2M+ samples with 63 features and 14 attack types
    """
    
    def __init__(self, data_path: str, test_split: float = 0.2, val_split: float = 0.1):
        """
        Initialize Edge-IIoTset preprocessor
        
        Args:
            data_path: Path to Edge-IIoTset CSV file
            test_split: Fraction of data for testing
            val_split: Fraction of training data for validation
        """
        self.data_path = data_path
        self.test_split = test_split
        self.val_split = val_split
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.attack_types = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load Edge-IIoTset dataset with memory optimization
        
        Returns:
            df: Loaded dataset
        """
        logger.info(f"Loading Edge-IIoTset dataset from {self.data_path}")
        
        try:
            # Load with memory optimization
            df = pd.read_csv(
                self.data_path,
                low_memory=False,  # Handle mixed types
                dtype={'Attack_label': 'int8'}  # Optimize memory
            )
            
            logger.info(f"✅ Dataset loaded successfully: {df.shape}")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataset
        
        Args:
            df: Raw dataset
            
        Returns:
            df: Cleaned dataset
        """
        logger.info("🧹 Cleaning Edge-IIoTset dataset...")
        
        # Store original shape
        original_shape = df.shape
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Handle missing values in categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        logger.info(f"Data cleaning completed: {original_shape} → {df.shape}")
        logger.info(f"Removed {original_shape[0] - df.shape[0]} rows")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on Edge-IIoTset data
        
        Args:
            df: Cleaned dataset
            
        Returns:
            df: Dataset with engineered features
        """
        logger.info("🔧 Performing feature engineering...")
        
        # Drop frame.time column to avoid parsing issues
        if 'frame.time' in df.columns:
            logger.info("🗑️ Dropping frame.time column to avoid parsing issues...")
            df = df.drop('frame.time', axis=1)
        
        # Network flow features
        if 'tcp.len' in df.columns:
            df['tcp.len'] = pd.to_numeric(df['tcp.len'], errors='coerce').fillna(0)
            df['tcp_len_log'] = np.log1p(df['tcp.len'])
        
        if 'udp.time_delta' in df.columns:
            df['udp.time_delta'] = pd.to_numeric(df['udp.time_delta'], errors='coerce').fillna(0)
            df['udp_time_delta_log'] = np.log1p(df['udp.time_delta'])
        
        # Protocol-specific features
        if 'tcp.srcport' in df.columns and 'tcp.dstport' in df.columns:
            # Ensure ports are numeric before subtraction
            df['tcp.srcport'] = pd.to_numeric(df['tcp.srcport'], errors='coerce').fillna(0)
            df['tcp.dstport'] = pd.to_numeric(df['tcp.dstport'], errors='coerce').fillna(0)
            df['port_range'] = df['tcp.dstport'] - df['tcp.srcport']
            df['is_high_port'] = (df['tcp.dstport'] > 1024).astype(int)
        
        # MQTT features
        mqtt_cols = [col for col in df.columns if col.startswith('mqtt.')]
        if mqtt_cols:
            df['mqtt_features_count'] = df[mqtt_cols].notna().sum(axis=1)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for intrusion detection
        
        Args:
            df: Dataset with engineered features
            
        Returns:
            df: Dataset with selected features
        """
        logger.info("🎯 Selecting features for intrusion detection...")
        
        # Features to exclude (non-predictive)
        exclude_features = [
            'frame.time',  # Time information
            'ip.src_host',  # IP addresses (too specific)
            'ip.dst_host',
            'arp.dst.proto_ipv4',
            'arp.src.proto_ipv4'
            # Note: Attack_type is kept for zero-day split functionality
        ]
        
        # Select features
        feature_columns = [col for col in df.columns if col not in exclude_features]
        
        # Ensure we have the target variable and attack type for zero-day split
        if 'Attack_label' not in feature_columns:
            feature_columns.append('Attack_label')
        if 'Attack_type' not in feature_columns:
            feature_columns.append('Attack_type')
        
        df_selected = df[feature_columns].copy()
        
        # Store feature names (exclude target and attack type columns)
        self.feature_names = [col for col in feature_columns if col not in ['Attack_label', 'Attack_type']]
        
        logger.info(f"Selected {len(self.feature_names)} features")
        logger.info(f"Features: {self.feature_names[:10]}...")  # Show first 10
        
        return df_selected
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features to numeric
        
        Args:
            df: Dataset with selected features
            
        Returns:
            df: Dataset with encoded features
        """
        logger.info("🔢 Encoding categorical features...")
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'Attack_label' in categorical_columns:
            categorical_columns.remove('Attack_label')
        
        # Encode categorical features
        for col in categorical_columns:
            if col in df.columns:
                # Use label encoding for categorical features
                df[col] = df[col].astype(str)
                df[col] = pd.Categorical(df[col]).codes
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        
        return df
    
    def handle_imbalanced_data(self, df: pd.DataFrame, target_col: str = 'Attack_label') -> pd.DataFrame:
        """
        Handle class imbalance in the dataset
        
        Args:
            df: Dataset
            target_col: Target column name
            
        Returns:
            df: Balanced dataset
        """
        logger.info("⚖️ Handling class imbalance...")
        
        # Check class distribution
        class_counts = df[target_col].value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        # Calculate imbalance ratio
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:  # Significant imbalance
            logger.info("Significant class imbalance detected. Applying sampling...")
            
            # Downsample majority class
            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()
            
            # Sample majority class to match minority class size
            majority_samples = df[df[target_col] == majority_class]
            minority_samples = df[df[target_col] == minority_class]
            
            # Sample majority class
            majority_sampled = majority_samples.sample(
                n=len(minority_samples),
                random_state=42
            )
            
            # Combine balanced samples
            df_balanced = pd.concat([majority_sampled, minority_samples], ignore_index=True)
            
            # Shuffle the dataset
            df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Balanced dataset: {df_balanced.shape}")
            logger.info(f"New class distribution: {df_balanced[target_col].value_counts().to_dict()}")
            
            return df_balanced
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Processed dataset
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("📊 Splitting data into train/validation/test sets...")
        
        # Separate features and target
        X = df.drop('Attack_label', axis=1).values
        y = df['Attack_label'].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_split, 
            random_state=42, 
            stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_split,
            random_state=42,
            stratify=y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples")
        logger.info(f"  Testing: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train, X_val, X_test: Feature arrays
            
        Returns:
            Scaled feature arrays
        """
        logger.info("📏 Scaling features...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_zero_day_split(self, df: pd.DataFrame, zero_day_attack: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create zero-day attack split by removing specified attack type from training
        
        Args:
            df: Dataset with Attack_type column
            zero_day_attack: Attack type to treat as zero-day
            
        Returns:
            train_df: Training data without zero-day attack
            test_df: Test data with only zero-day attack
        """
        logger.info(f"🎯 Creating zero-day attack split for: {zero_day_attack}")
        
        # Check if zero_day_attack exists in the dataset
        available_attacks = df['Attack_type'].unique()
        if zero_day_attack not in available_attacks:
            logger.warning(f"⚠️ Attack type '{zero_day_attack}' not found in dataset!")
            logger.info(f"Available attack types: {list(available_attacks)}")
            # Use the first available attack as fallback
            zero_day_attack = available_attacks[0]
            logger.info(f"Using '{zero_day_attack}' as zero-day attack instead")
        
        # Create training data (exclude zero-day attack)
        train_df = df[df['Attack_type'] != zero_day_attack].copy()
        
        # Create test data (only zero-day attack + some normal samples)
        zero_day_samples = df[df['Attack_type'] == zero_day_attack].copy()
        normal_samples = df[df['Attack_type'] == 'Normal'].copy()
        
        # Sample normal data for test set (to maintain some normal samples)
        normal_sample_size = min(len(zero_day_samples), len(normal_samples) // 2)
        if normal_sample_size > 0:
            normal_test_samples = normal_samples.sample(n=normal_sample_size, random_state=42)
            test_df = pd.concat([zero_day_samples, normal_test_samples], ignore_index=True)
        else:
            test_df = zero_day_samples
        
        # Shuffle test data
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Zero-day split created:")
        logger.info(f"  Training samples: {len(train_df)} (excluding {zero_day_attack})")
        logger.info(f"  Test samples: {len(test_df)} (only {zero_day_attack} + normal)")
        logger.info(f"  Zero-day attack samples: {len(zero_day_samples)}")
        
        return train_df, test_df
    
    def preprocess_edgeiiot_dataset(self, zero_day_attack: str = "DDoS_UDP") -> Dict:
        """
        Complete preprocessing pipeline for Edge-IIoTset dataset with zero-day attack simulation
        
        Args:
            zero_day_attack: Attack type to treat as zero-day (unseen during training)
            
        Returns:
            preprocessed_data: Dictionary containing processed data
        """
        logger.info("🚀 Starting Edge-IIoTset preprocessing pipeline...")
        logger.info(f"🎯 Zero-day attack simulation: {zero_day_attack}")
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 4: Select features (but keep Attack_type for zero-day split)
        df = self.select_features(df)
        
        # Step 5: Create zero-day attack split
        train_df, test_df = self.create_zero_day_split(df, zero_day_attack)
        
        # Step 6: Encode categorical features
        train_df = self.encode_categorical_features(train_df)
        test_df = self.encode_categorical_features(test_df)
        
        # Step 7: Handle class imbalance (on training data only)
        train_df = self.handle_imbalanced_data(train_df)
        
        # Step 8: Split training data into train/val
        X_train, X_val, y_train, y_val = self._split_train_val(train_df)
        
        # Step 9: Prepare test data
        X_test = test_df.drop('Attack_label', axis=1).values
        y_test = test_df['Attack_label'].values
        
        # Step 10: Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)
        
        # Store attack types for reference
        self.attack_types = df['Attack_type'].unique().tolist() if 'Attack_type' in df.columns else []
        
        # Calculate zero-day statistics
        zero_day_stats = self._calculate_zero_day_stats(df, zero_day_attack)
        
        # Prepare results
        preprocessed_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'attack_types': self.attack_types,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'zero_day_attack': zero_day_attack,
            'zero_day_stats': zero_day_stats
        }
        
        logger.info("✅ Edge-IIoTset preprocessing completed successfully!")
        logger.info(f"Final dataset shape: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Attack types: {len(self.attack_types)}")
        logger.info(f"Zero-day attack: {zero_day_attack}")
        
        return preprocessed_data
    
    def _split_train_val(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split training data into train and validation sets
        
        Args:
            df: Training dataframe
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Separate features and target
        X = df.drop('Attack_label', axis=1).values
        y = df['Attack_label'].values
        
        # Split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.val_split,
            random_state=42,
            stratify=y
        )
        
        return X_train, X_val, y_train, y_val
    
    def _calculate_zero_day_stats(self, df: pd.DataFrame, zero_day_attack: str) -> Dict:
        """
        Calculate statistics for zero-day attack simulation
        
        Args:
            df: Original dataset
            zero_day_attack: Zero-day attack type
            
        Returns:
            stats: Zero-day attack statistics
        """
        # Count samples by attack type
        attack_counts = df['Attack_type'].value_counts()
        
        # Zero-day attack statistics
        zero_day_count = attack_counts.get(zero_day_attack, 0)
        total_attacks = attack_counts.sum() - attack_counts.get('Normal', 0)
        zero_day_percentage = (zero_day_count / total_attacks) * 100 if total_attacks > 0 else 0
        
        stats = {
            'zero_day_attack': zero_day_attack,
            'zero_day_samples': int(zero_day_count),
            'zero_day_percentage': float(zero_day_percentage),
            'total_attack_samples': int(total_attacks),
            'normal_samples': int(attack_counts.get('Normal', 0)),
            'total_samples': len(df),
            'available_attacks': list(attack_counts.index)
        }
        
        logger.info(f"Zero-day attack statistics:")
        logger.info(f"  Attack type: {zero_day_attack}")
        logger.info(f"  Samples: {zero_day_count:,}")
        logger.info(f"  Percentage of attacks: {zero_day_percentage:.1f}%")
        
        return stats
    
    def sample_stratified_subset(self, X: np.ndarray, y: np.ndarray, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a stratified subset of the data
        
        Args:
            X: Features
            y: Labels
            n_samples: Number of samples to return
            
        Returns:
            X_subset, y_subset: Stratified subset
        """
        if len(X) <= n_samples:
            return X, y
        
        # Get unique classes
        unique_classes = np.unique(y)
        
        # Calculate samples per class
        samples_per_class = n_samples // len(unique_classes)
        
        X_subset_list = []
        y_subset_list = []
        
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            # Sample from this class
            n_class_samples = min(samples_per_class, len(class_X))
            if n_class_samples > 0:
                indices = np.random.choice(len(class_X), n_class_samples, replace=False)
                X_subset_list.append(class_X[indices])
                y_subset_list.append(class_y[indices])
        
        # Combine all classes
        X_subset = np.vstack(X_subset_list)
        y_subset = np.hstack(y_subset_list)
        
        # Shuffle
        shuffle_indices = np.random.permutation(len(X_subset))
        X_subset = X_subset[shuffle_indices]
        y_subset = y_subset[shuffle_indices]
        
        return X_subset, y_subset
