#!/usr/bin/env python3
"""
Blockchain Federated Learning - CICIDS2023 Preprocessor
Customized for Zero-Day Attack Detection

NOTE: This is a template. You need to:
1. Update attack_types dictionary with actual CICIDS2023 attack names
2. Update label_column name if different from 'Label'
3. Verify feature count matches your dataset
"""
import pandas as pd
import numpy as np
import logging
import warnings
import re
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICIDS2023Preprocessor(UNSWPreprocessor):
    """
    Customized Preprocessor for CICIDS2023 Dataset
    
    IMPORTANT: Before using, you must:
    1. Inspect your CICIDS2023 CSV files to identify attack types
    2. Update self.attack_types dictionary below
    3. Update label_column if different from 'Label'
    4. Verify input_dim in config.py matches your feature count
    """
    
    def __init__(self, data_path: str = "CICIDS2023_train.csv", test_path: str = "CICIDS2023_test.csv"):
        super().__init__(data_path, test_path)
        
        # CICIoT2023 attack types (from full dataset inspection)
        # Total: 34 unique labels including BenignTraffic
        self.attack_types = {
            'BenignTraffic': 0,  # BENIGN class
            'Backdoor_Malware': 1,
            'BrowserHijacking': 2,
            'CommandInjection': 3,
            'DDoS-ACK_Fragmentation': 4,
            'DDoS-HTTP_Flood': 5,
            'DDoS-ICMP_Flood': 6,
            'DDoS-ICMP_Fragmentation': 7,
            'DDoS-PSHACK_Flood': 8,
            'DDoS-RSTFINFlood': 9,
            'DDoS-SYN_Flood': 10,
            'DDoS-SlowLoris': 11,
            'DDoS-SynonymousIP_Flood': 12,
            'DDoS-TCP_Flood': 13,
            'DDoS-UDP_Flood': 14,
            'DDoS-UDP_Fragmentation': 15,
            'DNS_Spoofing': 16,
            'DictionaryBruteForce': 17,
            'DoS-HTTP_Flood': 18,
            'DoS-SYN_Flood': 19,
            'DoS-TCP_Flood': 20,
            'DoS-UDP_Flood': 21,
            'MITM-ArpSpoofing': 22,
            'Mirai-greeth_flood': 23,
            'Mirai-greip_flood': 24,
            'Mirai-udpplain': 25,
            'Recon-HostDiscovery': 26,
            'Recon-OSScan': 27,
            'Recon-PingSweep': 28,
            'Recon-PortScan': 29,
            'SqlInjection': 30,
            'Uploading_Attack': 31,
            'VulnerabilityScan': 32,
            'XSS': 33,
            # Also map BENIGN for compatibility
            'BENIGN': 0,
        }
        logger.info("CICIoT2023 Preprocessor initialized")
    
    def load_and_clean_columns(self, path):
        """Helper to load CSV and strip whitespace from column names with memory optimization"""
        logger.info(f"Loading CICIDS2023 CSV file: {path}")
        try:
            # MEMORY-EFFICIENT: Use chunking for large files
            logger.info("   Reading CSV header to determine structure...")
            sample_df = pd.read_csv(path, nrows=1000)
            logger.info(f"   Sample shape: {sample_df.shape}, columns: {len(sample_df.columns)}")
            
            # Determine optimal dtypes to reduce memory
            dtype_dict = {}
            for col in sample_df.columns:
                if sample_df[col].dtype == 'int64':
                    if sample_df[col].min() >= 0 and sample_df[col].max() <= 255:
                        dtype_dict[col] = 'uint8'
                    elif sample_df[col].min() >= -128 and sample_df[col].max() <= 127:
                        dtype_dict[col] = 'int8'
                    elif sample_df[col].min() >= 0 and sample_df[col].max() <= 65535:
                        dtype_dict[col] = 'uint16'
                    elif sample_df[col].min() >= -32768 and sample_df[col].max() <= 32767:
                        dtype_dict[col] = 'int16'
                    else:
                        dtype_dict[col] = 'int32'
                elif sample_df[col].dtype == 'float64':
                    dtype_dict[col] = 'float32'
            
            logger.info(f"   Loading full CSV with optimized dtypes...")
            
            # Check file size
            import os
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"   File size: {file_size_mb:.1f} MB")
            
            # Use chunked reading for large files
            chunk_sizes = [50000, 25000, 10000, 5000]
            df = None
            use_chunking_first = file_size_mb > 500
            
            if use_chunking_first:
                logger.info(f"   File is large ({file_size_mb:.1f} MB), using chunked reading...")
                df = None
            else:
                try:
                    df = pd.read_csv(path, dtype=dtype_dict, low_memory=False)
                    logger.info(f"   ✅ CSV loaded in one pass: {df.shape}")
                except MemoryError:
                    logger.warning("   Memory error, falling back to chunked reading...")
                    use_chunking_first = True
            
            if use_chunking_first or df is None:
                chunk_list = []
                for chunk_size in chunk_sizes:
                    try:
                        logger.info(f"   Trying chunk size: {chunk_size}")
                        for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=dtype_dict, low_memory=False):
                            chunk.columns = chunk.columns.str.strip()
                            chunk_list.append(chunk)
                        break
                    except MemoryError:
                        logger.warning(f"   Chunk size {chunk_size} too large, trying smaller...")
                        continue
                
                if chunk_list:
                    df = pd.concat(chunk_list, ignore_index=True)
                    logger.info(f"   ✅ CSV loaded via chunking: {df.shape}")
                else:
                    raise MemoryError("Could not load file even with smallest chunk size")
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")
            raise
    
    def preprocess_unsw_dataset(self, zero_day_attack: str = 'PortScan') -> dict:
        """
        Main Pipeline for CICIDS2023
        TODO: Update label_column name if different from 'Label'
        """
        logger.info(f"Starting CICIDS2023 preprocessing (Zero-Day Target: {zero_day_attack})")
        
        # CICIoT2023 uses lowercase 'label' column
        label_column = 'label'  # CICIoT2023 uses lowercase 'label'
        
        # 1. Load Data with column cleaning
        train_df = self.load_and_clean_columns(self.data_path)
        test_df = self.load_and_clean_columns(self.test_path)
        
        # Verify label column exists
        if label_column not in train_df.columns:
            logger.error(f"❌ Label column '{label_column}' not found!")
            logger.error(f"   Available columns: {train_df.columns.tolist()[:10]}...")
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # 2. Normalize Label column BEFORE sampling
        def normalize_label(label):
            """Normalize label to match attack_types keys"""
            if pd.isna(label):
                return 'BenignTraffic'  # CICIoT2023 uses BenignTraffic
            label_str = str(label).strip()
            
            # Try exact match first
            if label_str in self.attack_types:
                return label_str
            
            # Case-insensitive matching
            label_upper = label_str.upper()
            for key in self.attack_types.keys():
                if key.upper() == label_upper:
                    return key
            
            # Special handling for BenignTraffic variations
            if 'benign' in label_str.lower() or 'normal' in label_str.lower():
                return 'BenignTraffic'
            
            # If not found, log warning and return as-is (will cause error later)
            logger.warning(f"⚠️  Unknown attack type: '{label_str}' - add to attack_types dictionary!")
            return label_str
        
        logger.info("Normalizing labels...")
        train_df[label_column] = train_df[label_column].apply(normalize_label)
        test_df[label_column] = test_df[label_column].apply(normalize_label)
        
        # 3. Override feature engineering to skip UNSW-specific features
        # CICIoT2023 doesn't have sbytes/dbytes columns used by parent class
        
        # Process training data
        logger.info("\nProcessing training data...")
        train_quality = self.step1_data_quality_assessment(train_df)
        # Skip step2_feature_engineering (uses sbytes/dbytes which don't exist in CICIoT2023)
        train_df = self.step3_data_cleaning(train_df)
        train_df = self.step4_categorical_encoding(train_df)
        
        logger.info("\nProcessing test data...")
        test_quality = self.step1_data_quality_assessment(test_df)
        # Skip step2_feature_engineering
        test_df = self.step3_data_cleaning(test_df)
        test_df = self.step4_categorical_encoding_transform(test_df)
        
        # Continue with rest of preprocessing (feature selection, scaling, etc.)
        # We'll need to call the remaining steps manually or create a simplified pipeline
        # For now, let's use a simplified approach similar to CICIDS2017
        
        # Map labels to integers
        train_df['label_int'] = train_df[label_column].map(self.attack_types)
        test_df['label_int'] = test_df[label_column].map(self.attack_types)
        
        # Check for unmapped labels
        unmapped_train = train_df[train_df['label_int'].isna()]
        unmapped_test = test_df[test_df['label_int'].isna()]
        if len(unmapped_train) > 0:
            logger.warning(f"⚠️  {len(unmapped_train)} training samples with unmapped labels")
            logger.warning(f"   Unique unmapped labels: {unmapped_train[label_column].unique()}")
        if len(unmapped_test) > 0:
            logger.warning(f"⚠️  {len(unmapped_test)} test samples with unmapped labels")
            logger.warning(f"   Unique unmapped labels: {unmapped_test[label_column].unique()}")
        
        # Drop unmapped samples
        train_df = train_df.dropna(subset=['label_int'])
        test_df = test_df.dropna(subset=['label_int'])
        
        # Create binary labels (0=BenignTraffic, 1=Attack)
        train_df['binary_label'] = (train_df['label_int'] != 0).astype(int)
        test_df['binary_label'] = (test_df['label_int'] != 0).astype(int)
        
        # Store attack category for zero-day detection
        train_df['attack_cat'] = train_df[label_column]
        test_df['attack_cat'] = test_df[label_column]
        
        # Get feature columns (exclude label columns)
        exclude_cols = [label_column, 'label_int', 'binary_label', 'attack_cat']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        logger.info(f"Feature columns: {len(feature_cols)}")
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Handle missing values in features
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)
        
        # Split training data into train and validation sets
        from sklearn.model_selection import train_test_split
        train_df_split, val_df = train_test_split(
            train_df,
            test_size=0.2,  # 20% for validation
            random_state=42,
            stratify=train_df['binary_label']  # Stratify to maintain class balance
        )
        train_df = train_df_split
        
        logger.info(f"After train/val split - Training: {len(train_df)}, Validation: {len(val_df)}")
        
        # Scale features (using StandardScaler)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()
        train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
        test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        
        # Convert to tensors
        import torch
        X_train = torch.FloatTensor(train_scaled[feature_cols].values)
        y_train = torch.LongTensor(train_scaled['binary_label'].values)
        X_val = torch.FloatTensor(val_scaled[feature_cols].values)
        y_val = torch.LongTensor(val_scaled['binary_label'].values)
        X_test = torch.FloatTensor(test_scaled[feature_cols].values)
        y_test = torch.LongTensor(test_scaled['binary_label'].values)
        
        # Create zero-day indices
        zero_day_attack_label = self.attack_types.get(zero_day_attack, 29)
        zero_day_mask = test_scaled['attack_cat'] == zero_day_attack
        zero_day_indices = torch.where(torch.tensor(zero_day_mask.values, dtype=torch.bool))[0].tolist()
        
        logger.info("\n✅ CICIoT2023 preprocessing completed successfully!")
        logger.info(f"Final feature count: {len(feature_cols)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Zero-day samples ({zero_day_attack}, label={zero_day_attack_label}): {len(zero_day_indices)}")
        
        # Store multiclass labels
        y_train_multiclass = torch.LongTensor(train_scaled['label_int'].values.copy())
        y_val_multiclass = torch.LongTensor(val_scaled['label_int'].values.copy())
        y_test_multiclass = torch.LongTensor(test_scaled['label_int'].values.copy())
        
        logger.info(f"Validation samples: {len(X_val)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'y_train_multiclass': y_train_multiclass,
            'X_val': X_val,
            'y_val': y_val,
            'y_val_multiclass': y_val_multiclass,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_multiclass': y_test_multiclass,
            'test_attack_cat': test_scaled['attack_cat'].values,
            'zero_day_indices': zero_day_indices,
            'zero_day_attack': zero_day_attack,
            'zero_day_attack_label': zero_day_attack_label,
            'scaler': scaler,
            'feature_names': feature_cols,  # Use 'feature_names' to match expected format
            'attack_types': self.attack_types
        }

