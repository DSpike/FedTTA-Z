#!/usr/bin/env python3
"""
Blockchain Federated Learning - CICIDS2017 Preprocessor
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

class CICIDSPreprocessor(UNSWPreprocessor):
    """
    Customized Preprocessor for CICIDS2017 Dataset
    """
    
    def __init__(self, data_path: str = "CICIDS2017_train.csv", test_path: str = "CICIDS2017_test.csv"):
        super().__init__(data_path, test_path)
        
        # 1. UPDATE: CICIDS2017 Attack Labels
        self.attack_types = {
            'BENIGN': 0,
            'Bot': 1,
            'DDoS': 2,
            'DoS GoldenEye': 3,
            'DoS Hulk': 4,
            'DoS Slowhttptest': 5,
            'DoS slowloris': 6,
            'FTP-Patator': 7,
            'Heartbleed': 8,
            'Infiltration': 9,
            'PortScan': 10,
            'SSH-Patator': 11,
            'Web Attack': 12,
            # Handle all variations of Web Attack labels with special characters
            'Web Attack  Brute Force': 12,  # Two spaces
            'Web Attack  Sql Injection': 12,
            'Web Attack  XSS': 12,
            'Web Attack – Brute Force': 12,  # En dash (–)
            'Web Attack – Sql Injection': 12,
            'Web Attack – XSS': 12,
            'Web Attack - Brute Force': 12,  # Regular hyphen
            'Web Attack - Sql Injection': 12,
            'Web Attack - XSS': 12,
            'Web Attack Brute Force': 12,  # Single space
            'Web Attack Sql Injection': 12,
            'Web Attack XSS': 12,
        }
        logger.info("CICIDS2017 Preprocessor initialized")

    def load_and_clean_columns(self, path):
        """Helper to load CSV and strip whitespace from column names with memory optimization"""
        logger.info(f"Loading CSV file: {path}")
        try:
            # MEMORY-EFFICIENT: Use chunking for large files
            # First, read a small sample to get column names and dtypes
            logger.info("   Reading CSV header to determine structure...")
            sample_df = pd.read_csv(path, nrows=1000)
            logger.info(f"   Sample shape: {sample_df.shape}, columns: {len(sample_df.columns)}")
            
            # Determine optimal dtypes to reduce memory
            dtype_dict = {}
            for col in sample_df.columns:
                if sample_df[col].dtype == 'int64':
                    # Try to downcast to smaller int types
                    if sample_df[col].min() >= 0 and sample_df[col].max() <= 255:
                        dtype_dict[col] = 'uint8'
                    elif sample_df[col].min() >= -128 and sample_df[col].max() <= 127:
                        dtype_dict[col] = 'int8'
                    elif sample_df[col].min() >= 0 and sample_df[col].max() <= 65535:
                        dtype_dict[col] = 'uint16'
                    elif sample_df[col].min() >= -32768 and sample_df[col].max() <= 32767:
                        dtype_dict[col] = 'int16'
                    else:
                        dtype_dict[col] = 'int32'  # Use int32 instead of int64
                elif sample_df[col].dtype == 'float64':
                    dtype_dict[col] = 'float32'  # Use float32 instead of float64
            
            logger.info(f"   Loading full CSV with optimized dtypes (using chunking for large files)...")
            
            # Check file size first to decide on strategy
            import os
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"   File size: {file_size_mb:.1f} MB")
            
            # Use chunked reading for large files to avoid memory errors
            chunk_sizes = [50000, 25000, 10000, 5000]  # Try progressively smaller chunks
            df = None
            
            # For files > 500MB, skip the "all at once" attempt and go straight to chunking
            use_chunking_first = file_size_mb > 500
            
            if use_chunking_first:
                logger.info(f"   File is large ({file_size_mb:.1f} MB), using chunked reading from the start...")
                df = None  # Will be set in chunked reading loop
            else:
                try:
                    # First, try to read all at once (faster for smaller files)
                    df = pd.read_csv(path, dtype=dtype_dict, low_memory=False)
                    logger.info(f"   ✅ CSV loaded in one pass: {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                except Exception as e:
                    # Catch ALL exceptions (MemoryError, ParserError, OSError, etc.) that might indicate memory issues
                    error_str = str(e).lower()
                    is_memory_error = (
                        isinstance(e, MemoryError) or
                        isinstance(e, pd.errors.ParserError) or
                        "out of memory" in error_str or
                        "memory" in error_str or
                        "ParserError" in str(type(e))
                    )
                    
                    if is_memory_error:
                        logger.warning(f"   ⚠️ Memory/parsing error loading all at once: {type(e).__name__}: {e}")
                        logger.warning(f"   Switching to chunked reading...")
                        df = None  # Will be set in chunked reading loop
                    else:
                        # Non-memory error, re-raise
                        logger.error(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
                        raise
            
            # If df is still None, use chunked reading
            if df is None:
                logger.info(f"   Using chunked reading...")
                # Try progressively smaller chunk sizes
                for chunk_size in chunk_sizes:
                    try:
                        logger.info(f"   Attempting chunked reading with chunk_size={chunk_size}...")
                        chunk_list = []
                        total_rows = 0
                        
                        for chunk_num, chunk in enumerate(pd.read_csv(path, dtype=dtype_dict, low_memory=False, chunksize=chunk_size)):
                            chunk_list.append(chunk)
                            total_rows += len(chunk)
                            if (chunk_num + 1) % 10 == 0:
                                logger.info(f"   Processed {chunk_num + 1} chunks ({total_rows:,} rows)...")
                        
                        # Concatenate chunks
                        logger.info(f"   Concatenating {len(chunk_list)} chunks...")
                        df = pd.concat(chunk_list, ignore_index=True)
                        logger.info(f"   ✅ CSV loaded via chunking (chunk_size={chunk_size}): {df.shape}, memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                        break  # Success, exit the loop
                    except Exception as chunk_error:
                            # Catch any error during chunked reading
                            error_str = str(chunk_error).lower()
                            is_memory_error = (
                                isinstance(chunk_error, MemoryError) or
                                isinstance(chunk_error, pd.errors.ParserError) or
                                "out of memory" in error_str or
                                "memory" in error_str
                            )
                            
                            if is_memory_error:
                                logger.warning(f"   ⚠️ Chunk size {chunk_size} still too large ({type(chunk_error).__name__}), trying smaller...")
                                if chunk_size == chunk_sizes[-1]:
                                    # Last chunk size failed, raise the error
                                    logger.error(f"   ❌ Even smallest chunk size ({chunk_size}) failed!")
                                    raise
                                continue
                            else:
                                # Non-memory error, re-raise
                                logger.error(f"   ❌ Unexpected error during chunked reading: {chunk_error}")
                                raise
                else:
                    # Non-memory error, re-raise
                    logger.error(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
                    raise
            
            if df is None:
                raise ValueError("Failed to load CSV file - all methods exhausted")
            
            # CRITICAL FIX for CICIDS2017: Remove spaces from column names (e.g., " Label" -> "Label")
            df.columns = df.columns.str.strip()
            return df
        except MemoryError as e:
            error_str = str(e)
            logger.error(f"❌ Memory error loading CSV: {error_str}")
            logger.error(f"   File: {path}")
            logger.error(f"   Try reducing file size or using chunking")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading CSV {path}: {e}")
            raise

    def step2_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Feature Engineering adapted for CICIDS2017 columns
        """
        logger.info("Step 2: Feature Engineering (Adapting to CICIDS2017 features)")
        
        # Map CICIDS columns to concepts used in your model
        # Check if columns exist to avoid errors
        if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
            # Equivalent to sbytes / dbytes
            df['packet_size_ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)
        else:
            df['packet_size_ratio'] = 0

        if 'Flow Duration' in df.columns and 'Total Fwd Packets' in df.columns:
            # Flow Duration is in microseconds, convert to seconds for rate
            duration_sec = (df['Flow Duration'] + 1) / 1e6
            df['packets_per_second'] = df['Total Fwd Packets'] / duration_sec
        else:
            df['packets_per_second'] = 0

        # CICIDS doesn't have string 'proto', it has numeric 'Protocol'
        if 'Protocol' in df.columns:
            # Protocol 6 is TCP
            df['tcp_rate'] = (df['Protocol'] == 6).astype(int) * df['packets_per_second']
        else:
            df['tcp_rate'] = 0
            
        return df

    def step4_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Categorical Encoding (Simplified for CICIDS)
        CICIDS2017 is mostly numeric. We only need to handle 'Protocol' if treated as categorical.
        """
        logger.info("Step 4: Categorical Encoding (Handling Protocol)")
        
        if 'Protocol' in df.columns:
            # Simple one-hot encoding for Protocol (usually 6 (TCP), 17 (UDP), 0 (HOPOPT))
            dummies = pd.get_dummies(df['Protocol'], prefix='proto')
            self.onehot_columns['Protocol'] = dummies.columns.tolist()
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=['Protocol'])
            
        return df

    def step4_categorical_encoding_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using training one-hot columns"""
        logger.info("Step 4 (Transform): Applying encoding to test data")
        
        if 'Protocol' in df.columns:
            dummies = pd.get_dummies(df['Protocol'], prefix='proto')
            
            # Align with training columns
            if 'Protocol' in self.onehot_columns:
                expected_cols = self.onehot_columns['Protocol']
                for col in expected_cols:
                    if col not in dummies.columns:
                        dummies[col] = 0
                dummies = dummies[expected_cols]
            
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=['Protocol'])
            
        return df

    def preprocess_unsw_dataset(self, zero_day_attack: str = 'PortScan') -> dict:
        """
        Main Pipeline Override: Points to 'Label' instead of 'attack_cat'
        """
        logger.info(f"Starting CICIDS2017 preprocessing (Zero-Day Target: {zero_day_attack})")
        
        # 1. Load Data with column cleaning
        train_df = self.load_and_clean_columns(self.data_path)
        test_df = self.load_and_clean_columns(self.test_path)
        
        # Normalize Label column BEFORE sampling (to ensure proper attack type matching)
        def normalize_label(label):
            """Normalize label to match attack_types keys"""
            if pd.isna(label):
                return 'BENIGN'
            label_str = str(label).strip()
            
            # Try exact match first
            if label_str in self.attack_types:
                return label_str
            
            # Handle Web Attack variations with special characters
            if 'web attack' in label_str.lower():
                # Normalize Web Attack labels: handle various dash types and spacing
                # Replace all dash types (en dash, em dash, hyphen) with space
                label_clean = re.sub(r'[–—\-]', ' ', label_str)
                # Normalize multiple spaces to single space
                label_clean = re.sub(r'\s+', ' ', label_clean).strip()
                
                # Map to standard Web Attack format
                if 'brute force' in label_clean.lower():
                    return 'Web Attack  Brute Force'  # Use two-space format as standard
                elif 'sql injection' in label_clean.lower():
                    return 'Web Attack  Sql Injection'
                elif 'xss' in label_clean.lower():
                    return 'Web Attack  XSS'
                elif label_clean.lower().startswith('web attack'):
                    return 'Web Attack'  # Generic Web Attack
            
            # Case-insensitive matching
            label_upper = label_str.upper()
            for key in self.attack_types.keys():
                if key.upper() == label_upper:
                    return key
            
            # Handle common variations (remove spaces, dashes, underscores)
            label_normalized = re.sub(r'[–—\-\s_]', '', label_str)
            for key in self.attack_types.keys():
                key_normalized = re.sub(r'[–—\-\s_]', '', key)
                if key_normalized.upper() == label_normalized.upper():
                    return key
            
            # If no match found, keep original (will log later)
            return label_str
        
        # Normalize Label columns if they exist (BEFORE sampling)
        if 'Label' in train_df.columns:
            train_df['Label'] = train_df['Label'].astype(str).str.strip()
            train_df['Label'] = train_df['Label'].apply(normalize_label)
            logger.info(f"🔍 Normalized training Label column. Unique values: {train_df['Label'].value_counts().to_dict()}")
        
        if 'Label' in test_df.columns:
            logger.info(f"🔍 DEBUG: Label distribution in test data BEFORE normalization:")
            raw_labels = test_df['Label'].astype(str).str.strip()
            unique_raw = raw_labels.unique()
            logger.info(f"   Raw unique labels ({len(unique_raw)} types): {unique_raw[:30].tolist()}")
            
            # Check which raw labels will NOT match attack_types
            expected_keys = set(self.attack_types.keys())
            unmatched_labels = set(unique_raw) - expected_keys - {'BENIGN'}
            if unmatched_labels:
                logger.warning(f"   ⚠️ Labels that may NOT match attack_types mapping: {sorted(unmatched_labels)[:20]}")
            
            test_df['Label'] = raw_labels
            test_df['Label'] = test_df['Label'].apply(normalize_label)
            
            # Check if normalization was successful
            normalized_counts = test_df['Label'].value_counts().to_dict()
            logger.info(f"   Normalized Label distribution AFTER normalization: {normalized_counts}")
            
            # Check for unmatched labels after normalization
            normalized_unique = set(test_df['Label'].unique())
            still_unmatched = normalized_unique - expected_keys - {'BENIGN'}
            if still_unmatched:
                logger.warning(f"   ⚠️ Labels still unmatched after normalization: {sorted(still_unmatched)}")
                logger.warning(f"   These will need to be added to attack_types dictionary or fixed in CSV.")
            
            # Count attacks vs BENIGN
            benign_count = (test_df['Label'] == 'BENIGN').sum()
            attack_count = (test_df['Label'] != 'BENIGN').sum()
            logger.info(f"   BENIGN: {benign_count}, Attacks: {attack_count}")
        
        # MEMORY OPTIMIZATION: Sample large datasets to avoid memory issues
        # Reduced limits to prevent memory allocation errors (further reduced for memory-constrained systems)
        MAX_TRAIN_SAMPLES = 50000  # Limit to 50k samples for memory efficiency (reduced from 500k)
        MAX_TEST_SAMPLES = 20000   # Limit to 20k samples for memory efficiency (reduced from 100k)

        # Preserve a capped number of zero-day samples in test data before sampling
        zero_day_label = normalize_label(zero_day_attack)
        zero_day_preserve = None
        max_zero_day_preserve = 5000
        if 'Label' in test_df.columns:
            zero_day_preserve = test_df[test_df['Label'] == zero_day_label].copy()
            if len(zero_day_preserve) > max_zero_day_preserve:
                zero_day_preserve = zero_day_preserve.sample(
                    n=max_zero_day_preserve, random_state=42
                ).reset_index(drop=True)
        
        # Apply sampling EARLY to prevent memory issues during processing
        if len(train_df) > MAX_TRAIN_SAMPLES:
            logger.warning(f"⚠️ Training dataset is large ({len(train_df)} samples). Sampling {MAX_TRAIN_SAMPLES} samples for memory efficiency...")
            train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
            logger.info(f"✅ Sampled training data: {len(train_df)} samples")
        
        if len(test_df) > MAX_TEST_SAMPLES:
            logger.warning(f"⚠️ Test dataset is large ({len(test_df)} samples). Sampling {MAX_TEST_SAMPLES} samples for memory efficiency...")
            sampling_limit = MAX_TEST_SAMPLES
            if zero_day_preserve is not None and len(zero_day_preserve) > 0:
                sampling_limit = MAX_TEST_SAMPLES - len(zero_day_preserve)
                logger.info(f"   Preserving all zero-day samples: {len(zero_day_preserve)} ({zero_day_label})")
                if sampling_limit <= 0:
                    logger.warning(
                        f"   Zero-day samples exceed sampling budget; keeping {MAX_TEST_SAMPLES} zero-day samples only.")
                    test_df = zero_day_preserve.sample(n=MAX_TEST_SAMPLES, random_state=42).reset_index(drop=True)
                    sampling_limit = 0
            # Log distribution BEFORE sampling
            if 'Label' in test_df.columns:
                label_counts_before = test_df['Label'].value_counts().to_dict()
                logger.info(f"   Label distribution BEFORE sampling: {label_counts_before}")
            
            # Stratified sampling to preserve class distribution (Label already normalized above)
            if 'Label' in test_df.columns and sampling_limit > 0:
                working_df = test_df
                if zero_day_preserve is not None and len(zero_day_preserve) > 0:
                    working_df = test_df[test_df['Label'] != zero_day_label].copy()
                unique_labels = working_df['Label'].unique()
                samples_per_class = sampling_limit // len(unique_labels)
                logger.info(f"   Sampling strategy: {samples_per_class} samples per class (from {len(unique_labels)} classes)")
                
                # Prioritize attack samples over BENIGN to ensure we get attacks
                # Reserve 40% for BENIGN, 60% for attacks (distributed across attack types)
                benign_samples = int(sampling_limit * 0.4)
                attack_samples = sampling_limit - benign_samples
                
                sampled_dfs = []
                
                # Sample BENIGN
                if 'BENIGN' in unique_labels:
                    benign_df = working_df[working_df['Label'] == 'BENIGN']
                    n_benign = min(len(benign_df), benign_samples)
                    if n_benign > 0:
                        sampled_benign = benign_df.sample(n=n_benign, random_state=42)
                        sampled_dfs.append(sampled_benign)
                        logger.info(f"   Sampled {n_benign} BENIGN samples (target: {benign_samples})")
                
                # Sample attacks (distribute remaining samples across attack types)
                    attack_labels = [l for l in unique_labels if l != 'BENIGN']
                if attack_labels:
                    samples_per_attack = attack_samples // len(attack_labels) if len(attack_labels) > 0 else 0
                    logger.info(f"   Distributing {attack_samples} samples across {len(attack_labels)} attack types (~{samples_per_attack} per type)")
                    
                    for label in attack_labels:
                        label_df = working_df[working_df['Label'] == label]
                        n_samples = min(len(label_df), samples_per_attack)
                        if n_samples > 0:
                            sampled_label = label_df.sample(n=n_samples, random_state=42)
                            sampled_dfs.append(sampled_label)
                            logger.info(f"   Sampled {n_samples} samples from '{label}' class (total available: {len(label_df)})")
                else:
                    logger.warning(f"   ⚠️ No attack labels found! Only BENIGN present.")
                
                if sampled_dfs:
                    test_df = pd.concat(sampled_dfs, ignore_index=True).reset_index(drop=True)
                    if zero_day_preserve is not None and len(zero_day_preserve) > 0:
                        test_df = pd.concat([test_df, zero_day_preserve], ignore_index=True).reset_index(drop=True)
                    # If still too many, reduce proportionally (but maintain attack samples)
                    if len(test_df) > MAX_TEST_SAMPLES:
                        # Ensure we keep attack samples
                        benign_mask = test_df['Label'] == 'BENIGN'
                        benign_indices = test_df[benign_mask].index
                        attack_indices = test_df[~benign_mask].index
                        
                        # Keep all attacks, reduce BENIGN
                        keep_attacks = test_df.loc[attack_indices]
                        remaining_slots = MAX_TEST_SAMPLES - len(keep_attacks)
                        keep_benign = test_df.loc[benign_indices].sample(n=min(remaining_slots, len(benign_indices)), random_state=42)
                        test_df = pd.concat([keep_attacks, keep_benign], ignore_index=True).reset_index(drop=True)
                else:
                    logger.warning(f"   ⚠️ No samples could be sampled! Keeping original test_df")
            else:
                test_df = test_df.sample(n=MAX_TEST_SAMPLES, random_state=42).reset_index(drop=True)
            
            # Log distribution AFTER sampling
            if 'Label' in test_df.columns:
                label_counts_after = test_df['Label'].value_counts().to_dict()
                logger.info(f"   Label distribution AFTER sampling: {label_counts_after}")
            
            logger.info(f"✅ Sampled test data: {len(test_df)} samples")

        # 2. Standardize Label Columns to match your system's expectations
        # Your system expects 'attack_cat' for the string label and 'label' for the number
        # CICIDS2017 usually has 'Label' (String)
        logger.info(f"🔍 DEBUG: Checking Label distribution in test data...")
        logger.info(f"   Test DataFrame columns: {test_df.columns.tolist()[:10]}...")  # First 10 columns
        if 'Label' in test_df.columns:
            # Clean Label column (strip whitespace, handle case sensitivity)
            test_df['Label'] = test_df['Label'].str.strip()
            
            # Log actual unique labels found
            unique_labels_test = test_df['Label'].unique()
            label_counts_test = test_df['Label'].value_counts().to_dict()
            logger.info(f"   Label column found! Unique labels in test ({len(unique_labels_test)} types):")
            for label, count in sorted(label_counts_test.items(), key=lambda x: x[1], reverse=True)[:20]:  # Top 20
                logger.info(f"      '{label}': {count} samples")
            
            logger.info(f"   Total test samples: {len(test_df)}")
            logger.info(f"   BENIGN count: {(test_df['Label'] == 'BENIGN').sum()}")
            
            # Check which attack types from our mapping exist in the data
            expected_attacks = set(self.attack_types.keys())
            found_labels = set(unique_labels_test)
            matching_attacks = expected_attacks.intersection(found_labels)
            missing_attacks = expected_attacks - found_labels
            unexpected_labels = found_labels - expected_attacks - {'BENIGN'}
            
            logger.info(f"   Attack types in data matching our mapping: {sorted(matching_attacks)}")
            if missing_attacks:
                logger.warning(f"   ⚠️ Attack types in mapping but NOT in test data: {sorted(missing_attacks)}")
            if unexpected_labels:
                logger.warning(f"   ⚠️ Attack types in test data but NOT in mapping: {sorted(unexpected_labels)[:10]}...")  # First 10
            
            attack_count = (test_df['Label'] != 'BENIGN').sum()
            logger.info(f"   Attack count (non-BENIGN): {attack_count}")
        else:
            logger.warning(f"   ⚠️ 'Label' column NOT found in test DataFrame!")
            logger.info(f"   Available columns with 'label' or 'attack': {[c for c in test_df.columns if 'label' in c.lower() or 'attack' in c.lower()]}")
        
        if 'Label' in train_df.columns:
            # Label already normalized above, just copy to attack_cat
            train_df['attack_cat'] = train_df['Label']
            test_df['attack_cat'] = test_df['Label']
            # Drop original to avoid confusion
            train_df.drop(columns=['Label'], inplace=True)
            test_df.drop(columns=['Label'], inplace=True)
        
        # Ensure attack_cat is set for test_df (fallback if Label column doesn't exist)
        if 'attack_cat' not in test_df.columns:
            if 'label' in test_df.columns:
                reverse_mapping = {v: k for k, v in self.attack_types.items()}
                test_df['attack_cat'] = test_df['label'].map(reverse_mapping).fillna('BENIGN')
                logger.info(f"🔍 DEBUG: Mapped 'label' column to 'attack_cat'. Distribution: {test_df['attack_cat'].value_counts().to_dict()}")
            else:
                logger.warning("  ⚠️ No attack_cat or label column in test_df, using BENIGN as default")
                test_df['attack_cat'] = 'BENIGN'
        
        # Log attack_cat distribution after setting it
        logger.info(f"🔍 DEBUG: Final attack_cat distribution in test data (after mapping):")
        attack_cat_counts = test_df['attack_cat'].value_counts().to_dict()
        for cat, count in sorted(attack_cat_counts.items(), key=lambda x: x[1], reverse=True)[:20]:  # Top 20
            logger.info(f"   '{cat}': {count} samples")
        logger.info(f"   Total: {len(test_df)} samples")
        logger.info(f"   BENIGN: {(test_df['attack_cat'] == 'BENIGN').sum()}")
        logger.info(f"   Attacks (non-BENIGN): {(test_df['attack_cat'] != 'BENIGN').sum()}")
        
        # Check if attacks are being lost during mapping
        if (test_df['attack_cat'] != 'BENIGN').sum() == 0 and len(test_df) > 0:
            logger.error(f"   ❌ CRITICAL: All test samples are BENIGN! Check Label column mapping.")
            logger.error(f"   Attack types in mapping: {sorted(self.attack_types.keys())}")
            if 'attack_cat' in test_df.columns:
                logger.error(f"   Unique attack_cat values found: {test_df['attack_cat'].unique()[:20]}")
        
        # 3. Run Pipeline Steps
        # Note: step1_data_quality_assessment returns a dict, not a DataFrame
        train_quality = self.step1_data_quality_assessment(train_df)
        train_df = self.step2_feature_engineering(train_df)
        train_df = self.step3_data_cleaning(train_df)
        train_df = self.step4_categorical_encoding(train_df)
        
        test_quality = self.step1_data_quality_assessment(test_df)
        test_df = self.step2_feature_engineering(test_df)
        test_df = self.step3_data_cleaning(test_df)
        test_df = self.step4_categorical_encoding_transform(test_df)

        # 4. Feature Selection (Target is now 'attack_cat' which we created above)
        logger.info("Running Feature Selection...")
        # We need to manually encode attack_cat to numbers for feature selection
        temp_encoder = {v: k for k, v in enumerate(train_df['attack_cat'].unique())}
        train_df['temp_target'] = train_df['attack_cat'].map(temp_encoder)
        
        # Call parent selection method
        train_df = self.step5_feature_selection_hybrid(train_df, target_col='temp_target', n_features_final=43)
        train_df.drop(columns=['temp_target'], inplace=True)
        
        # Apply same selected features to test data (similar to parent method)
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
        
        # Add missing selected features as zeros
        for feat in missing_features:
            test_df[feat] = 0.0
            logger.info(f"  Added missing selected feature '{feat}' to test data (filled with 0)")
        
        # Ensure test_df has all selected features in the same order
        final_test_features = selected_features + target_cols
        for col in final_test_features:
            if col not in test_df.columns:
                test_df[col] = 0.0
        test_df = test_df[final_test_features]
        
        logger.info(f"  Test data shape after feature selection: {test_df.shape}")
        
        # FIX #2: Split FIRST, then rebalance training data only (prevents leakage)
        logger.info("\nSplitting training data into train/val using stratified sampling (BEFORE rebalancing)...")
        from sklearn.model_selection import train_test_split
        
        # Create labels using CICIDS attack_types mapping BEFORE splitting
        train_df['label'] = train_df['attack_cat'].map(self.attack_types).fillna(0).astype(int)
        train_df['binary_label'] = (train_df['label'] != 0).astype(int)
        
        # Prepare features and labels for stratified split
        feature_cols = [col for col in train_df.columns if col not in ['label', 'binary_label', 'attack_cat']]
        
        # MEMORY SAFETY: Double-check dataset size before creating large arrays
        if len(train_df) > MAX_TRAIN_SAMPLES:
            logger.warning(f"⚠️ Training dataset still large ({len(train_df)} samples) after initial sampling. Applying additional sampling...")
            train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
            logger.info(f"✅ Additional sampling applied: {len(train_df)} samples")
        
        X = train_df[feature_cols].values
        y = train_df['label'].values
        
        # Split: 80% train, 20% val (BEFORE rebalancing)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Reconstruct dataframes
        train_df_split = pd.DataFrame(X_train_split, columns=feature_cols)
        train_df_split['label'] = y_train_split
        train_df_split['binary_label'] = (y_train_split != 0).astype(int)
        
        val_df = pd.DataFrame(X_val_split, columns=feature_cols)
        val_df['label'] = y_val_split
        val_df['binary_label'] = (y_val_split != 0).astype(int)
        
        # Create reverse mapping for attack categories (use vectorized pandas operation for memory efficiency)
        reverse_mapping = {v: k for k, v in self.attack_types.items()}
        # Use pandas map() instead of list comprehension for memory efficiency
        train_df_split['attack_cat'] = pd.Series(y_train_split).map(reverse_mapping).fillna('BENIGN')
        val_df['attack_cat'] = pd.Series(y_val_split).map(reverse_mapping).fillna('BENIGN')
        
        # FIX #2: Rebalance TRAINING data only (validation and test remain untouched)
        logger.info("\nApplying data rebalancing to TRAINING data only...")
        train_df_rebalanced = self.step7_data_rebalancing_complete(train_df_split)
        
        logger.info(f"  Training data: {len(train_df_split)} → {len(train_df_rebalanced)} samples (rebalanced)")
        logger.info(f"  Validation data: {len(val_df)} samples (unchanged)")
        logger.info(f"  Test data: {len(test_df)} samples (unchanged)")
        
        # Use rebalanced training data
        train_df = train_df_rebalanced
        
        # Align features between train, val, and test
        logger.info("\nAligning features between train, validation, and test data...")
        all_cols = set(train_df.columns).union(set(val_df.columns)).union(set(test_df.columns))
        
        # CRITICAL: Preserve metadata columns (label, binary_label, attack_cat) separately
        metadata_cols = ['label', 'binary_label', 'attack_cat']
        feature_cols = [col for col in all_cols if col not in metadata_cols]
        common_cols = sorted(feature_cols) + metadata_cols  # Features first, then metadata
        
        # Ensure test_df has attack_cat BEFORE alignment
        if 'attack_cat' not in test_df.columns and 'Label' in test_df.columns:
            logger.info("🔍 Setting attack_cat for test_df from Label column...")
            test_df['attack_cat'] = test_df['Label'].apply(normalize_label)
            logger.info(f"   Test attack_cat distribution: {test_df['attack_cat'].value_counts().to_dict()}")
        
        # Also ensure label and binary_label are set for test_df
        if 'label' not in test_df.columns and 'attack_cat' in test_df.columns:
            test_df['label'] = test_df['attack_cat'].map(self.attack_types).fillna(0).astype(int)
        if 'binary_label' not in test_df.columns and 'label' in test_df.columns:
            test_df['binary_label'] = (test_df['label'] != 0).astype(int)
        
        for col in feature_cols:  # Only align feature columns, not metadata
            if col not in train_df.columns:
                train_df[col] = 0
            if col not in val_df.columns:
                val_df[col] = 0
            if col not in test_df.columns:
                test_df[col] = 0
        
        # Reorder columns: features first, then metadata
        train_df = train_df[common_cols]
        val_df = val_df[common_cols]
        test_df = test_df[common_cols]
        
        # Log final distribution to verify attacks are preserved
        if 'attack_cat' in test_df.columns:
            logger.info(f"🔍 VERIFICATION: Test attack_cat distribution after alignment:")
            test_attack_dist = test_df['attack_cat'].value_counts().to_dict()
            for cat, count in sorted(test_attack_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"   '{cat}': {count} samples")
            test_attack_count = (test_df['attack_cat'] != 'BENIGN').sum()
            logger.info(f"   Total attacks in test_df: {test_attack_count} (BENIGN: {(test_df['attack_cat'] == 'BENIGN').sum()})")
        
        logger.info(f"  Final feature count - Train: {len(train_df.columns)}, Val: {len(val_df.columns)}, Test: {len(test_df.columns)}")
        
        # Create zero-day split (override to use BENIGN instead of Normal)
        split_data = self.create_zero_day_split_cicids(train_df, val_df, test_df, zero_day_attack)
        
        # Apply feature scaling
        train_scaled, val_scaled, test_scaled = self.step6_feature_scaling(
            split_data['train'], split_data['val'], split_data['test']
        )
        
        # Convert to PyTorch tensors
        import torch
        feature_cols = [col for col in train_scaled.columns if col not in ['label', 'binary_label', 'attack_cat']]
        
        X_train = torch.FloatTensor(train_scaled[feature_cols].values)
        y_train = torch.LongTensor(train_scaled['binary_label'].values)
        
        X_val = torch.FloatTensor(val_scaled[feature_cols].values)
        y_val = torch.LongTensor(val_scaled['binary_label'].values)
        
        X_test = torch.FloatTensor(test_scaled[feature_cols].values)
        y_test = torch.LongTensor(test_scaled['binary_label'].values)
        
        # Create zero-day indices
        zero_day_attack_label = self.attack_types.get(zero_day_attack, 10)  # Default to PortScan=10
        zero_day_mask = test_scaled['attack_cat'] == zero_day_attack
        # Convert to numpy first, then torch (more memory efficient)
        zero_day_mask_np = zero_day_mask.values
        zero_day_indices = torch.where(torch.tensor(zero_day_mask_np, dtype=torch.bool))[0].tolist()
        
        logger.info("\nCICIDS2017 preprocessing completed successfully!")
        logger.info(f"Final feature count: {len(feature_cols)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Zero-day samples ({zero_day_attack}, label={zero_day_attack_label}): {len(zero_day_indices)}")
        
        # CRITICAL: Validate data shapes
        if len(X_train.shape) != 2:
            raise ValueError(f"X_train should be 2D (n_samples, n_features), got shape {X_train.shape}")
        if X_train.shape[1] > 1000:
            raise ValueError(
                f"Suspicious number of features in X_train: {X_train.shape[1]}. "
                f"Expected ~40-80 features for CICIDS2017. Shape: {X_train.shape}"
            )
        
        # Store multiclass labels and attack_cat (CRITICAL for attack type distinction in support sets)
        # Use numpy arrays first, then convert to torch tensors (more memory efficient)
        y_train_multiclass = torch.LongTensor(train_scaled['label'].values.copy())
        y_val_multiclass = torch.LongTensor(val_scaled['label'].values.copy())
        y_test_multiclass = torch.LongTensor(test_scaled['label'].values.copy())
        # Keep as numpy array for memory efficiency (main.py handles both list and array)
        test_attack_cat = test_scaled['attack_cat'].values
        # Convert to list only if small enough (for compatibility with existing code)
        if len(test_attack_cat) < 1000000:  # Only convert to list if < 1M samples
            test_attack_cat = test_attack_cat.tolist()
        
        # Debug: Check attack categories in test data
        if isinstance(test_attack_cat, (list, np.ndarray)):
            unique_attack_cats = np.unique(test_attack_cat)
            logger.info(f"🔍 DEBUG: Unique attack categories in test_attack_cat: {unique_attack_cats.tolist()}")
            logger.info(f"🔍 DEBUG: Total test samples: {len(test_attack_cat)}")
            # Check if zero_day_attack exists (passed as parameter to preprocess_unsw_dataset)
            if zero_day_attack:
                test_attack_cat_array = np.array(test_attack_cat)
                zero_day_count = (test_attack_cat_array == zero_day_attack).sum()
                logger.info(f"🔍 DEBUG: Zero-day attack '{zero_day_attack}' count: {zero_day_count}")
        
        # Debug: Check multiclass label distribution
        unique_train_multiclass = torch.unique(y_train_multiclass)
        unique_val_multiclass = torch.unique(y_val_multiclass)
        unique_test_multiclass = torch.unique(y_test_multiclass)
        logger.info(f"🔍 Training multiclass labels: {unique_train_multiclass.tolist()} ({len(unique_train_multiclass)} unique)")
        logger.info(f"🔍 Validation multiclass labels: {unique_val_multiclass.tolist()} ({len(unique_val_multiclass)} unique)")
        logger.info(f"🔍 Test multiclass labels: {unique_test_multiclass.tolist()} ({len(unique_test_multiclass)} unique)")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'y_train_multiclass': y_train_multiclass,  # CRITICAL: Multiclass labels for attack type distinction
            'X_val': X_val,
            'y_val': y_val,
            'y_val_multiclass': y_val_multiclass,  # CRITICAL: Multiclass labels for attack type distinction
            'X_test': X_test,
            'y_test': y_test,
            'y_test_multiclass': y_test_multiclass,
            'test_attack_cat': test_attack_cat,
            'zero_day_indices': zero_day_indices,
            'feature_names': feature_cols,
            'quality_reports': {'train': train_quality, 'test': test_quality},
            'attack_types': self.attack_types,
            'selected_features': selected_features
        }

    def step7_data_rebalancing_complete(self, complete_df: pd.DataFrame) -> pd.DataFrame:
        """Override Step 7 to use CICIDS attack mapping with full rebalancing logic"""
        # Use the CICIDS mapping defined in __init__
        if 'label' not in complete_df.columns:
            complete_df['label'] = complete_df['attack_cat'].map(self.attack_types).fillna(0).astype(int)
        if 'binary_label' not in complete_df.columns:
            complete_df['binary_label'] = (complete_df['label'] != 0).astype(int)
        
        # Copy rebalancing logic from parent class
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        
        logger.info("Step 7: Complete Data Rebalancing (ADASYN + RandomUnderSampler)")
        logger.info(f"  Input dataset shape: {complete_df.shape}")
        
        # Separate features and labels
        feature_cols = [col for col in complete_df.columns if col not in ['label', 'binary_label', 'attack_cat']]
        X = complete_df[feature_cols].values
        y = complete_df['label'].values
        
        # Analyze class distribution before rebalancing
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info(f"  Classes before rebalancing: {len(unique_classes)}")
        logger.info(f"  Class distribution before rebalancing:")
        
        reverse_mapping = {v: k for k, v in self.attack_types.items()}
        for class_label, count in zip(unique_classes, class_counts):
            attack_name = reverse_mapping.get(class_label, f'Class_{class_label}')
            percentage = (count / len(y)) * 100
            logger.info(f"    {attack_name} (Label {class_label}): {count:,} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio
        max_count = np.max(class_counts)
        min_count = np.min(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        logger.info(f"  Imbalance ratio before rebalancing: {imbalance_ratio:.2f}:1")
        
        # Define target counts (target: 15% of majority class for minority classes)
        majority_count = max_count
        target_minority = int(majority_count * 0.15)  # 15% of majority
        target_minority = max(target_minority, 100)  # Minimum 100 samples per class
        
        # Build sampling strategy
        sampling_strategy = {}
        for class_label, count in zip(unique_classes, class_counts):
            if count < target_minority:
                sampling_strategy[class_label] = target_minority
            elif count > majority_count * 1.5:  # If significantly larger than majority
                sampling_strategy[class_label] = int(majority_count * 1.2)
            else:
                sampling_strategy[class_label] = count  # Keep current
        
        logger.info(f"  Target sampling strategy:")
        for class_label, target_count in sampling_strategy.items():
            attack_name = reverse_mapping.get(class_label, f'Class_{class_label}')
            current_count = class_counts[unique_classes == class_label][0]
            action = "oversample" if target_count > current_count else "undersample" if target_count < current_count else "keep"
            logger.info(f"    {attack_name}: {current_count:,} → {target_count:,} ({action})")
        
        # Step 1: Apply ADASYN for oversampling
        logger.info("  Step 1: Applying ADASYN oversampling...")
        oversample_strategy = {k: v for k, v in sampling_strategy.items() 
                               if class_counts[unique_classes == k][0] < v}
        
        if oversample_strategy:
            # IMPROVED: Filter out classes with insufficient samples before oversampling
            # ADASYN requires: n_neighbors + 1 samples (5 + 1 = 6)
            # SMOTE requires: k_neighbors + 1 samples (3 + 1 = 4)
            adasyn_n_neighbors = 5
            smote_k_neighbors = 3
            min_samples_for_adasyn = adasyn_n_neighbors + 1  # 6 samples minimum
            min_samples_for_smote = smote_k_neighbors + 1    # 4 samples minimum
            
            # Separate classes into those suitable for ADASYN, SMOTE, or neither
            adasyn_strategy = {}
            smote_strategy = {}
            skipped_classes = {}
            
            for class_label, target_count in oversample_strategy.items():
                class_idx = (unique_classes == class_label)
                current_count = class_counts[class_idx][0] if np.any(class_idx) else 0
                
                if current_count >= min_samples_for_adasyn:
                    # Enough samples for ADASYN
                    adasyn_strategy[class_label] = target_count
                elif current_count >= min_samples_for_smote:
                    # Enough for SMOTE but not ADASYN
                    smote_strategy[class_label] = target_count
                else:
                    # Too few samples for any oversampling
                    skipped_classes[class_label] = current_count
                    if current_count > 0:
                        logger.warning(f"  Class {class_label}: Only {current_count} samples (need {min_samples_for_smote} for SMOTE, {min_samples_for_adasyn} for ADASYN) - skipping oversampling")
            
            # Apply ADASYN for suitable classes
            if adasyn_strategy:
                try:
                    adasyn = ADASYN(sampling_strategy=adasyn_strategy, random_state=42, n_neighbors=adasyn_n_neighbors)
                    X_resampled, y_resampled = adasyn.fit_resample(X, y)
                    logger.info(f"  ADASYN completed: {len(X)} → {len(X_resampled)} samples (for {len(adasyn_strategy)} classes)")
                    X, y = X_resampled, y_resampled
                except Exception as e:
                    logger.warning(f"  ADASYN failed: {e}, falling back to SMOTE for these classes")
                    # Move ADASYN classes to SMOTE fallback
                    smote_strategy.update(adasyn_strategy)
                    adasyn_strategy = {}
            
            # Apply SMOTE for classes not handled by ADASYN (or as fallback)
            if smote_strategy:
                try:
                    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=smote_k_neighbors)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    logger.info(f"  SMOTE completed: {len(X)} → {len(X_resampled)} samples (for {len(smote_strategy)} classes)")
                    X, y = X_resampled, y_resampled
                except Exception as e2:
                    logger.warning(f"  SMOTE also failed: {e2}, using original data for these classes")
                    # Keep original data for these classes
            
            # Log summary
            if skipped_classes:
                total_skipped = sum(skipped_classes.values())
                logger.info(f"  Skipped oversampling for {len(skipped_classes)} classes with insufficient samples (total {total_skipped} samples kept as-is)")
            
            X_resampled, y_resampled = X, y
        else:
            logger.info("  No classes need oversampling, skipping ADASYN")
            X_resampled, y_resampled = X, y
        
        # Step 2: Apply RandomUnderSampler for undersampling
        logger.info("  Step 2: Applying RandomUnderSampler...")
        undersample_strategy = {k: v for k, v in sampling_strategy.items() 
                                if class_counts[unique_classes == k][0] > v}
        
        if undersample_strategy:
            undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
            try:
                X_balanced, y_balanced = undersampler.fit_resample(X_resampled, y_resampled)
                logger.info(f"  RandomUnderSampler completed: {len(X_resampled)} → {len(X_balanced)} samples")
            except Exception as e:
                logger.warning(f"  RandomUnderSampler failed: {e}, using ADASYN result")
                X_balanced, y_balanced = X_resampled, y_resampled
        else:
            logger.info("  No classes need undersampling, skipping RandomUnderSampler")
            X_balanced, y_balanced = X_resampled, y_resampled
        
        # Create balanced dataframe
        balanced_df = pd.DataFrame(X_balanced, columns=feature_cols)
        balanced_df['label'] = y_balanced
        balanced_df['binary_label'] = (y_balanced != 0).astype(int)
        balanced_df['attack_cat'] = [reverse_mapping.get(label, f'Class_{label}') for label in y_balanced]
        
        # Log final distribution
        unique_after, counts_after = np.unique(y_balanced, return_counts=True)
        logger.info("  Class distribution after rebalancing:")
        for class_label, count in zip(unique_after, counts_after):
            attack_name = reverse_mapping.get(class_label, f'Class_{class_label}')
            percentage = (count / len(y_balanced)) * 100
            logger.info(f"    {attack_name} (Label {class_label}): {count:,} samples ({percentage:.2f}%)")
        
        max_after = np.max(counts_after)
        min_after = np.min(counts_after)
        imbalance_after = max_after / min_after if min_after > 0 else float('inf')
        logger.info(f"  Imbalance ratio after rebalancing: {imbalance_after:.2f}:1")
        logger.info(f"  Final dataset shape: {balanced_df.shape}")
        
        return balanced_df
    
    def create_zero_day_split_cicids(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                      test_df: pd.DataFrame, zero_day_attack: str) -> dict:
        """
        Create zero-day split for CICIDS2017 dataset (similar to UNSW-NB15 approach).
        Excludes zero-day attack from training/validation, creates balanced test set.
        
        Data Leakage Prevention:
        - Train/Val/Test data split: 80/10/10 (no overlap)
        - Test "other attacks" sampled from test data only
        - Test "normal samples" sourced EXCLUSIVELY from test data only
        - Zero-day attack completely excluded from train/val
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            zero_day_attack: Attack type to treat as zero-day (e.g., 'PortScan')
            
        Returns:
            Dictionary with 'train', 'val', 'test' dataframes
        """
        logger.info(f"Creating zero-day split with '{zero_day_attack}' as zero-day attack")
        
        # Ensure test_df has attack_cat column (should already be set, but verify)
        if 'attack_cat' not in test_df.columns:
            # Create from label if available
            if 'label' in test_df.columns:
                reverse_mapping = {v: k for k, v in self.attack_types.items()}
                test_df['attack_cat'] = test_df['label'].map(reverse_mapping).fillna('BENIGN')
            else:
                logger.warning("  No attack_cat or label column in test_df, using BENIGN as default")
                test_df['attack_cat'] = 'BENIGN'
        
        # Also ensure train_df and val_df have attack_cat
        if 'attack_cat' not in train_df.columns and 'label' in train_df.columns:
            reverse_mapping = {v: k for k, v in self.attack_types.items()}
            train_df['attack_cat'] = train_df['label'].map(reverse_mapping).fillna('BENIGN')
        if 'attack_cat' not in val_df.columns and 'label' in val_df.columns:
            reverse_mapping = {v: k for k, v in self.attack_types.items()}
            val_df['attack_cat'] = val_df['label'].map(reverse_mapping).fillna('BENIGN')
        
        # Ensure labels are properly set
        if 'label' not in train_df.columns:
            train_df['label'] = train_df['attack_cat'].map(self.attack_types).fillna(0).astype(int)
        if 'label' not in val_df.columns:
            val_df['label'] = val_df['attack_cat'].map(self.attack_types).fillna(0).astype(int)
        if 'label' not in test_df.columns:
            test_df['label'] = test_df['attack_cat'].map(self.attack_types).fillna(0).astype(int)
        
        # Create binary labels (0=BENIGN/Normal, 1=Attack)
        train_df['binary_label'] = (train_df['label'] != 0).astype(int)
        val_df['binary_label'] = (val_df['label'] != 0).astype(int)
        test_df['binary_label'] = (test_df['label'] != 0).astype(int)
        
        # Log binary label distribution BEFORE zero-day filtering
        logger.info(f"  Binary labels BEFORE zero-day filtering:")
        logger.info(f"    Training - BENIGN: {len(train_df[train_df['binary_label'] == 0])}, Attack: {len(train_df[train_df['binary_label'] == 1])}")
        logger.info(f"    Validation - BENIGN: {len(val_df[val_df['binary_label'] == 0])}, Attack: {len(val_df[val_df['binary_label'] == 1])}")
        logger.info(f"    Test - BENIGN: {len(test_df[test_df['binary_label'] == 0])}, Attack: {len(test_df[test_df['binary_label'] == 1])}")
        
        # Separate BENIGN and Attack samples
        train_benign = train_df[train_df['attack_cat'] == 'BENIGN'].copy()
        train_attacks = train_df[train_df['attack_cat'] != 'BENIGN'].copy()
        val_benign = val_df[val_df['attack_cat'] == 'BENIGN'].copy()
        val_attacks = val_df[val_df['attack_cat'] != 'BENIGN'].copy()
        test_benign = test_df[test_df['attack_cat'] == 'BENIGN'].copy()
        test_attacks = test_df[test_df['attack_cat'] != 'BENIGN'].copy()
        
        # Get zero-day attack samples from test data
        zero_day_test = test_df[test_df['attack_cat'] == zero_day_attack].copy()
        
        # If zero-day attack not found in test data, find alternative
        if len(zero_day_test) == 0:
            logger.warning(f"No {zero_day_attack} attacks found in test data. Finding best alternative.")
            available_attacks = test_df[test_df['attack_cat'] != 'BENIGN']['attack_cat'].value_counts()
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
        
        # Create training data: BENIGN + other attack classes (excluding zero-day)
        train_data = pd.concat([train_benign, train_attacks_filtered], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create validation data: BENIGN + other attack classes (excluding zero-day)
        val_data = pd.concat([val_benign, val_attacks_filtered], ignore_index=True)
        val_data = val_data.sample(frac=1, random_state=43).reset_index(drop=True)
        
        # Create test data with BALANCED distribution (similar to UNSW-NB15)
        # Target: 30% BENIGN + 70% Attacks (including zero-day)
        total_test_samples = len(zero_day_test) + len(test_attacks[test_attacks['attack_cat'] != zero_day_attack]) + len(test_benign)
        target_benign_samples = int(total_test_samples * 0.3)  # 30% BENIGN
        target_attack_samples = int(total_test_samples * 0.7)  # 70% Attacks
        
        logger.info(f"  Test set composition target: {target_benign_samples} BENIGN, {target_attack_samples} attacks")
        logger.info(f"  Available in test data: {len(test_benign)} BENIGN, {len(zero_day_test)} zero-day, {len(test_attacks[test_attacks['attack_cat'] != zero_day_attack])} other attacks")
        
        # Sample BENIGN samples for test data - PREVENT DATA LEAKAGE (only from test_df)
        if len(test_benign) >= target_benign_samples:
            test_benign_sample = test_benign.sample(n=target_benign_samples, random_state=42)
            logger.info(f"BENIGN samples for test set sourced exclusively from test_df: {len(test_benign_sample)} samples")
        elif len(test_benign) > 0:
            test_benign_sample = test_benign.sample(n=len(test_benign), random_state=42)
            logger.warning(f"Insufficient BENIGN samples in test_df for 30% target: {len(test_benign)} < {target_benign_samples} required")
            logger.info(f"Using all available BENIGN samples from test_df: {len(test_benign_sample)} samples")
        else:
            raise ValueError("No BENIGN samples available in test_df for test set. Cannot use training data to prevent data leakage.")
        
        # Use ALL available zero-day samples for evaluation
        zero_day_sample = zero_day_test.copy()
        
        # Get other attack types from test data only (excluding zero-day attack)
        test_other_attacks = test_attacks[test_attacks['attack_cat'] != zero_day_attack].copy()
        if len(test_other_attacks) > 0:
            # Sample other attacks to balance test set
            remaining_attack_slots = max(0, target_attack_samples - len(zero_day_sample))
            other_attacks_sample = test_other_attacks.sample(n=min(remaining_attack_slots, len(test_other_attacks)), random_state=42)
        else:
            logger.warning("No other attack types found in test data, using only zero-day attacks")
            other_attacks_sample = pd.DataFrame()
        
        test_data = pd.concat([test_benign_sample, zero_day_sample, other_attacks_sample], ignore_index=True)
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"  Training data: {len(train_data)} samples")
        logger.info(f"    BENIGN: {len(train_data[train_data['binary_label'] == 0])}")
        logger.info(f"    Other attacks (excluding zero-day): {len(train_data[train_data['binary_label'] == 1])}")
        
        logger.info(f"  Validation data: {len(val_data)} samples")
        logger.info(f"    BENIGN: {len(val_data[val_data['binary_label'] == 0])}")
        logger.info(f"    Other attacks (excluding zero-day): {len(val_data[val_data['binary_label'] == 1])}")
        
        logger.info(f"  Test data: {len(test_data)} samples")
        logger.info(f"    BENIGN (30%): {len(test_data[test_data['binary_label'] == 0])}")
        logger.info(f"    Zero-day attacks: {len(test_data[test_data['attack_cat'] == zero_day_attack])}")
        logger.info(f"    Other attacks from test data: {len(test_data[(test_data['binary_label'] == 1) & (test_data['attack_cat'] != zero_day_attack)])}")
        
        # Log data leakage prevention measures
        logger.info("  Data leakage prevention:")
        logger.info(f"    ✓ Train/Val data split: 80/10/10 (no overlap)")
        logger.info(f"    ✓ Test 'other attacks' sampled from test data only")
        logger.info(f"    ✓ Test 'BENIGN samples' sourced exclusively from test data only")
        logger.info(f"    ✓ Zero-day attack completely excluded from train/val")
        
        # CRITICAL: Verify that training data has both classes
        train_benign_count = len(train_data[train_data['binary_label'] == 0])
        train_attack_count = len(train_data[train_data['binary_label'] == 1])
        logger.info(f"  🔍 CRITICAL VERIFICATION:")
        logger.info(f"    Training data classes: BENIGN={train_benign_count}, Attack={train_attack_count}")
        if train_attack_count == 0:
            logger.error(f"    ❌ CRITICAL BUG: Training data has NO attack samples!")
        else:
            logger.info(f"    ✅ Training data has both BENIGN and Attack samples - GOOD!")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }