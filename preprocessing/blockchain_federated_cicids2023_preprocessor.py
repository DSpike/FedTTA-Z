#!/usr/bin/env python3
"""
Blockchain Federated Learning - CICIDS2023 Preprocessor
Full 7-Step Pipeline for Zero-Day Attack Detection (matching UNSW quality)

IMPROVEMENTS FROM ORIGINAL:
✅ Step 2: CICIDS-specific feature engineering (was SKIPPED)
✅ Step 5: Hybrid IG+RF feature selection (was SKIPPED)
✅ Step 7: SMOTE data rebalancing (was only stratified split)
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import logging
import warnings
import os
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICIDS2023Preprocessor(UNSWPreprocessor):
    """
    CICIDS2023 Preprocessor with Full 7-Step Pipeline

    Pipeline:
    1. Data Quality Assessment ✅
    2. Feature Engineering (CICIDS-specific) ✅ FIXED
    3. Data Cleaning ✅
    4. Categorical Encoding ✅
    5. Feature Selection (IG+RF hybrid) ✅ FIXED
    6. Feature Scaling ✅
    7. Data Rebalancing (SMOTE) ✅ FIXED
    """

    def __init__(self, data_path: str = "CICIoT2023_training.csv", test_path: str = "CICIoT2023_testing.csv"):
        super().__init__(data_path, test_path)

        # CICIoT2023 attack types (34 unique labels)
        self.attack_types = {
            'BenignTraffic': 0,
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
            'BENIGN': 0,  # Alias
        }
        logger.info("✅ CICIoT2023 Preprocessor initialized (FULL PIPELINE)")

    def load_and_clean_columns(self, path):
        """Memory-optimized CSV loading with dtype optimization"""
        logger.info(f"Loading CICIDS2023 CSV: {path}")
        try:
            # Read sample to determine dtypes
            sample_df = pd.read_csv(path, nrows=1000)
            logger.info(f"   Sample shape: {sample_df.shape}")

            # Optimize dtypes
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

            # Check file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"   File size: {file_size_mb:.1f} MB")

            # Load with chunking for large files
            if file_size_mb > 500:
                logger.info(f"   Using chunked reading...")
                chunks = []
                for chunk in pd.read_csv(path, chunksize=50000, dtype=dtype_dict, low_memory=False):
                    chunk.columns = chunk.columns.str.strip()
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(path, dtype=dtype_dict, low_memory=False)
                df.columns = df.columns.str.strip()

            logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")
            raise

    def step2_feature_engineering_cicids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: CICIDS2023-Specific Feature Engineering
        Adds network behavior features based on available CICIDS columns
        """
        logger.info("Step 2: CICIDS2023 Feature Engineering (FIXED)")

        df_engineered = df.copy()
        features_added = 0

        # Common CICIDS2023 column names (check which exist)
        # Reference: https://www.unb.ca/cic/datasets/iotdataset-2023.html

        # 1. Packet rate features
        if 'flow_duration' in df.columns and 'tot_fwd_pkts' in df.columns:
            df_engineered['fwd_packet_rate'] = df['tot_fwd_pkts'] / (df['flow_duration'] + 1e-6)
            features_added += 1
            logger.info("   ✅ Added: fwd_packet_rate")

        if 'flow_duration' in df.columns and 'tot_bwd_pkts' in df.columns:
            df_engineered['bwd_packet_rate'] = df['tot_bwd_pkts'] / (df['flow_duration'] + 1e-6)
            features_added += 1
            logger.info("   ✅ Added: bwd_packet_rate")

        # 2. Bytes per packet features
        if 'fwd_pkt_len_tot' in df.columns and 'tot_fwd_pkts' in df.columns:
            df_engineered['fwd_bytes_per_packet'] = df['fwd_pkt_len_tot'] / (df['tot_fwd_pkts'] + 1)
            features_added += 1
            logger.info("   ✅ Added: fwd_bytes_per_packet")

        if 'bwd_pkt_len_tot' in df.columns and 'tot_bwd_pkts' in df.columns:
            df_engineered['bwd_bytes_per_packet'] = df['bwd_pkt_len_tot'] / (df['tot_bwd_pkts'] + 1)
            features_added += 1
            logger.info("   ✅ Added: bwd_bytes_per_packet")

        # 3. Byte ratio (forward vs backward)
        if 'fwd_pkt_len_tot' in df.columns and 'bwd_pkt_len_tot' in df.columns:
            total_bytes = df['fwd_pkt_len_tot'] + df['bwd_pkt_len_tot']
            df_engineered['fwd_byte_ratio'] = df['fwd_pkt_len_tot'] / (total_bytes + 1)
            features_added += 1
            logger.info("   ✅ Added: fwd_byte_ratio")

        # 4. Packet ratio (forward vs backward)
        if 'tot_fwd_pkts' in df.columns and 'tot_bwd_pkts' in df.columns:
            total_pkts = df['tot_fwd_pkts'] + df['tot_bwd_pkts']
            df_engineered['fwd_packet_ratio'] = df['tot_fwd_pkts'] / (total_pkts + 1)
            features_added += 1
            logger.info("   ✅ Added: fwd_packet_ratio")

        # 5. Flow bytes per second
        if 'flow_byts_s' not in df.columns and 'flow_duration' in df.columns:
            if 'fwd_pkt_len_tot' in df.columns and 'bwd_pkt_len_tot' in df.columns:
                total_bytes = df['fwd_pkt_len_tot'] + df['bwd_pkt_len_tot']
                df_engineered['flow_bytes_per_sec'] = total_bytes / (df['flow_duration'] + 1e-6)
                features_added += 1
                logger.info("   ✅ Added: flow_bytes_per_sec")

        # Replace infinite values from division
        df_engineered.replace([np.inf, -np.inf], 0, inplace=True)

        logger.info(f"   Total features added: {features_added}")
        logger.info(f"   Shape: {df.shape} → {df_engineered.shape}")

        return df_engineered

    def preprocess_unsw_dataset(self, zero_day_attack: str = 'DDoS-HTTP_Flood') -> dict:
        """
        FULL 7-STEP PIPELINE for CICIDS2023 (FIXED VERSION)
        """
        logger.info(f"🔧 Starting CICIDS2023 FULL preprocessing pipeline")
        logger.info(f"   Zero-Day Target: {zero_day_attack}")

        label_column = 'label'  # CICIoT2023 uses lowercase

        # === STEP 0: Load Data ===
        train_df = self.load_and_clean_columns(self.data_path)
        test_df = self.load_and_clean_columns(self.test_path)

        if label_column not in train_df.columns:
            raise ValueError(f"Label column '{label_column}' not found!")

        # Normalize labels
        def normalize_label(label):
            if pd.isna(label):
                return 'BenignTraffic'
            label_str = str(label).strip()
            if label_str in self.attack_types:
                return label_str
            if 'benign' in label_str.lower() or 'normal' in label_str.lower():
                return 'BenignTraffic'
            logger.warning(f"⚠️  Unknown attack: '{label_str}'")
            return label_str

        logger.info("Normalizing labels...")
        train_df[label_column] = train_df[label_column].apply(normalize_label)
        test_df[label_column] = test_df[label_column].apply(normalize_label)

        # === STEP 1: Data Quality Assessment ===
        train_quality = self.step1_data_quality_assessment(train_df)
        test_quality = self.step1_data_quality_assessment(test_df)

        # === STEP 2: Feature Engineering (CICIDS-specific) ===
        train_df = self.step2_feature_engineering_cicids(train_df)
        test_df = self.step2_feature_engineering_cicids(test_df)

        # === STEP 3: Data Cleaning ===
        train_df = self.step3_data_cleaning(train_df)
        test_df = self.step3_data_cleaning(test_df)

        # === STEP 4: Categorical Encoding ===
        train_df = self.step4_categorical_encoding(train_df)
        test_df = self.step4_categorical_encoding_transform(test_df)

        # === Prepare labels ===
        train_df['label_int'] = train_df[label_column].map(self.attack_types)
        test_df['label_int'] = test_df[label_column].map(self.attack_types)

        # Drop unmapped labels
        train_df = train_df.dropna(subset=['label_int'])
        test_df = test_df.dropna(subset=['label_int'])

        train_df['binary_label'] = (train_df['label_int'] != 0).astype(int)
        test_df['binary_label'] = (test_df['label_int'] != 0).astype(int)
        train_df['attack_cat'] = train_df[label_column]
        test_df['attack_cat'] = test_df[label_column]

        # Get feature columns
        exclude_cols = [label_column, 'label_int', 'binary_label', 'attack_cat']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        logger.info(f"Features before selection: {len(feature_cols)}")

        # Convert to numpy for processing
        X_train_full = train_df[feature_cols].values
        y_train_full = train_df['binary_label'].values
        X_test_full = test_df[feature_cols].values
        y_test_full = test_df['binary_label'].values

        # === STEP 5: Feature Selection (IG + RF Hybrid) ===
        logger.info("Step 5: Feature Selection (IG + RF Hybrid) - FIXED")

        # Calculate Information Gain
        ig_scores = mutual_info_classif(X_train_full, y_train_full, random_state=42)

        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_train_full, y_train_full)
        rf_importance = rf.feature_importances_

        # Combine scores (average rank)
        ig_ranks = np.argsort(ig_scores)[::-1]
        rf_ranks = np.argsort(rf_importance)[::-1]

        # Average rank approach
        combined_ranks = np.zeros(len(feature_cols))
        for i, feat_idx in enumerate(ig_ranks):
            combined_ranks[feat_idx] += i
        for i, feat_idx in enumerate(rf_ranks):
            combined_ranks[feat_idx] += i

        # Select top features (keep top 60% of features)
        n_features_to_keep = max(20, int(len(feature_cols) * 0.6))
        selected_indices = np.argsort(combined_ranks)[:n_features_to_keep]
        selected_features = [feature_cols[i] for i in selected_indices]

        logger.info(f"   Selected {len(selected_features)} features (from {len(feature_cols)})")

        # Apply feature selection
        X_train_selected = X_train_full[:, selected_indices]
        X_test_selected = X_test_full[:, selected_indices]

        # === STEP 6: Feature Scaling ===
        logger.info("Step 6: Feature Scaling")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        # === STEP 7: Data Rebalancing (SMOTE) ===
        logger.info("Step 7: Data Rebalancing (SMOTE) - FIXED")

        # Store multiclass labels before SMOTE
        y_train_multiclass_full = train_df['label_int'].values

        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_full)

        logger.info(f"   Before SMOTE: {X_train_scaled.shape[0]} samples")
        logger.info(f"   After SMOTE:  {X_train_balanced.shape[0]} samples")
        logger.info(f"   Class distribution: Benign={np.sum(y_train_balanced==0)}, Attack={np.sum(y_train_balanced==1)}")

        # Create multiclass labels for SMOTE-augmented data
        # For new synthetic samples, use binary label (0 or 1)
        y_train_multiclass_balanced = np.zeros(len(y_train_balanced), dtype=int)
        original_count = len(y_train_full)
        y_train_multiclass_balanced[:original_count] = y_train_multiclass_full
        # Synthetic samples get generic attack label (1) or benign (0)
        y_train_multiclass_balanced[original_count:] = y_train_balanced[original_count:]

        # Split into train/validation (80/20)
        X_train, X_val, y_train, y_val, y_train_mc, y_val_mc = train_test_split(
            X_train_balanced, y_train_balanced, y_train_multiclass_balanced,
            test_size=0.2, random_state=42, stratify=y_train_balanced
        )

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        y_train_mc_t = torch.LongTensor(y_train_mc)

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        y_val_mc_t = torch.LongTensor(y_val_mc)

        X_test_t = torch.FloatTensor(X_test_scaled)
        y_test_t = torch.LongTensor(y_test_full)
        y_test_mc_t = torch.LongTensor(test_df['label_int'].values)

        # Zero-day indices
        zero_day_attack_label = self.attack_types.get(zero_day_attack, 4)
        zero_day_mask = test_df['attack_cat'] == zero_day_attack
        zero_day_indices = torch.where(torch.tensor(zero_day_mask.values, dtype=torch.bool))[0].tolist()

        logger.info("\n✅ CICIDS2023 FULL preprocessing completed!")
        logger.info(f"   Selected features: {len(selected_features)}")
        logger.info(f"   Training samples: {len(X_train_t)} (after SMOTE + split)")
        logger.info(f"   Validation samples: {len(X_val_t)}")
        logger.info(f"   Test samples: {len(X_test_t)}")
        logger.info(f"   Zero-day samples ({zero_day_attack}): {len(zero_day_indices)}")

        return {
            'X_train': X_train_t,
            'y_train': y_train_t,
            'y_train_multiclass': y_train_mc_t,
            'X_val': X_val_t,
            'y_val': y_val_t,
            'y_val_multiclass': y_val_mc_t,
            'X_test': X_test_t,
            'y_test': y_test_t,
            'y_test_multiclass': y_test_mc_t,
            'test_attack_cat': test_df['attack_cat'].values,
            'zero_day_indices': zero_day_indices,
            'zero_day_attack': zero_day_attack,
            'zero_day_attack_label': zero_day_attack_label,
            'scaler': scaler,
            'feature_names': selected_features,
            'attack_types': self.attack_types
        }
