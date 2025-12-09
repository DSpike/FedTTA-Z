# CICIDS2023 Dataset Setup Guide

## 📋 Overview

This guide explains how to adapt your project to work with the CICIDS2023 dataset.

## 🔍 Step 1: Understand CICIDS2023 Structure

**Note:** CICIDS2023 (CICIoT2023) has different attack types than CICIDS2017:
- **33 different attacks** categorized into **7 classes**
- Different feature structure (may have different number of features)
- Different label format

**Important:** You need to:
1. Check your CICIDS2023 CSV files to identify:
   - Column names
   - Label column name (might be "Label", "Attack", "Category", etc.)
   - Attack type names
   - Number of features

## 🔧 Step 2: Create CICIDS2023 Preprocessor

### Option A: Create New Preprocessor (Recommended)

Create a new file: `blockchain_federated_cicids2023_preprocessor.py`

```python
#!/usr/bin/env python3
"""
CICIDS2023 Preprocessor
Customized for Zero-Day Attack Detection
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
    """
    
    def __init__(self, data_path: str = "CICIDS2023_train.csv", test_path: str = "CICIDS2023_test.csv"):
        super().__init__(data_path, test_path)
        
        # TODO: Update with actual CICIDS2023 attack types from your dataset
        # You need to check your CSV file to see what attack types exist
        self.attack_types = {
            'BENIGN': 0,
            # Add all attack types from CICIDS2023 here
            # Example (update with actual names):
            # 'Attack_Type_1': 1,
            # 'Attack_Type_2': 2,
            # ... etc
        }
        logger.info("CICIDS2023 Preprocessor initialized")
    
    def load_and_clean_columns(self, path):
        """Load CSV and clean column names"""
        logger.info(f"Loading CICIDS2023 CSV file: {path}")
        try:
            # Load in chunks if file is large
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
    
    def preprocess_unsw_dataset(self, zero_day_attack: str = 'PortScan') -> dict:
        """
        Main Pipeline for CICIDS2023
        """
        logger.info(f"Starting CICIDS2023 preprocessing (Zero-Day Target: {zero_day_attack})")
        
        # 1. Load Data
        train_df = self.load_and_clean_columns(self.data_path)
        test_df = self.load_and_clean_columns(self.test_path)
        
        # 2. Normalize labels (adjust based on your dataset's label column name)
        # TODO: Check your CSV - what is the label column name?
        # Common names: 'Label', 'Attack', 'Category', 'Attack_Type'
        label_column = 'Label'  # CHANGE THIS to match your CSV
        
        def normalize_label(label):
            """Normalize label to match attack_types keys"""
            if pd.isna(label):
                return 'BENIGN'
            label_str = str(label).strip()
            
            # Try exact match first
            if label_str in self.attack_types:
                return label_str
            
            # Add custom normalization logic for CICIDS2023 if needed
            # ...
            
            return label_str
        
        # Apply normalization
        train_df[label_column] = train_df[label_column].apply(normalize_label)
        test_df[label_column] = test_df[label_column].apply(normalize_label)
        
        # 3. Continue with standard preprocessing pipeline
        # (The rest follows the same pattern as CICIDS2017)
        # ... (copy from CICIDS2017 preprocessor and adapt)
        
        # Return preprocessed data dictionary
        return {
            'X_train': ...,
            'y_train': ...,
            'X_test': ...,
            'y_test': ...,
            # ... etc
        }
```

### Option B: Modify Existing Preprocessor

Alternatively, you can modify `blockchain_federated_cicids_preprocessor.py` to support both datasets.

## ⚙️ Step 3: Update Configuration

### Update `config.py`:

```python
# config.py

# === DATA CONFIGURATION ===
data_path: str = "CICIDS2023_train.csv"  # Change to your CICIDS2023 train file
test_path: str = "CICIDS2023_test.csv"   # Change to your CICIDS2023 test file
zero_day_attack: str = "Your_Zero_Day_Attack_Name"  # Choose one attack as zero-day

# === ATTACK TYPES (CICIDS2023) ===
attack_types = {
    'BENIGN': 0,
    # Add all CICIDS2023 attack types here
    # Get these from your CSV file
    'Attack_Type_1': 1,
    'Attack_Type_2': 2,
    # ... etc (up to 33 attacks)
}

# === MODEL CONFIGURATION ===
input_dim: int = 43  # TODO: Check your CICIDS2023 feature count and update
# CICIDS2023 might have different number of features than CICIDS2017
```

## 🔄 Step 4: Update Main Code

### Update `main.py` (around line 450):

```python
# main.py

# Option A: Use new CICIDS2023 preprocessor
from blockchain_federated_cicids2023_preprocessor import CICIDS2023Preprocessor
self.preprocessor = CICIDS2023Preprocessor(
    data_path=self.config.data_path,
    test_path=self.config.test_path
)

# Option B: Modify existing preprocessor to auto-detect dataset
# (More complex, but allows switching between datasets)
```

## 📝 Step 5: Checklist

Before running, verify:

- [ ] CICIDS2023 CSV files are in the project directory
- [ ] Label column name is identified and updated in preprocessor
- [ ] All attack types are listed in `attack_types` dictionary
- [ ] Zero-day attack is selected and exists in dataset
- [ ] Input dimension matches CICIDS2023 feature count
- [ ] Preprocessor is updated/created
- [ ] Config file paths point to CICIDS2023 files
- [ ] Config attack_types dictionary is updated

## 🚀 Step 6: Quick Start Commands

1. **First, inspect your CICIDS2023 dataset:**
```python
import pandas as pd
df = pd.read_csv("CICIDS2023_train.csv", nrows=1000)
print("Columns:", df.columns.tolist())
print("Label column:", "Label" in df.columns)  # Check actual name
print("Unique labels:", df['Label'].unique() if 'Label' in df.columns else "Check column name")
print("Shape:", df.shape)
```

2. **Update config.py with your findings**

3. **Run the project:**
```bash
python main.py
```

## ⚠️ Important Notes

1. **Feature Count:** CICIDS2023 may have different number of features than CICIDS2017 (43). Check and update `input_dim` in config.

2. **Attack Types:** CICIDS2023 has 33 attacks in 7 classes. You need to map all of them to integers.

3. **Label Column:** The label column might be named differently. Common names:
   - `Label`
   - `Attack`
   - `Category`
   - `Attack_Type`

4. **Data Format:** Ensure CSV files are properly formatted and match the expected structure.

## 🆘 Troubleshooting

**Error: "Label column not found"**
- Check the actual column name in your CSV
- Update `label_column` in preprocessor

**Error: "Unknown attack type"**
- Add all attack types to `attack_types` dictionary
- Check for typos or variations in attack names

**Error: "Dimension mismatch"**
- Update `input_dim` in config.py to match CICIDS2023 feature count
- Check if feature selection is applied correctly

## 📚 References

- CICIDS2023 Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Original CICIDS2017 Preprocessor: `blockchain_federated_cicids_preprocessor.py`

