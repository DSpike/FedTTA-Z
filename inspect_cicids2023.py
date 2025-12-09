#!/usr/bin/env python3
"""
Helper script to inspect CICIDS2023 dataset structure
Run this first to understand your dataset before configuring the preprocessor
"""
import pandas as pd
import sys

def inspect_dataset(file_path, max_rows=10000):
    """Inspect dataset structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*60}\n")
    
    try:
        # Read sample
        print("Reading sample (first 10,000 rows)...")
        df = pd.read_csv(file_path, nrows=max_rows, low_memory=False)
        
        print(f"\n📊 Dataset Shape: {df.shape}")
        print(f"   Rows: {df.shape[0]:,}")
        print(f"   Columns: {df.shape[1]}")
        
        print(f"\n📋 Column Names ({len(df.columns)} total):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:3d}. {col}")
        
        # Find label column
        label_candidates = ['Label', 'label', 'Attack', 'attack', 'Category', 'category', 
                           'Attack_Type', 'attack_type', 'Class', 'class']
        label_column = None
        for candidate in label_candidates:
            if candidate in df.columns:
                label_column = candidate
                break
        
        if label_column:
            print(f"\n✅ Found label column: '{label_column}'")
            
            # Get unique labels
            unique_labels = df[label_column].unique()
            print(f"\n🏷️  Unique Labels ({len(unique_labels)} total):")
            for i, label in enumerate(sorted(unique_labels), 1):
                count = (df[label_column] == label).sum()
                print(f"   {i:3d}. '{label}' ({count:,} samples)")
            
            # Check for BENIGN/Normal
            benign_variants = ['BENIGN', 'Benign', 'benign', 'Normal', 'normal', 'BENIGN_TRAFFIC']
            has_benign = any(variant in unique_labels for variant in benign_variants)
            if has_benign:
                print(f"\n✅ BENIGN/Normal class found")
            else:
                print(f"\n⚠️  No BENIGN/Normal class found - check label names")
        else:
            print(f"\n⚠️  Could not find label column automatically")
            print(f"   Searched for: {label_candidates}")
            print(f"   Please check column names above and update preprocessor")
        
        # Check data types
        print(f"\n📊 Data Types:")
        print(f"   Numeric columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        print(f"   Object columns: {len(df.select_dtypes(include=['object']).columns)}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values:")
            for col in missing[missing > 0].index:
                print(f"   {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
        else:
            print(f"\n✅ No missing values found")
        
        # Feature count (excluding label)
        if label_column:
            feature_count = len(df.columns) - 1  # Exclude label column
            print(f"\n🔢 Feature Count (excluding label): {feature_count}")
            print(f"   ⚠️  Update 'input_dim' in config.py to {feature_count}")
        
        print(f"\n{'='*60}\n")
        
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'label_column': label_column,
            'unique_labels': unique_labels.tolist() if label_column else [],
            'feature_count': feature_count if label_column else len(df.columns)
        }
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        print(f"   Make sure the file exists in the current directory")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("CICIDS2023 Dataset Inspector")
    print("="*60)
    
    # Default file paths
    train_file = "CICIDS2023_train.csv"
    test_file = "CICIDS2023_test.csv"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
    if len(sys.argv) > 2:
        test_file = sys.argv[2]
    
    print(f"\nInspecting training file: {train_file}")
    train_info = inspect_dataset(train_file)
    
    if train_info and train_info.get('label_column'):
        print(f"\n{'='*60}")
        print("📝 NEXT STEPS:")
        print(f"{'='*60}")
        print(f"1. Update 'attack_types' in config.py with these labels:")
        print(f"   attack_types = {{")
        print(f"       'BENIGN': 0,")
        for i, label in enumerate(sorted(train_info['unique_labels']), 1):
            if str(label).upper() not in ['BENIGN', 'NORMAL', 'BENIGN_TRAFFIC']:
                print(f"       '{label}': {i},")
        print(f"   }}")
        print(f"\n2. Update 'input_dim' in config.py to: {train_info['feature_count']}")
        print(f"\n3. Update label_column in preprocessor if different from 'Label'")
        print(f"   Current label column: '{train_info['label_column']}'")
        print(f"\n4. Choose a zero-day attack from the list above")
        print(f"   Update 'zero_day_attack' in config.py")
    
    if len(sys.argv) <= 2:
        print(f"\n{'='*60}")
        print(f"To inspect test file, run:")
        print(f"   python inspect_cicids2023.py {train_file} {test_file}")

