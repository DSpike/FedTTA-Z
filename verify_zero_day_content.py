"""
Verify Zero-Day Content
Checks saved test sets to confirm zero-day samples are present and correctly labeled.
"""
import pickle
import numpy as np
import os
import torch
import sys

# Add current directory to path to import config
sys.path.append(os.getcwd())

try:
    from config import config
except ImportError:
    # Fallback if config cannot be imported
    class Config:
        zero_day_attack = "PortScan"
        use_category_grouping = True
        
        @property
        def zero_day_attack_label(self):
            return 4 if self.use_category_grouping else 10
            
    config = Config()

def check_file(filepath):
    print(f"\n🔍 Inspecting: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if 'y_test_multiclass' not in data:
            print("   ⚠️ 'y_test_multiclass' not found in this file.")
            return

        labels = data['y_test_multiclass']
        if torch.is_tensor(labels):
            labels = labels.numpy()
        
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"   Labels found: {dist}")
        
        # Check for PortScan
        # Fine-grained ID for PortScan is 10
        # Grouped ID for PortScan (if using category grouping) is 4
        
        zero_day_name = getattr(config, 'zero_day_attack', 'PortScan')
        zero_day_label = getattr(config, 'zero_day_attack_label', 4)
        print(f"   Target Zero-Day: {zero_day_name} (Label: {zero_day_label})")
        
        found = False
        
        # Check for the configured zero-day label
        if zero_day_label in unique:
            count = dist[zero_day_label]
            pct = (count / len(labels)) * 100
            print(f"   ✅ Found Label {zero_day_label} ({zero_day_name}): {count} samples ({pct:.1f}%)")
            found = True
            
        if not found:
            print(f"   ❌ Configured zero-day label ({zero_day_label}) NOT found in this file.")
            print(f"      (If you are using a different zero-day attack, check the label mapping in config.py)")
            
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")

def main():
    print("="*60)
    print("ZERO-DAY SAMPLE VERIFICATION TOOL")
    print("="*60)
    
    folder = "saved_test_sets"
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' does not exist.")
        print("   Run the main training script first to generate data.")
        return

    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    if not files:
        print(f"❌ No .pkl files found in '{folder}'.")
        return
        
    print(f"Found {len(files)} test set files.")
    for f in files:
        check_file(os.path.join(folder, f))

if __name__ == "__main__":
    main()