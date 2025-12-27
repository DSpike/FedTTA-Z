#!/usr/bin/env python3
"""Check saved test set for zero-day attack type"""

import pickle
from pathlib import Path

test_set_dir = Path("saved_test_sets")

if not test_set_dir.exists():
    print("❌ saved_test_sets directory does not exist")
    print("   This means no saved test set is being loaded")
    exit(0)

# Check for saved test set files
best_trial_path = test_set_dir / "test_set_best_trial.pkl"
trial13_path = test_set_dir / "test_set_trial_13.pkl"

files_found = []
if best_trial_path.exists():
    files_found.append(best_trial_path)
if trial13_path.exists():
    files_found.append(trial13_path)

if not files_found:
    print("✅ No saved test set files found")
    print("   The system will create a new test set with BruteForce as zero-day")
    exit(0)

print(f"📦 Found {len(files_found)} saved test set file(s):")
for file_path in files_found:
    print(f"   - {file_path.name}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        saved_zero_day = data.get('zero_day_attack', 'unknown')
        trial_number = data.get('trial_number', 'unknown')
        
        print(f"     Zero-day attack: {saved_zero_day}")
        print(f"     Trial number: {trial_number}")
        
        if saved_zero_day == "PortScan":
            print(f"     ⚠️  WARNING: This saved test set uses 'PortScan' as zero-day!")
            print(f"     ⚠️  The system should skip this and create a new test set with 'BruteForce'")
            print(f"     💡 Solution: Delete or rename this file to force creation of new test set")
        elif saved_zero_day == "BruteForce":
            print(f"     ✅ This saved test set uses 'BruteForce' as zero-day (matches current config)")
    except Exception as e:
        print(f"     ❌ Error reading file: {e}")



