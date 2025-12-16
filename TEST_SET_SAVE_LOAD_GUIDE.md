# Test Set Save/Load Guide

## Overview

This feature allows you to **save and reuse the exact test set** from optimization trials to ensure **identical evaluation conditions** between optimization runs and subsequent evaluation runs.

## Why This Is Needed

During hyperparameter optimization, each trial creates a different test set due to:
- Different sequence creation parameters (sequence_length, stride)
- Different random sampling during test set creation
- Different post-sequence filtering results

This causes **inconsistent evaluation** where:
- Optimization trial 13: Base ZDR = 91.67%, TTT improves to 100%
- Current run: Base ZDR = 100% (already perfect), TTT can't improve

By saving and reusing the test set from the best optimization trial, we ensure **fair and reproducible comparisons**.

---

## How It Works

### 1. **During Optimization** (`optimize_hyperparameters.py`)

After preprocessing data for each trial, the test set is automatically saved:

```python
# After preprocessing
if not system.preprocess_data():
    return float('-inf')
    
# Save test set for this trial
self._save_test_set(system.preprocessed_data, trial.number)
```

**Saved Files:**
- `saved_test_sets/test_set_trial_13.pkl` - Test set from trial 13
- `saved_test_sets/test_set_best_trial.pkl` - Copy of best trial (trial 13) for easy access

**Saved Data:**
- `X_test` - Test sequences (after sequence creation)
- `y_test` - Binary labels
- `y_test_multiclass` - Multiclass labels (for zero-day identification)
- `test_attack_cat` - Attack category names
- `X_test_original` - Original test data (before sequences)
- `y_test_original` - Original labels (before sequences)
- `test_attack_cat_original` - Original attack categories
- `zero_day_indices` - Indices of zero-day samples
- `zero_day_attack` - Name of zero-day attack type
- `trial_number` - Trial number for reference

---

### 2. **During Regular Runs** (`main.py`)

Before preprocessing, the system checks for a saved test set:

```python
# Check if saved test set exists
saved_test_set = self._load_saved_test_set()

# After sequence creation, replace with saved test set
if saved_test_set is not None:
    self.preprocessed_data['X_test'] = saved_test_set['X_test']
    # ... replace all test set data
```

**Load Priority:**
1. First tries: `saved_test_sets/test_set_best_trial.pkl` (best trial)
2. Falls back to: `saved_test_sets/test_set_trial_13.pkl` (trial 13)
3. If neither exists: Creates new test set normally

---

## Usage

### **Automatic Usage (Recommended)**

The system **automatically** uses saved test sets if available:

1. **Run optimization** - Test sets are automatically saved for each trial
2. **Run regular evaluation** - If `test_set_best_trial.pkl` exists, it will be used automatically

**No manual intervention needed!**

### **Manual Usage**

#### **Save Test Set Manually**

```python
from optimize_hyperparameters import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(...)
system = BlockchainFederatedIncentiveSystem(config)
system.preprocess_data()

# Save test set manually
optimizer._save_test_set(system.preprocessed_data, trial_number=13)
```

#### **Load Test Set Manually**

```python
import pickle
from pathlib import Path

test_set_path = Path("saved_test_sets/test_set_best_trial.pkl")
with open(test_set_path, 'rb') as f:
    test_set_data = pickle.load(f)
    
print(f"Test samples: {len(test_set_data['X_test'])}")
print(f"Trial number: {test_set_data['trial_number']}")
```

#### **Disable Test Set Loading**

If you want to use a **new test set** instead of the saved one:

1. **Rename/remove** the saved test set file:
   ```bash
   mv saved_test_sets/test_set_best_trial.pkl saved_test_sets/test_set_best_trial.pkl.backup
   ```

2. **Or modify** `main.py` to skip loading:
   ```python
   # In preprocess_data(), comment out:
   # saved_test_set = self._load_saved_test_set()
   saved_test_set = None  # Disable saved test set
   ```

---

## Expected Behavior

### **With Saved Test Set**

```
📦 Loading saved test set from: saved_test_sets/test_set_best_trial.pkl
✅ Loaded test set from trial 13
📦 Found saved test set - will use it after preprocessing
   Test set from trial: 13
   Zero-day attack: Backdoor
Preprocessing dataset...
...
🔄 Replacing test set with saved test set from optimization trial...
✅ Test set replaced: 240 sequences
   Using test set from trial 13
```

### **Without Saved Test Set**

```
📦 No saved test set found - will create new test set during preprocessing
Preprocessing dataset...
...
✅ Test sequences created: (240, 30, 49)
```

---

## File Structure

```
project_root/
├── saved_test_sets/              # Directory for saved test sets
│   ├── test_set_trial_0.pkl      # Test set from trial 0
│   ├── test_set_trial_1.pkl      # Test set from trial 1
│   ├── ...
│   ├── test_set_trial_13.pkl     # Test set from trial 13 (best)
│   └── test_set_best_trial.pkl   # Copy of best trial (easy access)
├── optimize_hyperparameters.py   # Optimization script (saves test sets)
├── main.py                       # Main script (loads saved test sets)
└── ...
```

---

## Benefits

✅ **Reproducible Results**: Same test set = same evaluation conditions  
✅ **Fair Comparison**: Base and TTT models evaluated on identical data  
✅ **Debugging**: Easier to compare optimization vs regular runs  
✅ **Validation**: Verify that optimized hyperparameters actually improve performance  

---

## Troubleshooting

### **Issue: Test set not loading**

**Check:**
1. Does `saved_test_sets/` directory exist?
2. Does `test_set_best_trial.pkl` exist?
3. Check logs for error messages

**Solution:**
```bash
ls saved_test_sets/
# Should show test_set_best_trial.pkl
```

### **Issue: Dimension mismatch errors**

**Cause:** Saved test set was created with different sequence parameters.

**Solution:**
1. Re-run optimization with current config to regenerate test sets
2. Or manually create test set with matching parameters

### **Issue: Zero-day attack mismatch**

**Cause:** Saved test set uses different zero-day attack than current config.

**Solution:**
1. Check `config.zero_day_attack` matches saved test set
2. Or regenerate test set with current attack type

---

## Notes

- **Test sets are saved as pickle files** - Ensure Python version compatibility
- **Test sets include tensors** - May be large files (check disk space)
- **Test sets are trial-specific** - Each trial gets its own test set saved
- **Best trial is copied** - Trial 13 test set is saved as `test_set_best_trial.pkl` for convenience










