# KDD Dataset Testing - Next Steps

## ✅ **Current Status**

### **Completed:**
1. ✅ Created new branch: `kdd-dataset-testing`
2. ✅ Created KDD preprocessor: `centralized_nids_kdd_preprocessor.py`
3. ✅ Updated `config.py` for KDD dataset:
   - Data paths: `KDDTrain+.csv`, `KDDTest+.csv`
   - Zero-day attack: `neptune` (label 3)
   - Attack types: All 40 KDD attack types mapped
   - Input dimension: 41 features
4. ✅ Updated `main.py` to detect and use KDD preprocessor
5. ✅ Files compile without errors

---

## 🎯 **Next Steps to Run KDD Testing**

### **Step 1: Verify Dataset Files Exist**
```bash
# Check if KDD files are in the project directory
ls KDDTrain+.csv KDDTest+.csv
```

### **Step 2: Run Preprocessing Test (Optional)**
Test the preprocessor to ensure it works:
```python
from centralized_nids_kdd_preprocessor import KDDPreprocessor
from config import SystemConfig

config = SystemConfig()
preprocessor = KDDPreprocessor(
    data_path=config.data_path,
    test_path=config.test_path
)
preprocessor.config = config  # Attach config for feature selection if needed

# Test preprocessing
data = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)
print(f"Training samples: {len(data['X_train'])}")
print(f"Test samples: {len(data['X_test'])}")
print(f"Zero-day samples: {(data['y_test_multiclass'] == 3).sum()}")
```

### **Step 3: Run Full Experiment**
```bash
python main.py
```

This will:
1. Load and preprocess KDD dataset
2. Train the model (excluding 'neptune' from training)
3. Evaluate base model performance
4. Run TTT adaptation
5. Evaluate TTT model performance
6. Generate performance plots and metrics

---

## 📋 **Expected Results**

### **Dataset Information:**
- **Training**: ~125,000 samples (normal + attacks, excluding neptune)
- **Test**: ~22,500 samples (normal + attacks, including neptune as zero-day)
- **Zero-day attack**: `neptune` (DoS attack, label 3)
- **Features**: 41 features after preprocessing

### **Metrics to Monitor:**
- Base model accuracy, F1, precision, recall
- TTT model improvements
- **Zero-Day Detection Rate (ZDR)** - should be > 0 if model detects neptune attacks
- False Alarm Rate (FAR)
- AUC-PR and AUC-ROC curves

---

## ⚠️ **Potential Issues to Watch For**

1. **ZDR = 0**: If zero-day detection rate is zero, check:
   - Zero-day samples are found in test set
   - Model predictions for zero-day samples
   - Threshold may be too conservative

2. **Memory Issues**: KDD dataset is large, may need:
   - Sampling for memory efficiency
   - Adjust batch size if needed

3. **Feature Mismatch**: If errors about feature count:
   - Verify `input_dim = 41` in config
   - Check if feature selection is enabled (may change feature count)

---

## 🔄 **To Switch Back to Previous Dataset**

If you need to switch back to CICIDS2017 or CICIoT2023:

1. **Switch branch**:
   ```bash
   git checkout cicids2023-implementation  # or your previous branch
   ```

2. **Or update config.py** in current branch:
   - Change `data_path` and `test_path`
   - Update `attack_types` dictionary
   - Update `input_dim`
   - Update `zero_day_attack`

---

## 📝 **Branch Management**

- **Current branch**: `kdd-dataset-testing`
- **Previous branch**: `cicids2023-implementation` (unchanged)
- **KDD changes**: Only in `kdd-dataset-testing` branch
- **Previous dataset code**: Safe in `cicids2023-implementation` branch

---

## ✅ **Ready to Test!**

Everything is configured. You can now run:
```bash
python main.py
```

The system will automatically:
- Detect KDD dataset from file names
- Use KDD preprocessor
- Configure for neptune as zero-day attack
- Run the full experiment pipeline





