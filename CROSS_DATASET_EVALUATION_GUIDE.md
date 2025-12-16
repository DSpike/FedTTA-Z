# Cross-Dataset Evaluation Guide

## ✅ **Cross-Dataset Evaluation Now Supported!**

The system now supports training on one dataset and testing on another dataset with **minimal configuration changes**.

---

## 🚀 **Quick Start**

### **Enable Cross-Dataset Evaluation**

In `config.py`, set:

```python
# === CROSS-DATASET EVALUATION ===
use_cross_dataset_evaluation: bool = True
source_data_path: str = "KDDTrain+.csv"      # Training dataset
target_test_path: str = "CICIDS2017_test.csv"  # Testing dataset
```

### **Example: Train on KDD, Test on CICIDS2017**

```python
# config.py
use_cross_dataset_evaluation: bool = True
source_data_path: str = "KDDTrain+.csv"
target_test_path: str = "CICIDS2017_test.csv"
zero_day_attack: str = "DoS"  # Zero-day attack for target dataset
```

---

## 📋 **How It Works**

### **1. Feature Alignment**

When cross-dataset evaluation is enabled:

- ✅ **Common features** between source and target datasets are automatically identified
- ✅ **Missing features** are logged with warnings
- ✅ **Only common features** are used for training and testing
- ✅ Features are **sorted** for consistency

### **2. Preprocessing Flow**

```
1. Load source dataset (training)
2. Load target dataset (testing)
3. Process both datasets separately
4. Align features (find common features)
5. Apply feature selection (if enabled)
6. Create train/test splits
```

### **3. Automatic Handling**

- ✅ Feature alignment happens **before** feature selection
- ✅ Missing features in test data are **padded with zeros** (same-dataset mode)
- ✅ Common features are **automatically selected** (cross-dataset mode)
- ✅ Label mapping uses **target dataset's attack types**

---

## ⚙️ **Configuration Options**

### **Same-Dataset Evaluation (Default)**

```python
use_cross_dataset_evaluation: bool = False
# Uses data_path and test_path from same dataset
```

### **Cross-Dataset Evaluation**

```python
use_cross_dataset_evaluation: bool = True
source_data_path: str = "KDDTrain+.csv"        # Training dataset
target_test_path: str = "CICIDS2017_test.csv"  # Testing dataset
```

---

## 📊 **Supported Scenarios**

### **Scenario 1: KDD → CICIDS2017**

```python
use_cross_dataset_evaluation: bool = True
source_data_path: str = "KDDTrain+.csv"
target_test_path: str = "CICIDS2017_test.csv"
zero_day_attack: str = "DoS"  # Or appropriate CICIDS2017 attack
```

### **Scenario 2: CICIDS2017 → KDD**

```python
use_cross_dataset_evaluation: bool = True
source_data_path: str = "CICIDS2017_train.csv"
target_test_path: str = "KDDTest+.csv"
zero_day_attack: str = "DoS"  # Or appropriate KDD attack
```

### **Scenario 3: Same Dataset (Default)**

```python
use_cross_dataset_evaluation: bool = False
# Uses config.data_path and config.test_path
```

---

## ⚠️ **Important Notes**

### **1. Feature Compatibility**

- Different datasets may have **different feature sets**
- Only **common features** will be used
- Missing features are **automatically excluded**

### **2. Label Mapping**

- Zero-day attack should match the **target dataset's** attack types
- Category grouping works with **target dataset's** categories

### **3. Feature Selection**

- Feature selection (IGRF-RFE) runs on **aligned features only**
- Selected features must exist in **both datasets**

### **4. Performance Impact**

- Cross-dataset evaluation may show **lower performance** (expected)
- This is normal due to **domain shift** between datasets

---

## 🔍 **Logging Output**

When cross-dataset evaluation is enabled, you'll see:

```
📊 CROSS-DATASET EVALUATION MODE
   Training on: KDDTrain+.csv
   Testing on: CICIDS2017_test.csv

🔗 Aligning features for cross-dataset evaluation...
   ⚠️  15 features in training data but not in test: ['feature1', 'feature2', ...]
   ⚠️  8 features in test data but not in training: ['feature3', 'feature4', ...]
   ✅ Using 28 common features
```

---

## ✅ **Benefits**

1. **Minimal Changes**: Only 3 config parameters needed
2. **Automatic Alignment**: Features aligned automatically
3. **Backward Compatible**: Same-dataset mode still works
4. **Flexible**: Works with any dataset combination

---

## 🎯 **Use Cases**

- **Domain Adaptation**: Test model generalization across datasets
- **Transfer Learning**: Evaluate pre-trained models on new datasets
- **Robustness Testing**: Test model performance on different data distributions
- **Research**: Compare performance across different NIDS datasets



