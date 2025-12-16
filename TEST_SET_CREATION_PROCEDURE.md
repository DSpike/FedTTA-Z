# Test Set Creation Procedure - Complete Guide

## 📋 **Overview**

The test set is created during the **preprocessing phase** before federated learning begins. This document explains the complete procedure from raw data to final test sequences.

---

## 🔄 **Complete Test Set Creation Flow**

### **Phase 1: Initial Preprocessing (Preprocessor)**

**Location:** `preprocessing/blockchain_federated_unsw_preprocessor.py` → `preprocess_unsw_dataset()`

**When:** Called at the start of `preprocess_data()` in `main.py`

#### **Step 1.1: Load Raw Data**
- Loads training and testing CSV files
- For UNSW-NB15: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
- For CICIDS: `CICIDS2017_train.csv` and `CICIDS2017_test.csv`

#### **Step 1.2: Data Cleaning & Feature Engineering**
- Cleans missing values, duplicates
- Performs feature engineering
- Encodes categorical variables
- Scales features

#### **Step 1.3: Zero-Day Split Creation**
**Method:** `create_zero_day_split()` (line 765 in preprocessor)

**What It Does:**
1. **Identifies zero-day attack** (e.g., "Exploits", "PortScan")
2. **Creates training set:**
   - Includes: Normal + All other attack types
   - Excludes: Zero-day attack completely
3. **Creates validation set:**
   - Includes: Normal + All other attack types  
   - Excludes: Zero-day attack completely
4. **Creates initial test set:**
   - Includes: Normal + Zero-day attack + Some other attacks
   - **Composition:** Targets 40% Normal, 35% Non-zero-day attacks, 25% Zero-day attacks

**Output:** 
```python
preprocessed_data = {
    'X_train': ...,  # Training features (no zero-day)
    'y_train': ...,  # Training labels
    'X_val': ...,    # Validation features (no zero-day)
    'y_val': ...,    # Validation labels
    'X_test': ...,   # Test features (includes zero-day)
    'y_test': ...,   # Test labels (binary: 0=Normal, 1=Attack)
    'y_test_multiclass': ...,  # Multiclass labels (0-9 for attack types)
    'test_attack_cat': ...,    # Attack category names
}
```

---

### **Phase 2: Sequence Creation & Filtering (`main.py`)**

**Location:** `main.py` → `preprocess_data()` method (line 726)

**When:** After preprocessor creates initial data, before federated learning starts

#### **Step 2.1: Check for Saved Test Set (Optional)**

**Location:** Line 740-751

```python
saved_test_set = None
if not skip_saved_test_set:
    saved_test_set = self._load_saved_test_set()
    # Checks for: saved_test_sets/test_set_best_trial.pkl
    # Or: saved_test_sets/test_set_trial_13.pkl
```

**Purpose:** Load previously saved test set from optimization trials for reproducibility.

**If Found:** Will use it AFTER creating new test set (if sizes/composition match).

---

#### **Step 2.2: Create Training & Validation Sequences**

**Location:** Lines 827-886

- Creates sequences for training data (subset: 50k samples)
- Creates sequences for validation data (subset: 10k samples)
- Uses sliding window: `sequence_length=25`, `stride=12`

---

#### **Step 2.3: Create Test Subset with Stratified Sampling**

**Location:** Lines 888-912

**What Happens:**

1. **Determine subset size:**
   ```python
   test_subset_size = min(10000, len(self.preprocessed_data['X_test']))
   ```

2. **Get multiclass labels:**
   ```python
   y_test_multiclass_original = self.preprocessed_data.get('y_test_multiclass', None)
   test_attack_cat_original = self.preprocessed_data.get('test_attack_cat', None)
   ```

3. **Create stratified subset:**
   **Method:** `_stratified_test_subset()` (line 539)
   
   **Target Composition:**
   - 40% Normal samples
   - 35% Non-zero-day attacks
   - 25% Zero-day attacks (or 30% before sequences, to account for dilution)
   
   **Process:**
   ```python
   # Sets temporary target for pre-sequence sampling
   self._temp_zero_day_target = 0.30  # 30% before sequences
   
   # Creates stratified subset
   X_test_subset, y_test_subset, y_test_multiclass_original, test_attack_cat_original = 
       self._stratified_test_subset(
           self.preprocessed_data['X_test'],
           self.preprocessed_data['y_test'],
           y_test_multiclass_original,
           test_attack_cat_original,
           test_subset_size  # 10,000 samples
       )
   ```

   **Output:** 
   - `X_test_subset`: 10,000 samples with correct composition
   - `y_test_subset`: Binary labels
   - `y_test_multiclass_original`: Multiclass labels (from subset)
   - `test_attack_cat_original`: Attack categories (from subset)

---

#### **Step 2.4: Create Sequences from Subset**

**Location:** Lines 929-937

```python
X_test_seq, y_test_seq = self.preprocessor.create_sequences(
    X_test_subset,      # 10,000 samples
    y_test_subset,
    sequence_length=25,
    stride=12,          # Sliding window
    zero_pad=True
)
```

**Result:** Creates sequences from 10k samples → ~832 sequences (depends on stride)

**Example:**
- Input: 10,000 samples
- Sequence length: 25
- Stride: 12
- Output: ~832 sequences

---

#### **Step 2.5: Map Multiclass Labels to Sequences**

**Location:** Lines 939-952

**What Happens:**
- Maps multiclass labels from original samples to sequences
- Each sequence gets label from the **last timestep** of that sequence

```python
for seq_idx in range(len(X_test_seq)):
    original_idx = seq_idx * sequence_stride + (sequence_length - 1)
    if original_idx < orig_len:
        original_label = y_test_multiclass_original[original_idx]
        y_test_multiclass_seq.append(original_label)
        test_attack_cat_seq.append(test_attack_cat_original[original_idx])
```

**Result:** 
- `y_test_multiclass_seq`: Multiclass labels for each sequence
- `test_attack_cat_seq`: Attack categories for each sequence

---

#### **Step 2.6: Post-Sequence Filtering**

**Location:** Lines 962-1023

**Purpose:** Achieve final target composition (25% zero-day) after sequence creation

**Process:**

1. **Count zero-day sequences:**
   ```python
   zero_day_mask = (y_test_multiclass_seq == self.config.zero_day_attack_label)
   zero_day_indices = torch.where(zero_day_mask)[0]
   non_zero_day_indices = torch.where(~zero_day_mask)[0]
   ```

2. **Calculate target counts:**
   ```python
   target_zero_day_percentage = 0.25  # 25% zero-day
   
   # Use ALL available zero-day sequences
   target_zero_day_count = available_zero_day
   
   # Calculate needed non-zero-day to maintain 25% ratio
   ratio_non_zero_day = (1.0 - 0.25) / 0.25 = 3.0
   target_non_zero_day_count = target_zero_day_count * 3.0
   ```

3. **Randomly sample sequences:**
   ```python
   selected_zero_day = np.random.choice(zero_day_indices, size=target_zero_day_count, replace=False)
   selected_non_zero_day = np.random.choice(non_zero_day_indices, size=target_non_zero_day_count, replace=False)
   selected_indices = np.concatenate([selected_zero_day, selected_non_zero_day])
   ```

4. **Filter sequences:**
   ```python
   X_test_seq = X_test_seq[selected_indices]
   y_test_seq = y_test_seq[selected_indices]
   y_test_multiclass_seq = y_test_multiclass_seq[selected_indices]
   ```

**Result:**
- **Example:** 188 zero-day + 564 non-zero-day = **752 sequences**
- **Final composition:** ~25% zero-day, ~75% non-zero-day

---

#### **Step 2.7: Store Original Subset for TTT**

**Location:** Lines 1041-1047

```python
# Store original subset (before sequences) for TTT adaptation
self.preprocessed_data['X_test_original'] = X_test_subset
self.preprocessed_data['y_test_original'] = y_test_subset
self.preprocessed_data['test_attack_cat_original'] = test_attack_cat_original
```

**Purpose:** Allows creating more sequences with smaller stride during TTT adaptation.

---

#### **Step 2.8: Check & Load Saved Test Set (Optional)**

**Location:** Lines 1049-1140

**What Happens:**

1. **Check if saved test set exists:**
   - Looks for `saved_test_sets/test_set_best_trial.pkl`
   - Or `saved_test_sets/test_set_trial_13.pkl`

2. **Validate saved test set:**
   - ✅ Check zero-day attack matches current config
   - ✅ Check sizes match (X_test, y_test, y_test_multiclass)
   - ✅ Check composition is correct (~25% zero-day, not 100%)

3. **If valid:** Replace newly created test set with saved one
   ```python
   if use_saved_test_set:
       self.preprocessed_data['X_test'] = saved_test_set['X_test']
       self.preprocessed_data['y_test'] = saved_test_set['y_test']
       self.preprocessed_data['y_test_multiclass'] = saved_test_set['y_test_multiclass']
       # ... also load test_attack_cat, X_test_original, etc.
   ```

4. **If invalid:** Use newly created test set (with correct composition)

---

#### **Step 2.9: Store Final Test Set**

**Location:** Lines 1174-1195

```python
if use_saved_test_set:
    # Don't overwrite - saved test set already set X_test/y_test
    self.preprocessed_data.update({
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        # X_test/y_test already set by saved test set
    })
else:
    # Use newly created sequences
    self.preprocessed_data.update({
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        'X_test': X_test_seq,      # ← Newly created sequences
        'y_test': y_test_seq,      # ← Newly created sequences
    })
```

---

## 📊 **Final Test Set Characteristics**

### **Composition (Target):**
- **40% Normal** samples
- **35% Non-zero-day attacks** (seen during training)
- **25% Zero-day attacks** (unseen during training)

### **Size:**
- **Original samples:** 10,000 (from preprocessor)
- **After sequences:** ~752 sequences (depends on stride)
- **After filtering:** 752 sequences (if target composition achieved)

### **Data Structure:**
```python
preprocessed_data = {
    'X_test': torch.Tensor,        # Shape: (752, 25, feature_dim) - sequences
    'y_test': torch.Tensor,        # Shape: (752,) - binary labels
    'y_test_multiclass': torch.Tensor,  # Shape: (752,) - multiclass labels
    'test_attack_cat': List[str],  # Shape: (752,) - attack category names
    'X_test_original': torch.Tensor,    # Original samples (before sequences)
    'y_test_original': torch.Tensor,    # Original labels
    'test_attack_cat_original': List[str],  # Original attack categories
}
```

---

## 🔍 **Key Methods**

### **1. `_stratified_test_subset()`**

**Location:** `main.py` line 539

**Purpose:** Create stratified subset with target composition

**Parameters:**
- `X_test`: Full test features
- `y_test`: Full test binary labels
- `y_test_multiclass`: Full test multiclass labels
- `test_attack_cat`: Full test attack categories
- `n_samples`: Target subset size (10,000)

**Returns:**
- `X_subset`: Stratified subset features
- `y_subset`: Stratified subset binary labels
- `y_multiclass_subset`: Stratified subset multiclass labels
- `attack_cat_subset`: Stratified subset attack categories

---

### **2. `create_sequences()`**

**Location:** `preprocessing/blockchain_federated_unsw_preprocessor.py`

**Purpose:** Create sequences from samples using sliding window

**Parameters:**
- `X`: Features tensor
- `y`: Labels tensor
- `sequence_length`: Length of each sequence (25)
- `stride`: Step size for sliding window (12)
- `zero_pad`: Whether to zero-pad

**Returns:**
- `X_seq`: Sequences tensor (num_sequences, sequence_length, feature_dim)
- `y_seq`: Labels tensor (num_sequences,)

---

## 🎯 **When Test Set Is Created**

1. **Called from:** `main()` function → `system.preprocess_data()`
2. **Timing:** **Before** federated learning starts
3. **Frequency:** Once per run (unless saved test set is loaded)
4. **During optimization:** Each trial creates its own test set (saved for reproducibility)

---

## 📝 **Summary Timeline**

```
1. System Initialization
   ↓
2. Preprocessor loads raw data
   ↓
3. Preprocessor creates zero-day split
   → Initial test set created (40% Normal, 35% Non-zero-day, 25% Zero-day)
   ↓
4. main.py: Create sequences for train/val
   ↓
5. main.py: Create stratified test subset (10k samples)
   ↓
6. main.py: Create sequences from subset (~832 sequences)
   ↓
7. main.py: Map multiclass labels to sequences
   ↓
8. main.py: Post-sequence filtering (achieve 25% zero-day)
   → Final test set: 752 sequences
   ↓
9. main.py: Check saved test set (optional)
   → If valid: Replace with saved test set
   → If invalid: Use newly created test set
   ↓
10. Store final test set in preprocessed_data
   ↓
11. Ready for federated learning & evaluation
```

---

## ✅ **Verification**

After test set creation, check logs for:
```
✅ After post-sequence filtering: 188/752 zero-day sequences (25.0%) [TARGET: 25%]
✅ Verified alignment: All test sequences have length 752 after filtering
```

This confirms the test set has the correct composition!









