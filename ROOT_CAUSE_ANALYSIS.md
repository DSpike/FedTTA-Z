# Root Cause Analysis: Why Only Fuzzers and Normal in Training Data

## 🔍 **Problem Identified**

Training data only contains 2 attack types:
- Normal (label 0): 70,416 samples
- Fuzzers (label 1): 50,000 samples

**Expected**: 8 attack types (excluding zero-day Backdoor):
- Normal, Fuzzers, Analysis, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

---

## 📊 **Debug Findings**

From `DEBUG_ATTACK_TYPES.py` output:

### **After Rebalancing (step7_data_rebalancing_complete):**
```
Normal (Label 0): 44,800 samples (31.94%)
Fuzzers (Label 1): 95,472 samples (68.06%)
```

**Only 2 labels found!** This means the input to `step7_data_rebalancing_complete` already only has 2 attack types.

---

## 🔍 **Code Flow Analysis**

### **Flow 1: Before Split**
```
train_df (from earlier steps)
├─ Has 'attack_cat' column? ✓ (from step4_categorical_encoding)
├─ Has 'label' column? ❓ Unknown - need to check
└─ Split happens using train_df['label'] (line 1072)
```

### **Flow 2: At Split (line 1072)**
```python
y = train_df['label'].values  # What's in this column?
```

**Issue**: If `train_df['label']` is binary (0, 1) at this point, then the split will only see 2 classes.

### **Flow 3: After Split**
```python
train_df_split['label'] = y_train_split  # Only has 2 labels (0, 1)
train_df_split['attack_cat'] = [reverse_mapping[label] for label in y_train_split]
```

**Problem**: `reverse_mapping` only maps labels 0 and 1:
- `0 → Normal`
- `1 → Fuzzers`
- **But what about labels 2-9?**

If `y_train_split` only contains 0 and 1, then `attack_cat` will only be Normal and Fuzzers!

---

## 🎯 **Root Cause Hypothesis**

**Hypothesis**: The `train_df['label']` column at line 1072 already contains binary labels (0, 1) instead of multiclass labels (0-9).

### **Possible Causes:**

1. **Step 4 (Categorical Encoding)**: May be creating binary labels instead of preserving multiclass
2. **Step 7 (Rebalancing)**: May be called earlier and converting to binary
3. **Data Source**: The original data split might already be binary

---

## 🔧 **Investigation Steps Needed**

1. ✅ Check what `train_df['label']` contains at line 1072 (before split)
2. ✅ Check what `step4_categorical_encoding` returns
3. ✅ Check if labels are being converted to binary somewhere before the split
4. ✅ Verify original data has all attack types

---

## 💡 **Expected Fix**

The `train_df['label']` should contain multiclass labels (0-9) before the split, not binary labels (0, 1).

**Fix Location**: Need to ensure `train_df['label']` is created from `attack_cat` using multiclass mapping BEFORE the split.

---

## 📋 **Next Steps**

1. Add debug logging to check `train_df['label']` at line 1072
2. Verify what `step4_categorical_encoding` outputs
3. Check if labels need to be created from `attack_cat` before the split
4. Fix the label creation to preserve all 8 attack types










