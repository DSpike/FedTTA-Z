# Test Set Composition Issue - Analysis

## 🔍 **Problem Identified**

From the latest run logs:
```
⚠️ Mismatch: 445 multiclass labels vs 752 sequences
Identified 752 zero-day sequences using test_attack_cat_original fallback
Zero-day samples: 752, Non-zero-day samples: 0
```

**Issue:** Test set shows **100% zero-day** when it should have:
- 40% Normal
- 35% Non-zero-day attacks
- 25% Zero-day attacks

---

## 🔍 **Root Cause Analysis**

### **Issue 1: Size Mismatch Between Multiclass Labels and Sequences**

**What's Happening:**
- `y_test_multiclass_seq`: 445 labels
- `y_test_tensor` (sequences): 752 sequences
- **Mismatch:** 445 ≠ 752

**Why This Happens:**
1. Test set is created with correct composition (40% Normal, 35% Non-zero-day, 25% Zero-day)
2. Sequences are created from test subset
3. Multiclass labels are mapped to sequences
4. **But the mapping doesn't create enough labels** (445 vs 752)

### **Issue 2: Incorrect Fallback Logic**

**When Size Mismatch Occurs:**
- Code falls back to using `test_attack_cat_original`
- `test_attack_cat_original` is from **BEFORE subset creation**
- Fallback logic checks ALL timesteps in sequences
- **Problem:** `test_attack_cat_original` may not match the subset that was used to create sequences

**Fallback Logic (lines 2844-2855):**
```python
# Check all timesteps in each sequence for zero-day attack
for seq_idx in range(len(y_test_tensor)):
    start_idx = seq_idx * sequence_stride
    end_idx = start_idx + sequence_length
    
    for original_idx in range(start_idx, min(end_idx, num_original_samples)):
        if test_attack_cat_original[original_idx] == zero_day_attack:
            zero_day_mask[seq_idx] = True  # Marks ENTIRE sequence as zero-day
            break
```

**Problem:** This marks a sequence as zero-day if **ANY timestep** contains the zero-day attack. But `test_attack_cat_original` is from the FULL original test data, not the subset!

---

## 🔍 **Where Test Set Is Created**

### **Step 1: Preprocessor Creates Test Set**

Location: Preprocessor (e.g., `blockchain_federated_cicids_preprocessor.py`)

**Target Composition:**
- 40% Normal
- 35% Non-zero-day attacks
- 25% Zero-day attacks

### **Step 2: Stratified Subset Creation**

Location: `main.py` line 906 (`_stratified_test_subset`)

**What It Does:**
- Takes test set from preprocessor
- Creates subset with target composition
- Returns: `X_test_subset`, `y_test_subset`, `y_test_multiclass_original`, `test_attack_cat_original`

### **Step 3: Sequence Creation**

Location: `main.py` line 929 (`create_sequences`)

**What It Does:**
- Creates sequences from subset
- Maps multiclass labels to sequences (lines 946-952)
- **Problem:** Mapping might not create enough labels

### **Step 4: Post-Sequence Filtering**

Location: `main.py` lines 962-1023

**What It Does:**
- Filters sequences to achieve target composition (25% zero-day)
- **Should ensure correct composition**

### **Step 5: Saved Test Set Loading**

Location: `main.py` lines 1051-1091

**What It Does:**
- Loads saved test set from optimization
- **Problem:** Saved test set might have wrong composition or mismatched sizes

---

## 🔍 **Why Zero-Day Mask Shows 100% Zero-Day**

### **Current Situation:**

1. **Size Mismatch:** Multiclass labels (445) ≠ Sequences (752)
2. **Fallback Triggered:** Uses `test_attack_cat_original`
3. **Wrong Original Data:** `test_attack_cat_original` is from BEFORE subset creation
4. **Incorrect Mapping:** Fallback checks original data, not subset data
5. **Result:** All sequences marked as zero-day

### **The Bug:**

The `test_attack_cat_original` used in fallback is from the **FULL original test data** (before subset), not from the **subset** that was used to create sequences!

**Example:**
- Full test data: 10,000 samples
- Subset: 1,000 samples (with correct 40/35/25 composition)
- Sequences created from subset: 752 sequences
- But `test_attack_cat_original` is from the **10,000 samples**, not the **1,000 subset**!

---

## 🔧 **How to Fix**

### **Fix 1: Store Subset-Level Attack Categories**

After creating the subset, store `test_attack_cat` from the **subset**, not the original:

```python
# After _stratified_test_subset
test_attack_cat_subset = ...  # Attack categories from SUBSET
self.preprocessed_data['test_attack_cat_subset'] = test_attack_cat_subset  # Store subset

# Use subset attack categories in fallback, not original
```

### **Fix 2: Fix Multiclass Label Mapping**

Ensure multiclass label mapping creates labels for ALL sequences:

```python
# Map multiclass labels to sequences - ensure ALL sequences get labels
y_test_multiclass_seq = []
for seq_idx in range(len(X_test_seq)):
    # Map correctly - ensure every sequence gets a label
    original_idx = seq_idx * sequence_stride + (sequence_length - 1)
    if original_idx < len(y_test_multiclass_original):
        y_test_multiclass_seq.append(y_test_multiclass_original[original_idx])
    else:
        # Handle edge case - use last label or create missing labels
        ...
```

### **Fix 3: Fix Fallback Logic**

Use subset-level attack categories, not original:

```python
# Use test_attack_cat from SUBSET, not original
if 'test_attack_cat_subset' in self.preprocessed_data:
    test_attack_cat = self.preprocessed_data['test_attack_cat_subset']  # Use subset!
else:
    # Fallback to original only if subset not available
    test_attack_cat = self.preprocessed_data.get('test_attack_cat_original')
```

---

## 🎯 **Recommended Fix Strategy**

1. ✅ **Store subset-level attack categories** after subset creation
2. ✅ **Fix multiclass label mapping** to ensure all sequences get labels
3. ✅ **Fix fallback logic** to use subset attack categories
4. ✅ **Verify composition** after sequence creation
5. ✅ **Check saved test sets** have correct composition

---

## 📋 **Next Steps**

1. Check where `test_attack_cat_original` is set (should be from subset, not original)
2. Fix multiclass label mapping to ensure 752 labels for 752 sequences
3. Update fallback to use subset-level data
4. Verify test set composition after all processing steps









