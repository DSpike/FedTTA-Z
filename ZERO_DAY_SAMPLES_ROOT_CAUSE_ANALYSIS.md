# Zero-Day Samples Not Generated - Root Cause Analysis

## 🔍 **Problem Summary**

Zero-day samples are showing as **0** in the evaluation plots, indicating that the zero-day identification logic is failing to find any zero-day samples in the test data.

---

## 📊 **Data Flow Analysis**

### **Phase 1: Preprocessing (Preprocessor)**
**Location:** `preprocessing/blockchain_federated_cicids_preprocessor.py`

1. **Initial Test Set Creation:**
   - Preprocessor creates test set with target composition: 40% Normal, 35% Non-zero-day attacks, 25% Zero-day attacks
   - Stores: `X_test`, `y_test`, `y_test_multiclass`, `test_attack_cat`
   - **Status:** ✅ Zero-day samples should be present at this stage

### **Phase 2: Stratified Subset Creation**
**Location:** `main.py` lines 1106-1127

1. **Subset Creation:**
   - Gets `y_test_multiclass_original` from `preprocessed_data` (line 1106)
   - Calls `_stratified_test_subset()` with 30% zero-day target
   - Returns: `X_test_subset`, `y_test_subset`, `y_test_multiclass_original`, `test_attack_cat_original`
   - **Potential Issue #1:** If `y_test_multiclass_original` is `None`, falls back to simple slicing (line 1132-1135)
   - **Potential Issue #2:** If `test_attack_cat_original` is `None`, zero-day identification will fail later

### **Phase 3: Sequence Creation**
**Location:** `main.py` lines 1144-1160

1. **Sequence Generation:**
   - Creates sequences from subset using `create_sequences()`
   - **Status:** ✅ Sequences are created correctly

### **Phase 4: Multiclass Label Mapping**
**Location:** `main.py` lines 1163-1257

1. **Label Mapping:**
   - Maps multiclass labels to sequences (lines 1169-1175)
   - Creates `y_test_multiclass_seq` from `y_test_multiclass_original`
   - **Critical Code:**
     ```python
     for seq_idx in range(len(X_test_seq)):
         original_idx = seq_idx * sequence_stride + (sequence_length - 1)
         if original_idx < orig_len:
             original_label = y_test_multiclass_original[original_idx]
             y_test_multiclass_seq.append(original_label)
     ```
   - **Potential Issue #3:** If `y_test_multiclass_original` is `None` or empty, `y_test_multiclass_seq` will be empty
   - **Potential Issue #4:** If `original_idx` is out of bounds, labels won't be mapped correctly

2. **Post-Sequence Filtering:**
   - Filters sequences to achieve 25% zero-day target (lines 1185-1246)
   - Stores filtered sequences: `X_test_seq`, `y_test_seq`, `y_test_multiclass_seq`
   - **Status:** ✅ If `y_test_multiclass_seq` has zero-day labels, filtering should work

3. **Storage:**
   - Stores `y_test_multiclass` in `preprocessed_data` (line 1257)
   - Stores `test_attack_cat_original` (line 1272)
   - **Status:** ✅ Data should be stored correctly

### **Phase 5: Saved Test Set Loading (CRITICAL)**
**Location:** `main.py` lines 1274-1350

1. **Saved Test Set Check:**
   - Checks if saved test set exists (line 1276)
   - Validates sizes and zero-day attack name (lines 1289-1337)
   - **Potential Issue #5:** If saved test set is loaded, it **OVERWRITES** `X_test`, `y_test`, and `y_test_multiclass` (lines 1341-1347)
   - **Critical Code:**
     ```python
     if use_saved_test_set:
         self.preprocessed_data['X_test'] = saved_test_set['X_test']
         self.preprocessed_data['y_test'] = saved_test_set['y_test']
         saved_multiclass = saved_test_set.get('y_test_multiclass')
         if saved_multiclass is not None:
             self.preprocessed_data['y_test_multiclass'] = saved_multiclass
         else:
             logger.warning("⚠️ Saved test set has no multiclass labels")
     ```
   - **Potential Issue #6:** If saved test set doesn't have `y_test_multiclass`, it's not restored, leaving `preprocessed_data['y_test_multiclass']` as the newly created one (which might be overwritten by `X_test`/`y_test` from saved set)
   - **Potential Issue #7:** `test_attack_cat_original` is **NOT** restored from saved test set, so it might be missing

### **Phase 6: Zero-Day Identification (Evaluation)**
**Location:** `main.py` lines 3031-3105

1. **Identification Logic:**
   - **Priority 1:** Uses `y_test_multiclass` if available (line 3040)
   - **Priority 2:** Falls back to `test_attack_cat_original` if available (line 3075)
   - **Priority 3:** Creates empty mask if neither available (line 3101-3104)

2. **Potential Failure Points:**
   - **Failure Point #1:** `y_test_multiclass` doesn't exist in `preprocessed_data`
   - **Failure Point #2:** `y_test_multiclass` exists but has wrong length (mismatch with sequences)
   - **Failure Point #3:** `y_test_multiclass` exists but doesn't contain zero-day attack label
   - **Failure Point #4:** `test_attack_cat_original` doesn't exist
   - **Failure Point #5:** `test_attack_cat_original` exists but zero-day attack name doesn't match

---

## 🎯 **Root Cause Hypotheses**

### **Hypothesis 1: Saved Test Set Overwrites Correct Data** ⚠️ **MOST LIKELY**

**Scenario:**
1. New test set is created with correct `y_test_multiclass` containing zero-day samples
2. Saved test set is loaded and overwrites `X_test`, `y_test`
3. Saved test set either:
   - Doesn't have `y_test_multiclass` → leaves old one (but `X_test`/`y_test` changed, causing mismatch)
   - Has `y_test_multiclass` but it's wrong/empty → overwrites correct one
4. `test_attack_cat_original` is not restored from saved test set
5. Evaluation tries to use `y_test_multiclass` but it doesn't match `X_test`/`y_test` anymore
6. Falls back to `test_attack_cat_original` but it's missing or wrong
7. Result: Zero zero-day samples found

**Evidence:**
- Code at line 1341-1347 shows saved test set overwrites data
- `test_attack_cat_original` is not restored from saved test set
- Warning at line 1350: "Saved test set has no multiclass labels"

### **Hypothesis 2: Multiclass Labels Not Created During Sequence Mapping**

**Scenario:**
1. `y_test_multiclass_original` is `None` or empty
2. Sequence mapping loop (lines 1169-1175) doesn't create any labels
3. `y_test_multiclass_seq` is empty
4. Evaluation can't find zero-day samples

**Evidence:**
- Code at line 1163 checks `if y_test_multiclass_original is not None`
- If `None`, mapping is skipped
- Warning at line 1262: "No multiclass labels mapped to sequences"

### **Hypothesis 3: Zero-Day Attack Label Mismatch**

**Scenario:**
1. Zero-day attack name in config doesn't match attack names in data
2. `zero_day_attack_label` is wrong
3. Comparison `y_test_multiclass_seq == zero_day_attack_label` never matches
4. Result: Zero zero-day samples found

**Evidence:**
- Code at line 3036: `zero_day_attack_label = attack_types.get(zero_day_attack, 1)`
- Defaults to label 1 if not found
- Comparison at line 3051: `zero_day_mask = (y_test_multiclass_seq == zero_day_attack_label)`

### **Hypothesis 4: Size Mismatch Between Sequences and Labels**

**Scenario:**
1. `y_test_multiclass_seq` length doesn't match `y_test_tensor` length
2. Code at line 3054 detects mismatch and creates empty mask
3. Result: Zero zero-day samples found

**Evidence:**
- Code at line 3050-3056 checks length match
- If mismatch, creates empty mask: `zero_day_mask = torch.zeros(len(y_test_tensor), dtype=torch.bool, device=self.device)`

---

## 🔧 **Recommended Fixes**

### **Fix 1: Ensure Saved Test Set Preserves All Required Data**

**Location:** `main.py` lines 1339-1350

**Change:**
```python
if use_saved_test_set:
    # All sizes match - safe to overwrite
    self.preprocessed_data['X_test'] = saved_test_set['X_test']
    self.preprocessed_data['y_test'] = saved_test_set['y_test']
    saved_multiclass = saved_test_set.get('y_test_multiclass')
    
    if saved_multiclass is not None:
        # Sizes already verified above - safe to use
        self.preprocessed_data['y_test_multiclass'] = saved_multiclass
        logger.info(f"✅ Multiclass labels loaded: {len(saved_multiclass)} labels aligned with {len(saved_test_set['X_test'])} sequences")
    else:
        logger.error(f"❌ Saved test set has no multiclass labels! Cannot identify zero-day samples.")
        logger.error(f"   Falling back to newly created test set to preserve zero-day identification capability.")
        use_saved_test_set = False  # Don't use saved set if it's missing critical data
    
    # CRITICAL: Also restore test_attack_cat_original if available
    saved_test_attack_cat_original = saved_test_set.get('test_attack_cat_original')
    if saved_test_attack_cat_original is not None:
        self.preprocessed_data['test_attack_cat_original'] = saved_test_attack_cat_original
        logger.info(f"✅ test_attack_cat_original restored from saved test set: {len(saved_test_attack_cat_original)} samples")
    else:
        logger.warning(f"⚠️ Saved test set has no test_attack_cat_original. Zero-day identification may fail if y_test_multiclass is missing.")
```

### **Fix 2: Add Validation After Sequence Mapping**

**Location:** `main.py` lines 1176-1183

**Change:**
```python
if len(y_test_multiclass_seq) > 0:
    y_test_multiclass_seq = torch.tensor(y_test_multiclass_seq, dtype=torch.int64)
    # CRITICAL VALIDATION: Check length match
    if len(y_test_multiclass_seq) != len(X_test_seq):
        logger.error(f"❌ CRITICAL: y_test_multiclass_seq length ({len(y_test_multiclass_seq)}) != X_test_seq length ({len(X_test_seq)})!")
        logger.error(f"   This will cause zero-day identification to fail. Recreating labels...")
        # Try to fix by recreating from original data
        # ... (add fix logic)
    else:
        # Debug: Count zero-day sequences in mapped labels
        zero_day_count_in_seq = (y_test_multiclass_seq == self.config.zero_day_attack_label).sum().item()
        total_seq_count = len(y_test_multiclass_seq)
        current_percentage = 100 * zero_day_count_in_seq / total_seq_count if total_seq_count > 0 else 0
        logger.info(f"🔍 Before post-sequence filtering: {zero_day_count_in_seq}/{total_seq_count} zero-day sequences ({current_percentage:.1f}%)")
        if zero_day_count_in_seq == 0:
            logger.error(f"❌ CRITICAL: No zero-day sequences found after mapping! Check zero_day_attack_label={self.config.zero_day_attack_label}")
            logger.error(f"   Unique labels in y_test_multiclass_seq: {torch.unique(y_test_multiclass_seq).tolist()}")
else:
    logger.error(f"❌ CRITICAL: No multiclass labels mapped to sequences! Zero-day identification will fail.")
    logger.error(f"   y_test_multiclass_original length: {len(y_test_multiclass_original) if y_test_multiclass_original is not None else 'None'}")
    logger.error(f"   X_test_seq length: {len(X_test_seq)}")
```

### **Fix 3: Improve Zero-Day Identification Fallback**

**Location:** `main.py` lines 3101-3104

**Change:**
```python
else:
    # Fallback: Cannot identify zero-day samples with binary labels only
    logger.error(f"❌ CRITICAL: No multiclass labels or attack_cat available. Cannot identify zero-day samples.")
    logger.error(f"   Available keys in preprocessed_data: {list(self.preprocessed_data.keys())}")
    logger.error(f"   This will result in zero zero-day samples in evaluation!")
    logger.error(f"   Check preprocessing to ensure y_test_multiclass or test_attack_cat_original is stored.")
    zero_day_mask = torch.zeros(len(y_test_tensor), dtype=torch.bool, device=self.device)
```

---

## 📋 **Diagnostic Checklist**

When zero-day samples are 0, check:

1. ✅ **Is `y_test_multiclass` in preprocessed_data?**
   - Check diagnostic output: `'y_test_multiclass' in preprocessed_data: True/False`

2. ✅ **Does `y_test_multiclass` length match sequences?**
   - Check diagnostic output: `y_test_multiclass length: X`
   - Compare with: `y_test_tensor length: Y`
   - Should be: X == Y

3. ✅ **Does `y_test_multiclass` contain zero-day attack label?**
   - Check diagnostic output: `Zero-day label {label} in y_test_multiclass: True/False`
   - Check: `y_test_multiclass unique labels: [...]`

4. ✅ **Is `test_attack_cat_original` in preprocessed_data?**
   - Check diagnostic output: `'test_attack_cat_original' in preprocessed_data: True/False`

5. ✅ **Does `test_attack_cat_original` contain zero-day attack name?**
   - Check diagnostic output: `Zero-day attack '{name}' found in test_attack_cat_original: True/False`

6. ✅ **Was saved test set loaded?**
   - Check logs for: "Saved test set verified" or "Skipping saved test set"
   - If loaded, check if it had multiclass labels

---

## 🎯 **Most Likely Root Cause**

Based on the code analysis, **Hypothesis 1** is most likely:

**The saved test set is overwriting the correctly created `y_test_multiclass` and `test_attack_cat_original`, leaving the evaluation without the data needed to identify zero-day samples.**

**Solution:** Ensure saved test set validation rejects sets without required data, or ensure all required data is restored when loading saved test set.



