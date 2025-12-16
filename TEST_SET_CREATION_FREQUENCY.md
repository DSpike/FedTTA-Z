# Test Set Creation Frequency - When Is It Created?

## ❌ **NO - Test Set is NOT Created Every Time**

The test set creation depends on whether a **saved test set** exists and is valid.

---

## 🔄 **Two Scenarios**

### **Scenario 1: Saved Test Set Exists & Is Valid** ✅

**What Happens:**

1. **System checks for saved test set** (line 745)

   ```python
   saved_test_set = self._load_saved_test_set()
   # Looks for: saved_test_sets/test_set_best_trial.pkl
   # Or: saved_test_sets/test_set_trial_13.pkl
   ```

2. **A NEW test set is still created** (during preprocessing steps 2.2-2.5)

   - Preprocessor creates initial test split
   - Sequences are created from subset
   - Post-sequence filtering is applied
   - Result: New test set with correct composition (~752 sequences)

3. **Saved test set is validated** (line 1049-1110)

   - ✅ Check zero-day attack matches
   - ✅ Check sizes match
   - ✅ Check composition is correct (~25% zero-day, not 100%)

4. **If saved test set is VALID:**
   - Newly created test set is **replaced** with saved test set
   - Saved test set is used (for reproducibility)

**Result:** Test set is **created but then replaced** - uses saved test set.

---

### **Scenario 2: No Saved Test Set OR Saved Test Set Invalid** 🆕

**What Happens:**

1. **System checks for saved test set** (line 745)

   ```python
   saved_test_set = self._load_saved_test_set()
   # Returns None if not found
   ```

2. **A new test set is created** (steps 2.2-2.5)

   - Preprocessor creates initial test split
   - Sequences are created
   - Post-sequence filtering applied
   - Result: New test set (~752 sequences)

3. **Saved test set validation skipped** (because `saved_test_set is None`)

4. **New test set is used**

**Result:** Test set is **created and used** - brand new test set.

---

## 📋 **When Test Set IS Created**

A new test set is created when:

1. ✅ **No saved test set exists** (`saved_test_sets/test_set_best_trial.pkl` not found)
2. ✅ **Saved test set has different zero-day attack** (e.g., saved="Exploits", current="Backdoor")
3. ✅ **Saved test set has wrong size** (e.g., saved=445 sequences, new=752 sequences)
4. ✅ **Saved test set has wrong composition** (e.g., 100% zero-day instead of 25%)
5. ✅ **`skip_saved_test_set=True` is passed** (used during optimization)

---

## 📋 **When Saved Test Set IS Used**

The saved test set is used when:

1. ✅ **Saved test set exists** (`saved_test_sets/test_set_best_trial.pkl` found)
2. ✅ **Zero-day attack matches** (saved="Exploits", current="Exploits")
3. ✅ **Sizes match** (saved=752 sequences, new=752 sequences)
4. ✅ **Composition is correct** (~25% zero-day, not 100% or 0%)

---

## 🔍 **Code Flow**

```python
# In main.py → preprocess_data()
def preprocess_data(self, skip_saved_test_set: bool = False):
    # Step 1: Check for saved test set
    saved_test_set = None
    if not skip_saved_test_set:
        saved_test_set = self._load_saved_test_set()  # ← Returns None if not found

    # Step 2: Run preprocessor (ALWAYS creates initial test data)
    self.preprocessed_data = self.preprocessor.preprocess_unsw_dataset(...)

    # Step 3: Create sequences from test data (ALWAYS happens)
    X_test_seq, y_test_seq = create_sequences(...)
    # Apply filtering to achieve 25% zero-day

    # Step 4: Check if saved test set should be used
    if saved_test_set is not None:
        # Validate saved test set
        if (zero_day_attack_matches AND
            sizes_match AND
            composition_correct):
            # Use saved test set (replaces newly created one)
            self.preprocessed_data['X_test'] = saved_test_set['X_test']
        else:
            # Use newly created test set
            pass
    else:
        # No saved test set - use newly created one
        pass
```

---

## 🎯 **Summary**

| Situation                             | Test Set Created?     | Test Set Used     |
| ------------------------------------- | --------------------- | ----------------- |
| **No saved test set**                 | ✅ Yes                | ✅ Newly created  |
| **Saved test set exists & valid**     | ✅ Yes (but replaced) | ✅ Saved test set |
| **Saved test set exists but invalid** | ✅ Yes                | ✅ Newly created  |
| **`skip_saved_test_set=True`**        | ✅ Yes                | ✅ Newly created  |

---

## 💡 **Key Points**

1. **A new test set is ALWAYS created** during preprocessing (steps 2.2-2.5)
2. **But it may be replaced** if a valid saved test set exists
3. **This ensures reproducibility** - same test set across runs
4. **But allows flexibility** - new test set if configuration changes

---

## 🔧 **How to Force New Test Set Creation**

### **Option 1: Delete Saved Test Set**

```bash
# On Windows PowerShell
Remove-Item saved_test_sets\test_set_best_trial.pkl
Remove-Item saved_test_sets\test_set_trial_13.pkl
```

### **Option 2: Change Zero-Day Attack**

```python
# In config.py
zero_day_attack: str = "Backdoor"  # Different from saved test set
```

### **Option 3: Change Sequence Parameters**

```python
# In config.py
sequence_length: int = 30  # Different from saved test set
sequence_stride: int = 15  # Different from saved test set
```

---

## 📝 **Example Logs**

### **When Saved Test Set is Used:**

```
📦 Loading saved test set from: saved_test_sets/test_set_best_trial.pkl
✅ Loaded test set from trial 13
📦 Found saved test set - will use it after preprocessing
...
🔄 Checking saved test set from optimization trial...
✅ Saved test set verified: 752 sequences, 752 multiclass labels, zero-day attack: 'Exploits'
✅ Test set replaced: 752 sequences
   Using test set from trial 13
```

### **When New Test Set is Created:**

```
⏭️  No saved test set found - will create new test set
...
✅ After post-sequence filtering: 188/752 zero-day sequences (25.0%) [TARGET: 25%]
✅ Verified alignment: All test sequences have length 752 after filtering
```

---

## 🎯 **Bottom Line**

**Test set creation happens every time, but:**

- If saved test set exists and is valid → **saved test set is used** (new one is discarded)
- If no saved test set or invalid → **newly created test set is used**

This design ensures:

- ✅ **Reproducibility** when using saved test sets
- ✅ **Flexibility** when configuration changes
- ✅ **Consistency** in evaluation conditions








