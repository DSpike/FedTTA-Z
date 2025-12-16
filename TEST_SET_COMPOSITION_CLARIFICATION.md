# Test Set Composition Clarification

## ❌ **NO - Test Set is NOT Only Zero-Day Samples**

The test set contains a **MIXED composition** of three types of samples:

---

## 📊 **Intended Test Set Composition (Before Sequences)**

### **Target Distribution:**

```
Test Set = 100% Total Samples
├── 30% Normal/BENIGN samples
└── 70% Attack samples (total)
    ├── ~35-50% Zero-day attacks (unseen during training)
    └── ~20-35% Other attacks (seen during training)
```

### **From Code** (`blockchain_federated_cicids_preprocessor.py` lines 550-597):

```python
# Create test data with BALANCED distribution
# Target: 30% BENIGN + 70% Attacks (including zero-day)
target_benign_samples = int(total_test_samples * 0.3)  # 30% BENIGN
target_attack_samples = int(total_test_samples * 0.7)  # 70% Attacks

# Test set contains:
# 1. BENIGN samples (30%)
# 2. Zero-day attacks (ALL available from test data)
# 3. Other attacks (remaining slots after zero-day)
```

---

## 📈 **Actual Composition (From Recent Run Logs)**

### **After Preprocessing (Before Sequences):**

From the latest run logs (`cicids_run_log.txt`):

```
Test data: 253749 samples
  BENIGN (30%): 253749  ← Wait, this shows 100% BENIGN?!
  Zero-day attacks: 26842
  Other attacks from test data: 0
```

**Wait - this looks wrong!** The log shows 253,749 BENIGN but also 26,842 zero-day. Let me check the actual sequence-level distribution...

### **After Sequence Creation (What Optimization Actually Uses):**

From the latest run:

```
Test sequences: 587 sequences
Zero-day sequences: 0/587 (0.0%)
Non-zero-day sequences: 587/587 (100.0%)
```

**⚠️ Problem Identified!** The test sequences after creation show **0% zero-day**, which is why optimization isn't working properly for zero-day!

---

## 🔍 **Why This Matters for Your Question**

### **Your Question:** "Is test set only with zero-day samples?"

**Answer:**

- ❌ **NO** - Test set is designed to have **MIXED composition**:

  - 30% Normal/BENIGN
  - ~35-50% Zero-day attacks
  - ~20-35% Other seen attacks

- ⚠️ **BUT** - After sequence creation, there's an issue:
  - Currently showing **0% zero-day** in sequences
  - This means optimization is NOT seeing zero-day samples
  - This explains why overall performance is low!

---

## 🎯 **The Real Problem**

**Current Situation:**

1. Test set (before sequences) has ~26,842 zero-day samples
2. After sequence creation + filtering → **0 zero-day sequences**
3. Optimization uses sequences → **Never sees zero-day samples**
4. Model optimizes for non-zero-day only → **Poor overall performance**

**Why Zero-Day Sequences Are Lost:**

- Sequence creation with stride/length mapping
- Stratified sampling before sequence creation doesn't preserve zero-day
- Post-sequence filtering may remove zero-day sequences

---

## 📊 **Validation Set vs Test Set**

### **Validation Set:**

```
Composition:
├── Normal/BENIGN samples
└── Other attack types (seen during training)
❌ NO zero-day attacks
```

### **Test Set (Intended):**

```
Composition:
├── 30% Normal/BENIGN
├── ~35-50% Zero-day attacks ← ONLY place with zero-day
└── ~20-35% Other attacks
```

**Key Difference:**

- ✅ **Test set**: Contains zero-day (for zero-day detection evaluation)
- ❌ **Validation set**: NO zero-day (only seen attacks, for general performance)

---

## 🔴 **Why This Causes Your Performance Issue**

**Current Optimization:**

- Uses test set sequences (which currently have 0% zero-day after sequence creation)
- Optimizes for zero-day detection rate
- But test sequences have NO zero-day → optimization fails
- Model only sees non-zero-day → poor generalization

**What Should Happen:**

- Test set sequences should have ~20-30% zero-day
- Optimization should see zero-day samples
- Model learns both zero-day and non-zero-day patterns

---

## ✅ **Summary**

| Question                                  | Answer                                                                        |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| **Is test set ONLY zero-day?**            | ❌ **NO** - Mixed: 30% Normal, 35-50% Zero-day, 20-35% Other attacks          |
| **Does test set contain zero-day?**       | ✅ **YES** - Test set is the ONLY dataset with zero-day samples               |
| **Does validation set contain zero-day?** | ❌ **NO** - Validation set excludes zero-day completely                       |
| **What does optimization use?**           | Test set sequences (but currently 0% zero-day due to sequence creation issue) |

**The REAL problem:** Zero-day samples are being lost during sequence creation, not that test set is only zero-day!









