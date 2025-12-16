# Zero-Day Configuration Mismatch - Critical Issue Found!

## 🚨 CRITICAL CONFIGURATION BUG DETECTED

Your system has **TWO DIFFERENT zero-day attacks configured** causing severe confusion!

---

## 📊 Evidence from Logs

### **Configuration Says:**
```
Line 11: zero_day_attack: DoS Hulk
Line 467: zero_day_attack: DoS Hulk (label: 4)
```

### **BUT Evaluation Uses:**
```
Line 1476: Zero-day attack: 'Generic', label: 1
Line 1564: Zero-day attack: 'Generic', label: 1
Line 1646: Zero-day attack: 'Generic', label: 1
Line 2186: Zero-day attack: 'Generic', label: 1
```

### **AND Preprocessing Shows:**
```
Creating zero-day split with 'DoS Hulk' as zero-day attack
✅ After post-sequence filtering: 45/180 zero-day sequences (25.0%) [TARGET: 25%]
```

### **BUT Final Evaluation Reports:**
```
🔍 Identified 4 zero-day sequences from 180 sequences (2.2%)
Zero-day attack: 'Generic', label: 1
```

---

## 🔍 What's Happening

### **Timeline of Confusion:**

1. **Preprocessing Stage:**
   - Uses "DoS Hulk" as zero-day ✅
   - Creates 45/180 (25%) zero-day sequences ✅
   - Correct configuration

2. **Evaluation Stage:**
   - **SWITCHES to "Generic"** ❌
   - Only finds 4/180 (2.2%) sequences ❌
   - Wrong attack type!

---

## 🚨 The Real Problem

**"Generic" is NOT your zero-day attack!**

Your actual zero-day configuration (from config):
```python
zero_day_attack: "DoS Hulk"  # This is correct
```

But somewhere in evaluation code, it's looking for "Generic" instead!

---

## 📊 Actual Test Set Composition

From preprocessing logs:

### **DoS Hulk (True Zero-Day):**
```
Before filtering: 59/195 sequences (30.3%)
After filtering:  45/180 sequences (25.0%) ✅ CORRECT
```

### **"Generic" (Wrong Label):**
```
Found in evaluation: 4/180 sequences (2.2%)
```

**"Generic" is NOT even a valid attack type in CICIDS2017!**

---

## 🔬 Root Cause Analysis

### **Issue #1: Label Mapping Bug**

Check your code for:
```python
# Somewhere in evaluation code
zero_day_label = 'Generic'  # ← BUG! Should be 'DoS Hulk'
```

### **Issue #2: Binary Classification Confusion**

From logs:
```
Available labels for training: [0, 1]
Test label distribution: tensor([ 59, 121])
Zero-day attack: 'Generic', label: 1
```

**Problem:** After binary conversion:
- Label 0 = Normal
- Label 1 = Attack (ALL attacks combined)

**"Generic"** is being used as placeholder for "all attacks" in binary mode!

### **Issue #3: Category Grouping Issue**

From config.py, you have:
```python
use_category_grouping: bool = True
```

**This might be mapping specific attacks to generic categories!**

---

## 🎯 Where the Bug Is

### **Most Likely Location:**

Check [main.py](main.py) evaluation section where it identifies zero-day:

```python
# Bug is probably here:
zero_day_attack = 'Generic'  # Should be config.zero_day_attack
# OR
zero_day_label = 1  # Using binary label instead of multiclass
```

### **Search for:**
```bash
grep -n "Generic" main.py
grep -n "zero_day_attack.*=.*Generic" main.py
```

---

## ✅ The Fix

### **Option 1: Fix Evaluation Code**

Find where evaluation sets `zero_day_attack = 'Generic'` and change to:
```python
zero_day_attack = config.zero_day_attack  # Use from config
```

### **Option 2: Use Multiclass Labels**

Ensure evaluation uses **multiclass labels** (0-14) not binary (0-1):
```python
# Current (WRONG):
zero_day_label = 1  # Binary: all attacks

# Fixed (CORRECT):
zero_day_label = 4  # Multiclass: DoS Hulk specifically
```

### **Option 3: Disable Category Grouping**

If category grouping is causing issues:
```python
# config.py
use_category_grouping: bool = False  # Use specific attack types
```

---

## 📊 Explaining Your Performance "Paradox"

### **What's Actually Happening:**

1. **Preprocessing creates 45 DoS Hulk samples (25%)** ✅

2. **Evaluation mistakenly looks for "Generic"** ❌

3. **Finds only 4 samples labeled "Generic"** (probably mislabeled data)

4. **These 4 samples are easy to detect** (outliers/errors)
   - Gets 100% accuracy on these 4 samples

5. **Evaluates known attacks on remaining 176 samples**
   - Gets 77.84% accuracy

**Result:** Appears that zero-day (100%) > known attacks (78%)

### **But Reality:**

- **True zero-day (DoS Hulk): 45 samples** - NOT evaluated!
- **"Generic" (4 samples): NOT your zero-day** - accidentally evaluated
- **Known attacks: evaluated correctly**

---

## 🔍 Diagnostic Commands

### **Find the Bug:**

```bash
# Search for 'Generic' in main.py
grep -n "Generic" main.py

# Search for zero_day_attack assignment
grep -n "zero_day_attack.*=" main.py

# Search for binary label usage in evaluation
grep -n "label.*1.*zero" main.py
```

### **Check Category Mapping:**

```python
# In config.py, check if 'Generic' is in attack_category_mapping
print(config.attack_category_mapping)
```

---

## ✅ Immediate Actions

1. **Find where "Generic" is set**
   ```bash
   grep -rn "Generic" main.py coordinators/ models/
   ```

2. **Check if using binary vs multiclass labels**
   - Binary: {0: Normal, 1: Attack} ❌ Too coarse
   - Multiclass: {0: Normal, 1: Bot, ..., 4: DoS Hulk, ...} ✅ Correct

3. **Verify zero_day_attack is passed correctly**
   ```python
   # Should be:
   zero_day_attack = config.zero_day_attack  # "DoS Hulk"
   # NOT:
   zero_day_attack = 'Generic'
   ```

4. **Check category_grouping setting**
   ```python
   # config.py
   use_category_grouping: bool = ???
   ```

---

## 🎯 Expected Behavior After Fix

### **With Correct DoS Hulk Evaluation (45 samples):**

```
DoS Hulk (Zero-Day):    70-85% ✅ Realistic
Known Attacks:          80-90% ✅ Better than zero-day
Overall:                75-88% ✅ In between
```

### **Current (WRONG - evaluating "Generic"):**

```
"Generic" (4 samples):  100% ← Not your zero-day!
Known Attacks:          78% ← Correct evaluation
Overall:                84% ← Mixed results
```

---

## Summary

**Root Cause:** Evaluation code uses "Generic" (label 1) instead of "DoS Hulk" (label 4)

**Why 100% on "Zero-Day":** Because it's evaluating 4 random "Generic" samples, not your actual 45 DoS Hulk zero-day samples

**Why Lower on Known:** Because known attack evaluation is correct (176 samples)

**Fix:** Change evaluation to use multiclass label 4 (DoS Hulk) instead of binary label 1 ("Generic"/all attacks)

**Impact:** After fix, you'll see realistic zero-day performance (70-85%) and known attacks will properly outperform (80-90%)
