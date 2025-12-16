# Test Sample Count Explanation

## 🔍 Why Test Samples Reduced from 332 → 123

The test sample count is determined by **three sequential filtering stages**:

---

## 📊 **Three-Stage Filtering Process**

### **Stage 1: Pre-Sequence Sampling**

- **Input**: Full test dataset
- **Action**: Samples up to **5,000 samples** with **40% zero-day target** (before sequence creation)
- **Purpose**: Ensure enough zero-day samples before sequence creation (will be diluted)
- **Result**: ~5,000 samples (40% zero-day, 60% non-zero-day)

---

### **Stage 2: Sequence Creation**

- **Input**: ~5,000 samples from Stage 1
- **Formula**: `num_sequences = (num_samples - sequence_length) // stride + 1`
- **Parameters**:
  - `sequence_length = 30`
  - `stride = 15`
- **Calculation**: `(5000 - 30) // 15 + 1 = 332 sequences`
- **Result**: **~332 sequences** (zero-day percentage diluted from 40% → lower %)

**Why sequences are fewer than samples**:

- Each sequence uses 30 timesteps
- Sequences overlap with stride=15
- Formula accounts for overlap and reduces total count

---

### **Stage 3: Post-Sequence Filtering (30% Zero-Day Target)**

- **Input**: ~332 sequences from Stage 2
- **Target**: Maintain **exactly 30% zero-day sequences**
- **Logic**:
  1. Use **ALL available zero-day sequences** (`target_zero_day_count = available_zero_day`)
  2. Calculate needed non-zero-day sequences: `M = (7/3) * N` (to maintain 30:70 ratio)
  3. Filter to maintain exact ratio

**Example Calculation**:

- **Available**: 37 zero-day sequences, ~295 non-zero-day sequences
- **Target zero-day**: 37 (use all available)
- **Target non-zero-day**: `(37 * 7) // 3 = 86` (to maintain 30% ratio)
- **Final total**: `37 + 86 = 123 sequences` ✅

**Result**: **123 sequences** (37 zero-day = 30.1%, 86 non-zero-day = 69.9%)

---

## 🔍 **Why Total Count Varies**

The final test sample count depends on:

### **1. Available Zero-Day Sequences After Sequence Creation**

- **More zero-day sequences** → **More total sequences** (formula: `total = zero_day + (7/3)*zero_day`)
- **Fewer zero-day sequences** → **Fewer total sequences**

### **2. Attack Type Distribution**

- **Different attack types** have different distributions in the dataset
- **Backdoor**: May have fewer samples → fewer sequences
- **DoS/Generic**: May have more samples → more sequences

### **3. Sequence Creation Dilution**

- Sequence creation reduces zero-day percentage (from 40% → lower %)
- If initial zero-day samples are spread out, sequence creation may miss some

---

## 📈 **Historical Counts**

| Run      | Attack Type | Pre-Seq Samples | Sequences Created | Final After Filtering | Zero-Day Count |
| -------- | ----------- | --------------- | ----------------- | --------------------- | -------------- |
| Initial  | Various     | 5,000           | ~332              | 332                   | ~2 (0.6%)      |
| Previous | Backdoor    | 5,000           | ~332              | 190+                  | ~57 (30%)      |
| Current  | Backdoor    | 5,000           | ~332              | **123**               | **37 (30.1%)** |

---

## 🔧 **Factors That Control Test Sample Count**

### **1. Pre-Sequence Sampling (`test_subset_size`)**

```python
test_subset_size = min(5000, len(X_test))  # Line 823
```

- **Increase** this → More sequences created → Potentially more final sequences
- **Default**: 5,000 samples

### **2. Sequence Creation Parameters**

```python
sequence_length = 30  # config.py
stride = 15           # config.py
```

- **Smaller stride** → More sequences (less overlap) → Potentially more final sequences
- **Larger stride** → Fewer sequences (more overlap) → Fewer final sequences

### **3. Post-Sequence Filtering Logic**

```python
target_zero_day_count = available_zero_day  # Use all available
target_non_zero_day_count = int((target_zero_day_count * 7) // 3)  # Maintain 30% ratio
```

- **30% zero-day target** is fixed
- **Formula**: `total = N + (7/3)*N` where `N = zero_day_count`
- **Limited by**: Available zero-day sequences after sequence creation

---

## 🎯 **Why 123 is Correct**

The current result (**123 sequences**) is **correct** because:

1. ✅ **30% zero-day ratio maintained**: 37/123 = 30.1%
2. ✅ **Uses all available zero-day sequences**: 37 (maximizes zero-day coverage)
3. ✅ **Maintains exact ratio**: Formula ensures 30:70 split
4. ✅ **Consistent across runs**: Same filtering logic ensures reproducibility

**The lower count is a consequence of**:

- Limited zero-day sequences available after sequence creation (37 sequences)
- Maintaining strict 30% zero-day ratio (prioritizes quality over quantity)
- Attack type distribution (Backdoor may have fewer samples in dataset)

---

## 🔧 **How to Increase Test Sample Count**

If you want more test samples, you can:

### **Option 1: Increase Pre-Sequence Sampling**

```python
test_subset_size = min(10000, len(X_test))  # Increase from 5000 to 10000
```

- **Impact**: More sequences created → Potentially more zero-day sequences → More final sequences

### **Option 2: Reduce Sequence Stride**

```python
sequence_stride = 10  # Reduce from 15 to 10 (more overlap = more sequences)
```

- **Impact**: More sequences from same input → Potentially more zero-day sequences → More final sequences
- **Trade-off**: More overlap between sequences (may affect independence)

### **Option 3: Adjust Zero-Day Target Percentage**

```python
target_zero_day_percentage = 0.20  # Reduce from 30% to 20%
```

- **Impact**: Formula changes from `M = (7/3)N` to `M = 4N` → More total sequences
- **Trade-off**: Fewer zero-day samples in test set

---

## 📊 **Summary**

**Current Configuration**:

- Pre-sequence: 5,000 samples (40% zero-day target)
- Sequence creation: ~332 sequences (stride=15, length=30)
- Post-sequence filtering: **123 sequences** (30% zero-day, 70% non-zero-day)

**Key Limiting Factor**:

- **Available zero-day sequences after sequence creation** (37 sequences)
- **30% zero-day target** ensures quality but limits total count
- **Formula**: `total = zero_day + (7/3)*zero_day = 37 + 86 = 123` ✅

The count is **lower than before** because:

1. **Post-sequence filtering** was added to maintain exact 30% ratio
2. **Backdoor attack** may have fewer samples in the dataset
3. **Sequence creation** may have missed some zero-day samples due to distribution

**This is expected behavior** - the filtering ensures high-quality test set with proper zero-day distribution, even if total count is reduced.









