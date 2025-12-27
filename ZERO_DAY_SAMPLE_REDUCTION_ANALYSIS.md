# Zero-Day Sample Reduction Analysis: 706 → 46

## 🔍 **Problem Summary**

**Observation**: Only **46 zero-day samples** are evaluated when **706 DoS Hulk samples** are available in the raw test data.

**Question**: Why is there such a massive reduction (706 → 46, ~93.5% loss)?

---

## 📊 **Root Cause: Sequence Label Mapping Logic**

### **The Problem**

The sequence label mapping at line 1170 in `main.py` uses a **restrictive formula** that only checks the **last timestep** of each sequence:

```python
original_idx = seq_idx * sequence_stride + (sequence_length - 1)
```

**What this means:**
- For `sequence_stride = 15` and `sequence_length = 30`
- Sequences are created at positions: 0, 15, 30, 45, 60, 75, ...
- Labels are assigned by checking positions: **29, 44, 59, 74, 89, ...** (last timestep of each sequence)
- **Only zero-day samples at these specific positions are captured!**

### **Why This Causes Massive Loss**

**Example Scenario:**
- **706 DoS Hulk samples** distributed throughout test data
- **Sequence stride = 15**: Only checks positions 29, 44, 59, 74, 89, 104, ...
- **If DoS Hulk samples are at positions**: 30, 31, 32, 45, 46, 60, 61, ...
- **Result**: Many zero-day samples are **missed** because they're not at the exact positions checked!

**Mathematical Impact:**
- With stride=15, only **1 out of every 15 positions** is checked
- If zero-day samples are randomly distributed, only **~6.7%** (1/15) would be captured
- **706 × 0.067 ≈ 47 samples** ← This matches the observed 46 samples!

---

## 🔬 **Three-Stage Filtering Process**

### **Stage 1: Pre-Sequence Sampling**
- **Input**: Full test dataset (706 DoS Hulk samples available)
- **Action**: Stratified sampling with 30% zero-day target
- **Result**: ~5,000 samples (should include ~1,500 DoS Hulk samples if available)

### **Stage 2: Sequence Creation**
- **Input**: ~5,000 samples from Stage 1
- **Formula**: `num_sequences = (num_samples - sequence_length) // stride + 1`
- **Parameters**: `sequence_length = 30`, `stride = 15`
- **Calculation**: `(5000 - 30) // 15 + 1 = 332 sequences`
- **Label Mapping**: Only checks last timestep of each sequence (positions: 29, 44, 59, ...)
- **Result**: **Only ~46 zero-day sequences** captured (due to restrictive label mapping)

### **Stage 3: Post-Sequence Filtering**
- **Input**: ~332 sequences from Stage 2 (only 46 zero-day sequences)
- **Target**: 25% zero-day ratio
- **Formula**: `total = zero_day + (3/1)*zero_day = 46 + 138 = 184 sequences`
- **But**: If non-zero-day sequences are limited, total may be reduced further
- **Result**: **46 zero-day sequences** (limited by available zero-day sequences from Stage 2)

---

## ⚠️ **The Critical Issue**

### **Current Label Mapping Logic (Line 1170)**

```python
original_idx = seq_idx * sequence_stride + (sequence_length - 1)
```

**Problems:**
1. **Only checks last timestep**: Ignores all other timesteps in the sequence
2. **Sparse sampling**: With stride=15, only 1/15 positions are checked
3. **Misses distributed samples**: If zero-day samples are spread out, most are missed
4. **Position-dependent**: Only works if zero-day samples happen to be at checked positions

### **Why This Is Problematic**

**Scenario 1: Zero-Day Samples at Wrong Positions**
- DoS Hulk samples at positions: 30, 31, 32, 45, 46, 60, 61, ...
- Sequence label mapping checks: 29, 44, 59, 74, ...
- **Result**: Most zero-day samples are missed!

**Scenario 2: Zero-Day Samples in Sequence Middle**
- Sequence spans positions 0-29 (last timestep = 29)
- DoS Hulk sample at position 15 (middle of sequence)
- **Result**: Sequence is NOT labeled as zero-day (only checks position 29)

**Scenario 3: Zero-Day Samples at Sequence Start**
- Sequence spans positions 0-29 (last timestep = 29)
- DoS Hulk sample at position 0 (start of sequence)
- **Result**: Sequence is NOT labeled as zero-day (only checks position 29)

---

## 🔧 **Proposed Solutions**

### **Solution 1: Check ANY Timestep in Sequence (Recommended)**

**Change label mapping to check if ANY timestep in the sequence is zero-day:**

```python
# Instead of only checking last timestep:
original_idx = seq_idx * sequence_stride + (sequence_length - 1)

# Check ALL timesteps in the sequence:
sequence_start = seq_idx * sequence_stride
sequence_end = min(sequence_start + sequence_length, orig_len)
sequence_indices = range(sequence_start, sequence_end)

# Check if ANY timestep is zero-day
is_zero_day = False
for idx in sequence_indices:
    if idx < orig_len:
        label = y_test_multiclass_original[idx].item() if torch.is_tensor(y_test_multiclass_original[idx]) else y_test_multiclass_original[idx]
        if label == zero_day_attack_label:
            is_zero_day = True
            break

# Assign label based on zero-day presence
if is_zero_day:
    y_test_multiclass_seq.append(zero_day_attack_label)
else:
    # Use last timestep as fallback
    original_idx = sequence_end - 1
    original_label = y_test_multiclass_original[original_idx].item() if torch.is_tensor(y_test_multiclass_original[original_idx]) else y_test_multiclass_original[original_idx]
    y_test_multiclass_seq.append(original_label)
```

**Benefits:**
- ✅ Captures ALL zero-day samples in sequences
- ✅ No position-dependent bias
- ✅ Maximizes zero-day sequence count
- ✅ More accurate zero-day detection evaluation

**Expected Impact:**
- **Before**: ~46 zero-day sequences (6.7% capture rate)
- **After**: ~100-200 zero-day sequences (much higher capture rate)
- **Total test set**: Increases from 46 to 200-400 sequences

---

### **Solution 2: Majority Vote**

**Use majority vote to determine sequence label:**

```python
sequence_start = seq_idx * sequence_stride
sequence_end = min(sequence_start + sequence_length, orig_len)
sequence_labels = []

for idx in range(sequence_start, sequence_end):
    if idx < orig_len:
        label = y_test_multiclass_original[idx].item() if torch.is_tensor(y_test_multiclass_original[idx]) else y_test_multiclass_original[idx]
        sequence_labels.append(label)

# Use most common label (majority vote)
from collections import Counter
label_counts = Counter(sequence_labels)
most_common_label = label_counts.most_common(1)[0][0]
y_test_multiclass_seq.append(most_common_label)
```

**Benefits:**
- ✅ Handles mixed sequences (zero-day + non-zero-day)
- ✅ More robust to noise
- ✅ Better represents sequence content

**Trade-offs:**
- ⚠️ May dilute zero-day sequences if they're minority in sequence
- ⚠️ More computationally expensive

---

### **Solution 3: Reduce Stride (Quick Fix)**

**Reduce sequence stride to check more positions:**

```python
# Current: stride = 15 (checks 1/15 positions)
# Proposed: stride = 5 (checks 1/5 positions)
sequence_stride = 5  # Instead of 15
```

**Benefits:**
- ✅ Simple change
- ✅ Checks 3x more positions
- ✅ Captures more zero-day samples

**Trade-offs:**
- ⚠️ Creates more sequences (3x more)
- ⚠️ More overlap between sequences
- ⚠️ Still misses samples at unchecked positions

**Expected Impact:**
- **Before**: ~46 zero-day sequences (stride=15)
- **After**: ~138 zero-day sequences (stride=5, 3x improvement)
- **Still not optimal**: Only checks 1/5 positions instead of all

---

## 📈 **Expected Results After Fix**

### **Current (Restrictive Label Mapping)**
- **Zero-day samples available**: 706
- **Zero-day sequences captured**: ~46 (6.5% capture rate)
- **Total test sequences**: ~184 (46 zero-day + 138 non-zero-day)

### **After Solution 1 (Check ANY Timestep)**
- **Zero-day samples available**: 706
- **Zero-day sequences captured**: ~150-250 (21-35% capture rate)
- **Total test sequences**: ~600-1000 (150-250 zero-day + 450-750 non-zero-day)

### **After Solution 2 (Majority Vote)**
- **Zero-day samples available**: 706
- **Zero-day sequences captured**: ~100-200 (14-28% capture rate)
- **Total test sequences**: ~400-800 (100-200 zero-day + 300-600 non-zero-day)

### **After Solution 3 (Reduce Stride)**
- **Zero-day samples available**: 706
- **Zero-day sequences captured**: ~138 (19.5% capture rate)
- **Total test sequences**: ~552 (138 zero-day + 414 non-zero-day)

---

## 🎯 **Recommendation**

**Use Solution 1 (Check ANY Timestep)** because:
1. ✅ **Maximizes zero-day capture**: Captures ALL zero-day samples in sequences
2. ✅ **No position bias**: Works regardless of sample distribution
3. ✅ **Better evaluation**: More accurate zero-day detection metrics
4. ✅ **Logical correctness**: If a sequence contains zero-day samples, it should be labeled as zero-day

**Implementation Priority**: **HIGH** - This is a critical bug that severely limits zero-day detection evaluation.

---

## 📊 **Summary**

| Stage | Current Behavior | Issue | Impact |
|-------|-----------------|-------|--------|
| **Pre-Sequence** | 706 DoS Hulk samples available | ✅ OK | None |
| **Sequence Creation** | Creates ~332 sequences | ✅ OK | None |
| **Label Mapping** | Only checks last timestep (positions: 29, 44, 59, ...) | ❌ **CRITICAL** | **93.5% loss** (706 → 46) |
| **Post-Sequence Filter** | Maintains 25% ratio | ✅ OK | Limited by Stage 3 output |

**Root Cause**: Restrictive label mapping (only checks last timestep) causes massive zero-day sample loss.

**Fix**: Check ANY timestep in sequence for zero-day samples (Solution 1).

---

*Documentation Date: December 17, 2025*  
*Code Reference: `main.py` lines 1169-1175*



