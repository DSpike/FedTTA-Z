# Test Set Size Difference Explanation

## 🔍 **Problem**
The current run has **100 test samples** (20 zero-day, 80 non-zero-day), but the optimization trial had a larger test set (potentially 200-400+ samples).

## 🎯 **Root Cause**

The test set size is determined by **post-sequence filtering** that maintains a target zero-day percentage. The filtering logic works as follows:

### **Post-Sequence Filtering Logic**

1. **After sequence creation**, the system counts available zero-day sequences
2. **For a 20% zero-day target**, it calculates: `non-zero-day = 4 × zero-day` (since 20% zero-day = 80% non-zero-day)
3. **Total test set = zero-day + non-zero-day**

### **Why Smaller Test Set?**

The test set size depends on:
- ✅ **Available zero-day sequences** after sequence creation
- ✅ **Sequence length** (47 in optimized config vs 31 in old)
- ✅ **Sequence stride** (13)
- ✅ **Pre-sequence sampling** (10,000 samples, but filtered to maintain ratio)

### **Current Run Analysis**

From the logs:
- **Final test set**: 100 sequences
- **Zero-day**: 20 sequences (20%)
- **Non-zero-day**: 80 sequences (80%)

This means only **20 zero-day sequences** were available after sequence creation, so:
- `non-zero-day = 20 × 4 = 80`
- `total = 20 + 80 = 100`

## 🔄 **Comparison: Optimization vs Current Run**

| Factor | Optimization Trial | Current Run | Impact |
|--------|-------------------|-------------|---------|
| **Sequence Length** | 31 | **47** | ⬇️ Fewer sequences (longer windows) |
| **Sequence Stride** | 13 | 13 | Same |
| **Pre-sequence sampling** | 10,000 | 10,000 | Same |
| **Zero-day target** | 30% (likely) | **20%** | ⬇️ Lower target = smaller set |
| **Available zero-day sequences** | ~60-80 | **~20** | ⬇️ Much fewer |

## 📊 **Why Fewer Zero-Day Sequences?**

1. **Longer sequence length (47 vs 31)**:
   - Creates fewer overlapping sequences
   - Formula: `sequences = (samples - sequence_length) / stride + 1`
   - More aggressive filtering removes overlapping sequences

2. **Post-sequence filtering maintains ratio**:
   - Target: 20% zero-day (reduced from 30%)
   - If only 20 zero-day sequences available → total = 100
   - If 60 zero-day sequences were available → total = 300

3. **Available zero-day samples in pre-sequence data**:
   - The pre-sequence stratified sampling targets 35% zero-day
   - But sequence creation dilutes this
   - Final available zero-day sequences depend on original distribution

## ✅ **Solutions to Increase Test Set Size**

### **Option 1: Increase Pre-Sequence Sampling**
```python
# In main.py, line 890-892
test_subset_size = min(15000, len(self.preprocessed_data['X_test']))  # Increase from 10k
```

### **Option 2: Adjust Zero-Day Target Percentage**
```python
# In main.py, line 963
target_zero_day_percentage = 0.15  # Reduce from 0.20 to allow more total samples
```

### **Option 3: Use Smaller Sequence Length**
```python
# In config.py
sequence_length: int = 31  # Reduce from 47 to create more sequences
```

### **Option 4: Use Saved Test Set from Optimization**
The optimization trial saved its test set. You can:
- Use the saved test set from trial 6 (if available)
- Or ensure the current run uses the same configuration as optimization

## 🎯 **Recommended Fix**

**For reproducibility with optimization trial**, you have two options:

### **A. Use Saved Test Set** (Best for comparison)
```python
# The system already tries to load saved test sets
# Check if saved_test_sets/test_set_best_trial.pkl exists
# If not, the optimization trial didn't save it (saving was added later)
```

### **B. Match Optimization Configuration**
To get similar test set sizes, ensure:
1. **Same sequence length** as optimization (check optimization logs)
2. **Same zero-day target percentage** (check optimization logs)
3. **Same pre-sequence sampling size** (10k is already set)

## 📝 **Notes**

1. **The smaller test set doesn't affect accuracy metrics** - it's just fewer samples
2. **The 100-sample test set is valid** - it maintains the 20% zero-day ratio
3. **For fair comparison**, use the same test set (saved from optimization) or match the exact configuration

## 🔍 **How to Check Optimization Trial Configuration**

If you want to see what the optimization trial used:
1. Check optimization logs for sequence_length
2. Check optimization logs for zero-day target percentage
3. Check if `saved_test_sets/test_set_trial_6.pkl` exists

The optimization trial likely had:
- **Sequence length**: 31 (not 47)
- **Zero-day target**: 30% (not 20%)
- **Result**: ~200-400 test samples

---

**Summary**: The test set is smaller because longer sequences (47 vs 31) create fewer sequences, and only 20 zero-day sequences were available, resulting in 100 total sequences (20 + 80).










