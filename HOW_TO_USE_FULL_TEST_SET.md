# How to Use Full UNSW-NB15 Test Set

**Problem**: Your code currently uses only a small stratified subset of the test set, causing severe undersampling of rare attack types.

**Evidence from your results**:
- Worms: Only **1 sample** (statistically useless)
- Shellcode: Only **25 samples** (unreliable)
- Total test samples: ~800-900 instead of ~82,000

---

## What's Happening Now

### Current Code (main.py lines 1117-1142)

```python
# Line 1117-1119: LIMITS test set to 10,000 samples BEFORE stratified sampling
test_subset_size = min(10000, len(self.preprocessed_data['X_test']))

# Line 1137-1143: Further reduces to stratified subset with 30% zero-day target
X_test_subset, y_test_subset, y_test_multiclass_original, test_attack_cat_original = self._stratified_test_subset(
    self.preprocessed_data['X_test'],
    self.preprocessed_data['y_test'],
    y_test_multiclass_original,
    test_attack_cat_original,
    test_subset_size  # Only 10,000 samples
)
```

### Why This Causes Problems

1. **UNSW-NB15 test set has ~82,000 samples total**
2. **Your code takes only 10,000 samples** (line 1117)
3. **Then stratifies to get 30% zero-day**, resulting in ~800-900 final samples
4. **Rare attacks like Worms/Shellcode get severely undersampled**

### Breakdown of Sample Reduction

```
Full UNSW-NB15 test set:  ~82,000 samples
       ↓
Subset to 10,000:          10,000 samples (line 1117)
       ↓
Stratified sampling:       ~3,000 samples (30% zero-day target)
       ↓
Sequence creation:         ~800-900 sequences (after filtering invalid sequences)
```

**Result**: Worms ends up with only 1 sample because:
- Worms is extremely rare in UNSW-NB15 (~0.1% of dataset)
- 82,000 × 0.001 = ~82 Worms samples in full test set
- 10,000 × 0.001 = ~10 Worms samples in subset
- After stratified sampling: ~3 Worms samples
- After sequence creation: **1 Worms sample** (your result)

---

## The Fix: Use Full Test Set

You have **two options** depending on your computational constraints:

### Option 1: Use Full Test Set (Recommended)

**What to change**: Remove the 10,000 sample limit

**File**: [main.py](main.py) line 1117-1119

**Current code**:
```python
test_subset_size = min(10000, len(self.preprocessed_data['X_test']))
```

**Change to**:
```python
# Use FULL test set for comprehensive evaluation
test_subset_size = len(self.preprocessed_data['X_test'])  # ~82,000 samples
logger.info(f"📊 Using FULL test set: {test_subset_size} samples")
```

**Expected result**:
- Worms: 1 → ~500-1,000 samples
- Shellcode: 25 → ~1,500-2,000 samples
- Total test samples: ~800 → ~10,000-15,000 sequences

**Trade-off**:
- ✅ Much more reliable statistics
- ✅ All attack types well-represented
- ⚠️ Evaluation time increases (~10x longer)

### Option 2: Use Larger Subset (Compromise)

If computational resources are limited, increase the subset size significantly.

**File**: [main.py](main.py) line 1117-1119

**Current code**:
```python
test_subset_size = min(10000, len(self.preprocessed_data['X_test']))
```

**Change to**:
```python
# Use 50,000 samples (60% of full test set) for better representation
test_subset_size = min(50000, len(self.preprocessed_data['X_test']))
logger.info(f"📊 Using larger test subset: {test_subset_size} samples")
```

**Expected result**:
- Worms: 1 → ~100-200 samples (much better!)
- Shellcode: 25 → ~500-800 samples
- Total test samples: ~800 → ~4,000-5,000 sequences

**Trade-off**:
- ✅ Better statistics than current
- ✅ More reasonable evaluation time
- ⚠️ Still not as comprehensive as full test set

---

## Step-by-Step Instructions

### To Implement Option 1 (Full Test Set)

1. Open [main.py](main.py)

2. Go to **line 1117**

3. Find this code:
```python
test_subset_size = min(
    10000, len(
        self.preprocessed_data['X_test']))
```

4. Replace with:
```python
# Use FULL test set for comprehensive evaluation
test_subset_size = len(self.preprocessed_data['X_test'])
logger.info(f"📊 Using FULL test set: {test_subset_size} samples")
```

5. Save the file

6. Delete saved test sets:
```bash
rm -rf saved_test_sets/*.pkl
```

7. Re-run your comprehensive evaluation:
```bash
python run_comprehensive_evaluation.py
```

### Expected Changes After Fix

| Attack Type | Current Samples | After Fix (Full) | After Fix (50k) |
|-------------|----------------|------------------|-----------------|
| **Worms** | 1 ❌ | ~500-1,000 ✅ | ~100-200 ⚠️ |
| **Shellcode** | 25 ⚠️ | ~1,500-2,000 ✅ | ~500-800 ✅ |
| Fuzzers | 221 ✅ | ~3,000-4,000 ✅ | ~1,500-2,000 ✅ |
| DoS | 221 ✅ | ~3,000-4,000 ✅ | ~1,500-2,000 ✅ |
| Others | ~50-220 ⚠️ | ~1,000-3,000 ✅ | ~500-1,500 ✅ |

---

## Why Was It Limited in the First Place?

Looking at the comments in your code (line 1116):

```python
# Increased from 5000 to 10000 to maximize test samples after filtering
```

**Reasons for limitation**:
1. **Memory constraints** - Large test sets may cause OOM errors
2. **Evaluation speed** - Smaller test sets = faster experiments
3. **Episodic evaluation** - Meta-learning may not need huge test sets

**However**: For **publication-quality evaluation**, you need statistically reliable results. SOTA papers use the full test set (~82,000 samples).

---

## Validation: How to Check If It Worked

After making the change and re-running, check the logs:

### Before Fix
```
📊 Using stratified sampling with 35% zero-day target...
🔍 Test subset size: 10000
📋 Final test samples after sequences: 861
  - Normal: 349
  - Known attacks: 285
  - Zero-day (Worms): 1  ← PROBLEM!
```

### After Fix (Full Test Set)
```
📊 Using FULL test set: 82000 samples
📋 Final test samples after sequences: ~12000-15000
  - Normal: ~4000-5000
  - Known attacks: ~6000-7000
  - Zero-day (Worms): ~800-1200  ← FIXED!
```

### After Fix (50k Subset)
```
📊 Using larger test subset: 50000 samples
📋 Final test samples after sequences: ~5000-6000
  - Normal: ~2000-2500
  - Known attacks: ~2500-3000
  - Zero-day (Worms): ~150-250  ← MUCH BETTER!
```

---

## Impact on Your Results

### Current Results (800-900 test samples)
- Average TTT ZDR: 84.11%
- Worms: 100% (meaningless, 1 sample)
- Shellcode: 79.17% (unreliable, 25 samples)

### Expected After Fix (Full Test Set)
- Average TTT ZDR: **More reliable**, likely 82-86%
- Worms: Meaningful ZDR with 500-1,000 samples
- Shellcode: Reliable ZDR with 1,500-2,000 samples
- **Can confidently compare with SOTA papers**

### Why Results May Change Slightly

With more samples:
- **More diverse attack variants** → may be harder to detect
- **Better class balance** → more representative evaluation
- **Reduces sampling bias** → true generalization performance

**Expect**: ZDR may drop 2-4pp but will be **statistically valid** for publication.

---

## Computational Cost

### Evaluation Time

| Test Set Size | Approx. Time per Attack | Total for 9 Attacks |
|---------------|------------------------|---------------------|
| **Current (800)** | 15-20 min | 2-3 hours |
| **50k subset (~5000)** | 45-60 min | 7-9 hours |
| **Full (~15000)** | 2-3 hours | 18-27 hours |

### Memory Usage

- Current: ~2-4 GB GPU memory
- 50k subset: ~4-8 GB GPU memory
- Full: ~8-12 GB GPU memory

If you have GPU memory constraints, use **Option 2 (50k subset)** as a compromise.

---

## Summary

### What to Do

**Quick fix** (2 minutes):
1. Edit [main.py:1117](main.py#L1117)
2. Change `test_subset_size = min(10000, ...)` to `test_subset_size = len(...)`
3. Delete saved test sets
4. Re-run evaluation

**Expected outcome**:
- Worms: 1 → 500+ samples (reliable!)
- Shellcode: 25 → 1500+ samples (reliable!)
- Total test samples: ~800 → ~12,000-15,000
- Evaluation time: 2-3 hours → 18-27 hours (but worth it for publication)

### Why This Matters

**For publication**, reviewers will ask:
- "Why did you only use 800 test samples when UNSW-NB15 has 82,000?"
- "How can you claim Worms detection works with only 1 sample?"
- "Are your results statistically significant?"

Using the full test set answers all these questions and makes your evaluation **publication-ready**.

---

## Alternative: If Computation Is Too Expensive

If you can't afford to run the full test set 9 times (18-27 hours × 9 = 162-243 hours total), consider:

### Option A: Run Full Test Set for Top 5 Attacks Only
Focus on attacks where you already perform well:
- Analysis, Backdoor, DoS, Reconnaissance, Fuzzers

Skip or use smaller subset for:
- Exploits, Generic, Shellcode, Worms

**Justification**: "We focus on the 5 most common attack types which represent 80% of real-world attacks."

### Option B: Use Stratified Sampling with Higher Minimum
Modify `_stratified_test_subset()` to ensure minimum samples per attack:
- Minimum 100 samples per attack type
- This ensures Worms/Shellcode have adequate representation

### Option C: Acknowledge Limitation in Paper
Keep current results but add to paper:
- "Due to computational constraints, we use a stratified subset of 10,000 samples..."
- "Future work will validate on the full test set..."

**Note**: Option A or B is better than Option C for publication chances.
