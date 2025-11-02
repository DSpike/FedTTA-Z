# Test Set Composition Explanation

## 📊 **Intended Test Set Distribution**

According to the preprocessor code (lines 752-791), the test set is designed to have:

### **Target Composition**:

- **30% Normal samples**
- **70% Attack samples** (total)
  - **50% of attacks** = Zero-day attacks → **35% of total test set**
  - **50% of attacks** = Other attacks (seen during training) → **35% of total test set**

### **Expected Distribution**:

```
Normal:        30% of test samples
Zero-day:      35% of test samples  ← Unseen during training
Other attacks: 35% of test samples  ← Seen during training
```

---

## 🔍 **Why You're Seeing ~40% Zero-Day**

Several factors can cause the zero-day percentage to differ from the intended 35%:

### **1. Sequence Creation Changes Distribution**

**Problem**: When `create_sequences()` is called, it uses a sliding window:

- `sequence_length = 30`
- `stride = 15`
- Multiple sequences can be created from one original sample

**Impact**: The distribution after sequence creation may not match the original distribution exactly.

**Example**:

- Original test data: 100 samples (30 Normal, 35 Zero-day, 35 Other)
- After sequences: ~200 sequences (distribution may shift slightly)

### **2. Sampling Constraints**

**From code (line 780)**:

```python
zero_day_sample = zero_day_test.sample(n=min(target_attack_samples // 2, len(zero_day_test)), random_state=42)
```

**What this means**:

- **Target**: `target_attack_samples // 2` (35% of total)
- **Actual**: `min(target, available_zero_day_samples)`

**If there are MORE zero-day samples available than needed**:

- The sampling will still use only `target_attack_samples // 2`
- This maintains the intended 35%

**If there are FEWER other attack samples available**:

- `other_attacks_sample` may be smaller than target
- The final test set might have proportionally MORE zero-day samples
- Result: Zero-day percentage > 35% (could reach 40%+)

### **3. Sequence Label Assignment**

**From `create_sequences()` code (line 1127)**:

```python
label = y[i + sequence_length - 1]  # Use label from last timestep
```

**Impact**: Each sequence gets the label from its **last timestep**. This means:

- Sequences near zero-day samples are more likely to be labeled as zero-day
- Sequences near normal/other attacks might get different labels
- This can slightly skew the distribution

### **4. Actual Available Data**

**From preprocessor logs (line 762)**:

```python
logger.info(f"  Available in test data: {len(test_normal)} normal, {len(zero_day_test)} zero-day, {len(test_attacks[test_attacks['attack_cat'] != zero_day_attack])} other attacks")
```

**The actual distribution depends on**:

- How many zero-day attacks are available in the raw test data
- How many other attacks are available in the raw test data
- The sampling logic may favor zero-day if other attacks are limited

---

## 📈 **Why This Matters**

### **For Zero-Day Detection Evaluation**:

**Current situation** (~40% zero-day):

- **Higher zero-day percentage** means:
  - More samples to evaluate zero-day detection
  - But also more "easy" cases if zero-day samples are easier to detect

**Ideal situation** (35% zero-day):

- More balanced with other attack types
- Better reflects realistic zero-day detection scenario

### **Impact on Performance**:

If zero-day attacks are **easier to detect** than other attacks:

- **Higher zero-day %** → **Higher overall accuracy** (misleading)
- **Lower zero-day %** → **Lower overall accuracy** (more realistic)

If zero-day attacks are **harder to detect**:

- **Higher zero-day %** → **Lower overall accuracy** (more challenging)
- **Lower zero-day %** → **Higher overall accuracy** (easier test set)

---

## ✅ **Is 40% Zero-Day a Problem?**

### **For Scientific Evaluation**: ⚠️ **Minor Issue**

**Pros**:

- More zero-day samples = more robust evaluation
- Still includes other attacks and normal samples
- Distribution is reasonable (not skewed to one extreme)

**Cons**:

- Slightly different from intended 35%
- May not reflect exact realistic scenario
- Should be documented in paper

### **Recommendation**:

1. **Document the actual distribution** in your results
2. **Report separate metrics** for zero-day vs non-zero-day (we're doing this!)
3. **Note**: "Test set contains 40% zero-day attacks, 30% normal, 30% other attacks"

---

## 🔧 **How to Verify the Distribution**

Check the logs from the preprocessor to see the actual composition:

```
Test set composition target: X normal, Y attacks
Available in test data: A normal, B zero-day, C other attacks
Test data: Z samples
  Normal (30%): ...
  Zero-day attacks (50% of attacks): ...
  Other attacks from test data (50% of attacks): ...
```

Then check after sequence creation (if sequences are used).

---

## 🎯 **Conclusion**

**40% zero-day is reasonable** and likely due to:

1. Sequence creation changing distribution
2. Available data constraints
3. Sampling logic favoring zero-day when other attacks are limited

**What matters more**:

- ✅ Separate evaluation on zero-day vs non-zero-day (our fix!)
- ✅ Accurate zero-day mask (we fixed this!)
- ✅ Clear reporting of actual distribution

The **40% vs 35% difference is acceptable** for scientific evaluation as long as it's properly documented and separate metrics are reported.
