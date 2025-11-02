# TTT vs Base Model Distribution Comparison

## 🔍 **Key Finding: Different Sampling Methods**

The **distribution will NOT be exactly the same** between base model and TTT, but for different reasons at different stages.

---

## 📊 **Distribution Breakdown**

### **1. TTT Adaptation Query Set** (During Adaptation)

**Location**: `main.py` lines 2124-2126

```python
query_size = min(200, len(X_test))
query_indices = torch.randperm(len(X_test))[:query_size]
query_x = X_test_tensor[query_indices]
```

**Sampling Method**: **Random sampling** (NOT stratified)

**Distribution**:

- ✅ Uses same test set as base model
- ❌ **Random sampling may not preserve exact distribution**
- ❌ **Could have 25% zero-day or 50% zero-day** depending on random selection
- ⚠️ **No guarantee of maintaining 30% Normal / 40% Zero-day / 30% Other**

**Impact**: The TTT adaptation happens on a randomly sampled subset, which may have a different class distribution than the full test set.

---

### **2. Base Model Evaluation** (After Training)

**Location**: `main.py` lines 1980-2015

```python
X_test = self.preprocessed_data['X_test']  # Full test set
y_test = self.preprocessed_data['y_test']  # Full test set
```

**Sampling Method**: **Full test set** (no sampling)

**Distribution**:

- ✅ **Full test set**: ~332 samples (from our investigation)
- ✅ **Distribution**: 30% Normal, ~40% Zero-day, ~30% Other attacks
- ✅ **St metaphorical**: Preprocessor maintains this distribution

---

### **3. TTT Model Evaluation** (After Adaptation)

**Location**: `main.py` lines 2168-2203

```python
X_test = self.preprocessed_data['X_test']  # Full test set
y_test = self.preprocessed_data['y_test']  # Full test set
```

**Sampling Method**: **Full test set** (no sampling)

**Distribution**:

- ✅ **Full test set**: Same as base model (~332 samples)
- ✅ **Distribution**: Same as base model (30% Normal, ~40% Zero-day, ~30% Other)
- ✅ **Fair comparison**: Both evaluated on identical test set

---

## 🎯 **Summary**

| Stage               | Sampling Method      | Distribution      | Same as Base Model? |
| ------------------- | -------------------- | ----------------- | ------------------- |
| **TTT Adaptation**  | Random (200 samples) | ❌ **May differ** | ❌ **No**           |
| **Base Evaluation** | Full test set        | ✅ 30/40/30       | ✅ **N/A**          |
| **TTT Evaluation**  | Full test set        | ✅ 30/40/30       | ✅ **Yes**          |

---

## ⚠️ **Important Implications**

### **For TTT Adaptation**:

- **Random sampling** means the adaptation query set might have:
  - More zero-day samples (harder adaptation)
  - Fewer zero-day samples (easier adaptation)
  - Different Normal/Attack ratio

### **For Final Evaluation**:

- ✅ **Both models use the same full test set**
- ✅ **Same distribution guaranteed**
- ✅ **Fair comparison**

---

## 🔧 **Potential Issue**

### **Problem**: Random Sampling for TTT Adaptation

The TTT adaptation uses **random sampling** which may not reflect the true test distribution. This could lead to:

- Adaptation on an easier subset (fewer zero-day samples)
- Adaptation on a harder subset (more zero-day samples)
- Inconsistent adaptation quality across runs

### **Solution Options**:

1. **Use Stratified Sampling** (Recommended):

   ```python
   # Instead of random sampling, use stratified sampling
   from sklearn.model_selection import train_test_split
   _, query_x, _, query_y = train_test_split(
       X_test_tensor, y_test_tensor,
       train_size=200,
       stratify=y_test_tensor,  # Maintain class distribution
       random_state=42
   )
   ```

2. **Use Full Test Set** (If memory allows):

   ```python
   query_x = X_test_tensor  # Use all samples
   ```

3. **Use Stratified Subset** (If preprocessor has method):
   ```python
   if hasattr(self.preprocessor, 'sample_stratified_subset'):
       query_x, query_y = self.preprocessor.sample_stratified_subset(
           X_test_tensor, y_test_tensor, n_samples=200
       )
   ```

---

## ✅ **Current State**

### **Evaluation Distribution**: ✅ **SAME**

- Base model and TTT model are evaluated on **identical full test set**
- Distribution: 30% Normal, ~40% Zero-day, ~30% Other attacks

### **Adaptation Distribution**: ⚠️ **MAY DIFFER**

- TTT adaptation uses **random 200 samples**
- Distribution may not match test set distribution
- Could affect adaptation quality

---

## 🎯 **Recommendation**

**For Fair Evaluation**: ✅ **Current setup is fine**

- Both models evaluated on same full test set
- Distribution is identical
- Fair comparison guaranteed

**For Better TTT Adaptation**: 🔧 **Consider stratified sampling**

- Maintains test distribution during adaptation
- More consistent adaptation quality
- Better reflects real-world scenario

---

## 📝 **Conclusion**

**Question**: "Is the distribution going to be the same for TTT also?"

**Answer**:

- ✅ **YES** for final evaluation (both use full test set with same distribution)
- ❌ **NO** for TTT adaptation (uses random 200 samples, distribution may differ)
- ⚠️ **This is acceptable** but could be improved with stratified sampling

The important thing is that **final evaluation uses the same distribution**, which it does! The adaptation distribution difference is less critical but could be improved.

