# Warning Root Cause Analysis: "Only 583 zero-day samples available, targeting 3500"

## 🔍 **Problem Description**

The warning appears every trial:
```
WARNING - ⚠️  Only 583 zero-day samples available, targeting 3500. Using all 583.
```

## 📊 **Root Cause Analysis**

### **Step-by-Step Logic Flow**

1. **Line 833-835**: `test_subset_size = 10000` (hardcoded to sample 10,000 test samples)
2. **Line 848**: `self._temp_zero_day_target = 0.35` (35% target for zero-day samples)
3. **Line 849**: `_stratified_test_subset()` is called with `n_samples = 10000`
4. **Line 561-562**: Inside `_stratified_test_subset`:
   ```python
   zero_day_target_percentage = 0.35  # From _temp_zero_day_target
   zero_day_target_count = int(n_samples * zero_day_target_percentage)
   zero_day_target_count = int(10000 * 0.35) = 3500  # ❌ PROBLEM!
   ```
5. **Line 604-605**: Code checks available zero-day samples:
   ```python
   available_zero_day = len(zero_day_indices)  # Only 583 samples exist!
   ```
6. **Line 609**: Calculates actual count:
   ```python
   actual_zero_day_count = min(3500, 583) = 583  # Falls back to available
   ```
7. **Line 611**: Warning triggers because `583 < 3500`

---

## ⚠️ **The Issue**

**The problem is the order of operations:**

1. ❌ **Code first calculates**: `target = 10000 * 0.35 = 3500`
2. ❌ **Then checks availability**: Only 583 exist
3. ❌ **Warning appears**: Because target > available

**What should happen:**

1. ✅ **First check availability**: 583 zero-day samples exist
2. ✅ **Then calculate reasonable subset size**: Based on available samples
3. ✅ **Or adjust target percentage**: Based on available samples

---

## 🎯 **Why This Happens**

### **Mathematical Mismatch**

- **Target**: 35% of 10,000 = **3,500 zero-day samples**
- **Reality**: Only **583 zero-day samples** exist in entire test dataset
- **Mismatch**: Trying to get 6x more samples than exist!

### **Why 10,000 samples?**

The code uses `test_subset_size = 10000` to maximize sequences after filtering, but:
- It doesn't consider that zero-day samples are **limited** (only 583 total)
- The target percentage (35%) assumes infinite zero-day samples
- The calculation is **backwards**: It should check availability FIRST

---

## 🔧 **The Fix**

### **Option 1: Check Availability First (Recommended)**

Modify `_stratified_test_subset` to check available zero-day samples **before** calculating target count:

```python
def _stratified_test_subset(self, X_test, y_test, y_test_multiclass, test_attack_cat, n_samples):
    # ... existing code ...
    
    # Check available zero-day samples FIRST
    zero_day_label = self.config.zero_day_attack_label
    if y_test_multiclass is not None:
        zero_day_mask = (y_test_multiclass == zero_day_label)
        zero_day_indices = np.where(zero_day_mask)[0]
        available_zero_day = len(zero_day_indices)
    else:
        available_zero_day = 0
    
    # Get target percentage
    zero_day_target_percentage = getattr(self, '_temp_zero_day_target', 0.30)
    
    # Calculate target count based on AVAILABLE samples, not arbitrary n_samples
    # If we want 35% zero-day, calculate max subset size that can achieve this
    if available_zero_day > 0:
        # Reverse calculation: if we have N zero-day samples and want P%,
        # then max subset size = N / P
        max_subset_size_by_zero_day = int(available_zero_day / zero_day_target_percentage)
        # Use the minimum of requested n_samples and what's achievable
        effective_n_samples = min(n_samples, max_subset_size_by_zero_day)
        zero_day_target_count = int(effective_n_samples * zero_day_target_percentage)
    else:
        effective_n_samples = n_samples
        zero_day_target_count = 0
    
    # Rest of the function uses effective_n_samples and zero_day_target_count
    # ...
```

### **Option 2: Adjust Target Based on Availability**

Calculate target percentage dynamically based on available samples:

```python
# After checking available_zero_day
if available_zero_day > 0:
    # Calculate what percentage we can actually achieve
    max_achievable_percentage = available_zero_day / n_samples
    # Use minimum of target and achievable
    zero_day_target_percentage = min(
        getattr(self, '_temp_zero_day_target', 0.30),
        max_achievable_percentage
    )
    zero_day_target_count = int(n_samples * zero_day_target_percentage)
    
    if zero_day_target_percentage < getattr(self, '_temp_zero_day_target', 0.30):
        logger.info(f"⚠️  Adjusted zero-day target from {getattr(self, '_temp_zero_day_target', 0.30)*100:.1f}% to {zero_day_target_percentage*100:.1f}% (limited by available samples: {available_zero_day})")
```

### **Option 3: Reduce test_subset_size**

Simply reduce the subset size to a more reasonable value:

```python
# Instead of 10000, use a value that makes sense given 583 zero-day samples
# If we want 35% zero-day, max subset = 583 / 0.35 ≈ 1665
max_reasonable_subset = min(
    int(available_zero_day / 0.35) if available_zero_day > 0 else 5000,
    10000
)
test_subset_size = min(max_reasonable_subset, len(self.preprocessed_data['X_test']))
```

---

## 📈 **Expected Behavior After Fix**

### **Before Fix:**
```
WARNING - ⚠️  Only 583 zero-day samples available, targeting 3500. Using all 583.
```

### **After Fix (Option 1):**
```
INFO - 📊 Available zero-day samples: 583
INFO - 📊 Adjusted subset size: 1665 (from 10000) to achieve 35% zero-day target
INFO - 📊 Zero-day target: 583 samples (35% of 1665)
✅ No warnings - target matches availability
```

### **After Fix (Option 2):**
```
INFO - ⚠️  Adjusted zero-day target from 35.0% to 5.8% (limited by available samples: 583)
INFO - 📊 Using all 583 zero-day samples
```

---

## 🎯 **Recommendation**

**Use Option 1** because:
- ✅ Prevents the warning by calculating achievable subset size first
- ✅ Maximizes test set size while respecting zero-day availability
- ✅ Maintains the target percentage (35%) when possible
- ✅ Provides clear logging of adjustments

This ensures the code calculates realistic targets based on actual data availability rather than arbitrary assumptions.










