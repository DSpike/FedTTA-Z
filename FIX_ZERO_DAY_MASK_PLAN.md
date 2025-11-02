# Fix Zero-Day Mask and Evaluation Plan

## 🎯 **Objective**

Fix the incorrect zero-day mask that marks ALL test samples as zero-day, causing misleading high base model performance. Implement proper evaluation that separately reports:

1. Performance on **zero-day attacks only**
2. Performance on **non-zero-day test samples**
3. **Overall performance**

---

## 📋 **Proposed Plan**

### **Phase 1: Fix Zero-Day Mask Creation** ⚠️ **CRITICAL**

**Problem**: Lines 1990-1997 in `main.py` incorrectly mark ALL test samples as zero-day.

**Solution**: Properly map zero-day indices from original data to sequence indices.

**Steps**:

1. **Understand sequence creation mapping**:

   - Original data indices → Sequence indices
   - When `create_sequences()` is called, multiple sequences can be created from one original sample
   - Need to track which sequences come from zero-day samples

2. **Fix `evaluate_base_model_only()` method** (lines 1965-2100):

   ```python
   # Instead of:
   zero_day_mask = torch.ones(len(y_test), dtype=torch.bool)  # ❌ WRONG

   # Do:
   # Map zero_day_indices from original data to sequence indices
   # If sequences were created, we need to find sequences originating from zero-day samples
   ```

3. **Fix `evaluate_adapted_model_only()` method** (similar issue around line 2150)

---

### **Phase 2: Separate Evaluation Reporting** 📊

**Problem**: Current evaluation mixes zero-day and non-zero-day samples.

**Solution**: Separate performance metrics for each category.

**Steps**:

1. **Add separate metric calculation**:

   ```python
   # Evaluate on zero-day attacks only
   zero_day_only_metrics corruption calculate_metrics(
       y_test[zero_day_mask],
       predictions[zero_day_mask]
   )

   # Evaluate on non-zero-day samples only
   non_zero_day_mask = ~zero_day_mask
   non_zero_day_metrics = calculate_metrics(
       y_test[non_zero_day_mask],
       predictions[non_zero_day_mask]
   )

   # Overall metrics (existing)
   overall_metrics = calculate_metrics(y_test, predictions)
   ```

2. **Update results dictionary** to include:
   - `zero_day_only`: {accuracy, precision, recall, f1_score, confusion_matrix}
   - `non_zero_day`: {accuracy, precision, recall, f1_score, confusion_matrix}
   - `overall`: {accuracy, precision, recall, f1_score, confusion_matrix}

---

### **Phase 3: Fix Sequence-to-Original Mapping** 🔄

**Problem**: When sequences are created, original data indices need to be mapped to sequence indices.

**Solution**: Track sequence-to-original mapping during preprocessing.

**Steps**:

1. **Modify preprocessor** to return mapping:

   ```python
   # In blockchain_federated_unsw_preprocessor.py
   def create_sequences(..., return_indices=True):
       # Track which original sample each sequence comes from
       original_indices = []
       # ... create sequences ...
       return X_seq, y_seq, original_indices
   ```

2. **Store mapping in preprocessed_data**:

   ```python
   preprocessed_data['test_sequence_to_original'] = original_indices
   ```

3. **Use mapping to create zero-day mask**:
   ```python
   # Map zero_day_indices (original data) to sequence indices
   sequence_to_original = preprocessed_data.get('test_sequence_to_original', None)
   if sequence_to_original is not None:
       zero_day_mask = torch.zeros(len(y_test), dtype=torch.bool)
       for seq_idx, orig_idx in enumerate(sequence_to_original):
           if orig_idx in zero_day_indices:
               zero_day_mask[seq_idx] = True
   ```

---

### **Phase 4: Update Visualization** 📈

**Problem**: Visualizations don't show separate zero-day vs non-zero-day performance.

**Solution**: Add separate plots and metrics.

**Steps**:

1. **Update `performance_visualization.py`**:

   - Add plot for zero-day-only performance comparison
   - Add plot for non-zero-day performance comparison
   - Update confusion matrices to show both

2. **Update JSON output** to include separate metrics

---

### **Phase 5: Validation & Testing** ✅

**Steps**:

1. **Verify zero-day mask**:

   ```python
   # Log zero-day mask statistics
   logger.info(f"Zero-day mask: {zero_day_mask.sum().item()}/{len(zero_day_mask)} samples")
   logger.info(f"Expected zero-day samples: {len(zero_day_indices)}")
   ```

2. **Compare performance**:

   - Zero-day-only accuracy should be **LOWER** than overall accuracy
   - Non-zero-day accuracy should be **HIGHER** than overall accuracy
   - This validates the fix is working

3. **Run full system test**:
   - Verify base model evaluation
   - Verify TTT model evaluation
   - Verify visualizations

---

## 🎯 **Implementation Priority**

### **High Priority (Must Fix)**:

1. ✅ Fix zero-day mask creation (Phase 1)
2. ✅ Add separate evaluation reporting (Phase 2)

### **Medium Priority (Should Fix)**:

3. ✅ Fix sequence-to-original mapping (Phase 3)
4. ✅ Update visualizations (Phase 4)

### **Low Priority (Nice to Have)**:

5. ✅ Comprehensive validation (Phase 5)

---

## 📝 **Expected Outcomes**

### **Before Fix**:

- Base model accuracy: **90%** (misleading - includes non-zero-day samples)
- Zero-day detection rate: **74.7%** (but this is across ALL samples)

### **After Fix**:

- **Zero-day-only accuracy**: ~50-70% (realistic for unseen attacks)
- **Non-zero-day accuracy**: ~85-95% (expected - model has seen these)
- **Overall accuracy**: ~70-80% (weighted average)

### **Key Insight**:

The base model **should** perform poorly on zero-day attacks (it hasn't seen them), which will show:

- TTT adaptation is **necessary** and **effective**
- True value of test-time training for zero-day detection

---

## 🔧 **Implementation Steps (Quick Start)**

1. **First, fix the immediate issue** (lines 1990-1997):

   - If no sequence mapping available, use a simpler approach:
   - Check if preprocessed_data has `zero_day_attack` label
   - Filter test samples where `y_test == zero_day_attack_label`

2. **Add logging** to verify the fix:

   ```python
   logger.info(f"Zero-day mask: {zero_day_mask.sum()}/{len(zero_day_mask)} samples")
   logger.info(f"Zero-day samples breakdown: {torch.bincount(y_test[zero_day_mask])}")
   ```

3. **Calculate separate metrics** and log them

4. **Test and verify** the fix works correctly

---

## ❓ **Questions to Resolve**

1. **Sequence mapping**: Do we have access to sequence-to-original mapping, or do we need to create it?
2. **Evaluation scope**: Should we evaluate on ALL test samples or just zero-day attacks for the final comparison?
3. **Metric priority**: Which metrics are most important for zero-day detection evaluation?

