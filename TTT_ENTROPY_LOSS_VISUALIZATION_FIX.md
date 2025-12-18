# TTT Entropy Loss Visualization Fix

## 🐛 **Problem Identified**

The entropy loss was not showing properly in the TTT adaptation plot because of a **scale mismatch** between entropy loss and L2 regularization loss.

---

## 🔍 **Root Cause**

### **The Issue**:

1. **L2 Regularization Loss Storage**:
   - L2 reg loss is stored as **UNWEIGHTED** value (raw L2 sum)
   - From run: L2 reg (unweighted) = 0.0 → 0.7892
   - L2 reg (weighted with 0.01) = 0.0 → 0.007892

2. **Entropy Loss Values**:
   - Entropy loss = 0.0979 → 0.0294
   - Much smaller than unweighted L2 reg

3. **Y-Axis Scale Problem**:
   - Plot code used **unweighted L2 reg** (0-0.79) to set y-axis range
   - Entropy loss (0.03) appeared as tiny flat line at bottom
   - **Entropy loss was invisible** due to scale mismatch

---

## ✅ **Fix Applied**

### **Changes Made**:

1. **Get L2 Weight at Function Start** (lines 244-253):
   ```python
   # Get L2 regularization weight for proper scaling
   l2_weight = 0.01  # Default
   try:
       from config_loader import get_dataset_config
       config = get_dataset_config()
       l2_weight = getattr(config, 'ttt_l2_reg_weight', 0.01)
   except:
       l2_weight = ttt_adaptation_data.get('ttt_l2_reg_weight', 0.01)
   ```

2. **Apply Weight When Plotting L2 Reg** (line 648):
   ```python
   # CRITICAL FIX: Apply weight for proper scale
   l2_float = [float(l2) * l2_weight for l2 in l2_reg_losses]
   ax2.plot(steps_float, l2_float, 'r-', ...,
            label=f'L2 Regularization Loss (×{l2_weight:.3f})')
   ```

3. **Apply Weight in Y-Axis Range Calculation** (line 664):
   ```python
   # CRITICAL: Apply L2 weight for proper scale (L2 stored is unweighted)
   l2_float = [float(l2) * l2_weight for l2 in l2_reg_losses]
   y_min = min(y_min, min(l2_float))
   y_max = max(y_max, max(l2_float))
   ```

---

## 📊 **Before vs After**

### **Before Fix**:

| Component | Value Range | Visibility |
|-----------|-------------|------------|
| **Total Loss** | 0.10 → 0.04 | ✅ Visible |
| **Entropy Loss** | 0.098 → 0.029 | ❌ **Invisible** (tiny at bottom) |
| **L2 Reg (unweighted)** | 0.0 → 0.79 | ✅ Visible (dominates scale) |

**Problem**: Y-axis range = 0 to 0.79 (set by unweighted L2 reg)
- Entropy loss (0.029) appears as flat line at bottom
- Not visible in plot

---

### **After Fix**:

| Component | Value Range | Visibility |
|-----------|-------------|------------|
| **Total Loss** | 0.10 → 0.04 | ✅ Visible |
| **Entropy Loss** | 0.098 → 0.029 | ✅ **Now Visible** |
| **L2 Reg (weighted)** | 0.0 → 0.0079 | ✅ Visible (proper scale) |

**Solution**: Y-axis range = 0 to 0.10 (includes all components)
- Entropy loss (0.029) now clearly visible
- L2 reg (0.0079) also visible
- All components on same scale

---

## 🎯 **Expected Result**

After the fix, the TTT adaptation plot will show:

1. **Total Loss** (blue line): 0.10 → 0.04
2. **Entropy Loss** (magenta line): 0.098 → 0.029 ✅ **Now visible**
3. **L2 Regularization Loss** (red line): 0.0 → 0.0079 ✅ **Proper scale**

All three components will be clearly visible on the same plot with appropriate scaling.

---

## 🔧 **Technical Details**

### **Why L2 Reg is Stored Unweighted**:

In `coordinators/centralized_coordinator.py` (line 554):
```python
# L2 reg is stored BEFORE applying weight
reg_loss = l2_reg  # This is the unweighted value (0-0.79)
adaptation_data['l2_reg_losses'].append(reg_loss.item())
```

The weight (0.01) is applied later in the loss calculation:
```python
total_loss = total_loss + ttt_config.ttt_l2_reg_weight * l2_reg
```

### **Why This Matters**:

- **For Loss Calculation**: Weight is applied correctly
- **For Visualization**: Need to apply weight to match actual contribution
- **For Comparison**: Entropy and L2 reg should be on same scale

---

## ✅ **Verification**

To verify the fix works:

1. Run the system again
2. Check the TTT adaptation plot
3. Verify entropy loss is now visible (magenta line)
4. Verify all three loss components are on the same scale

---

**Fix Applied**: Entropy loss visualization now works correctly  
**Status**: Ready for testing  
**Files Modified**: `visualization/performance_visualization.py`

