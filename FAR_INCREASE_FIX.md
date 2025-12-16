# Fix: TTT FAR Increase Issue

## 🔍 Problem Identified

The TTT FAR increased after implementing the `far_optimized` strategy. Root causes:

1. **Missing Variable Definition**: `zero_day_mask_np` was referenced before being defined, causing the strategy to fail and fall back to PR-optimized (which has no FAR constraint)

2. **Fallback to PR-Optimized**: If `far_optimized` strategy failed, it would fall back to PR-optimized, which doesn't prioritize FAR and can have high FAR

3. **Search Logic Issue**: The threshold search might not be finding optimal thresholds if model probabilities are not well-calibrated

## ✅ Fixes Applied

### 1. **Fixed Variable Definition** (`main.py` line ~4081)

- Moved `zero_day_mask_np` definition to the top of the threshold optimization block
- Now available to all strategies (far_optimized, balanced_zdr_far, zdr_optimized)

### 2. **Improved FAR-Optimized Strategy** (`main.py` lines ~4085-4160)

- Wrapped in try-except to prevent fallback to PR-optimized
- Always returns a threshold (even if FAR > 1% target)
- Uses best available threshold (lowest FAR found) if target cannot be met
- Better error handling with high threshold (0.95) fallback

### 3. **Enhanced Search Strategy**

- Search from high to low thresholds (0.99 → 0.7)
- Tracks best threshold even if FAR slightly exceeds target
- Prioritizes thresholds with FAR ≤ 1%, then maximizes ZDR within constraint

### 4. **Better Fallback Handling**

- If `far_optimized` fails, uses high threshold (0.95) instead of PR-optimized
- Prevents fallback to strategies without FAR constraint

## 📊 Expected Behavior

### Before Fix:

- `far_optimized` strategy fails → falls back to PR-optimized
- PR-optimized has no FAR constraint → FAR can be high (20-50%)

### After Fix:

- `far_optimized` strategy always returns a threshold
- Uses best available threshold (lowest FAR found)
- If strategy fails, uses high threshold (0.95) as fallback
- FAR should be minimized (target: < 1%, fallback: lowest possible)

## 🔧 Additional Recommendations

If FAR is still high after this fix:

1. **Increase Confidence Rejection Threshold**: Already set to 0.90, but can increase to 0.95
2. **Improve Model Calibration**: Use temperature scaling or Platt scaling
3. **Adjust Search Range**: If needed, search even higher thresholds (0.95-0.99)
4. **Check Model Probabilities**: Verify that attack probabilities are well-calibrated

## 🎯 Testing

Run the code and check logs for:

- `✅ Threshold Strategy: FAR-optimized`
- `Results: FAR=X.XXXX` (should be ≤ 0.01 or as low as possible)
- `Selected threshold: X.XXXX` (should be high, e.g., 0.85-0.99)



