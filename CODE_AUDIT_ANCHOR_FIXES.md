# Code Audit: Are Anchor Fixes Implemented?

## Summary

**Status**: ✅ **Anchor-based k-means initialization IS implemented** BUT ❌ **NOT being used in actual runs**

## What I Found in the Code

### Fix #1: Anchor-Based K-Means ✅ IMPLEMENTED (but not working)

**Location**: `coordinators/centralized_coordinator.py` lines 377-418

**Implementation**:
```python
# Generate anchor prototypes from training data
if self.train_data is not None and self.train_labels is not None:
    # Sample 5 shots per class from training data
    # Compute embeddings → Create prototypes
    # Use Nearest Anchor assignment instead of K-Means
    dists = euclidean_distances(embeddings_np, anchor_prototypes_np)
    cluster_labels = np.argmin(dists, axis=1)  # Assign to nearest anchor
```

**What it does**:
- Samples 5 examples per class from training data
- Computes "anchor prototypes" (Normal anchor, Attack anchor)
- Assigns test samples to nearest anchor (instead of blind k-means)
- This prevents class label swapping

**Problem**: NOT being used! Logs show:
```
✅ Using k-means clustering for pseudo-labels (k=2)  ← Basic k-means
```

**NOT**:
```
✅ Using Nearest Anchor assignment  ← Should see this!
```

**Why**: Either:
1. `self.train_data` or `self.train_labels` is None
2. Exception occurs during anchor generation (silently caught)
3. `anchor_prototypes_np` ends up None for some reason

### Fix #2: Confidence-Based Filtering ❌ NOT IMPLEMENTED

**What it would do**:
```python
# Only use high-confidence predictions for clustering
predictions = model(support_x)
confidence = softmax(predictions).max(dim=1)
confident_mask = confidence > 0.9
confident_samples = support_x[confident_mask]
# Cluster only confident samples
```

**Status**: NOT in the code

### Fix #3: Larger Support Set ❌ NOT IMPLEMENTED

**Current**:
```python
support_size = 100  # Fixed at 100 samples
```

**Better**:
```python
support_size = min(500, len(query_x) // 2)  # Adaptive, up to 500
```

**Status**: NOT in the code (still using 100)

### Fix #4: Disable Prototype Updates ❌ NOT IMPLEMENTED

**Current**:
```python
prototype_update_interval = 10  # Update every 10 steps
```

**Alternative to test**:
```python
prototype_update_interval = 999999  # Never update
```

**Status**: Still updating every 10 steps

### Fix #5: Consistency Regularization ❌ NOT IMPLEMENTED

**What it would do**:
```python
consistency_loss = (adapted_predictions - base_predictions).pow(2).mean()
total_loss = entropy_loss + pseudo_loss + 0.1 * consistency_loss
```

**Status**: NOT in the code

## Diagnostic Logging Added

I added extensive debug logging to understand why anchor-based assignment isn't working:

**Lines added**:
- Line 380: Check if train_data/train_labels available
- Lines 383-418: Detailed logging at each step of anchor generation

**What we'll see in next run**:
```
🔍 DEBUG: train_data available: True/False, train_labels available: True/False
🔍 Attempting to generate anchor prototypes from training data...
🔍 Found 2 classes in training data: [0, 1]
🔍 Class 0: 3476 samples, using 5 for anchors
🔍 Class 1: 689 samples, using 5 for anchors
🔍 Total anchor samples: 10
🔍 Computing embeddings for anchor samples...
🔍 Anchor embeddings shape: torch.Size([10, 256])
✅ Generated anchor prototypes from training data for KMeans alignment
🔍 Anchor prototypes shape: (2, 256)
```

OR if it fails:
```
⚠️ Failed to generate anchor prototypes: [error message]
🔍 Traceback: [full traceback]
```

## Next Steps

### Step 1: Diagnose Why Anchors Not Generated ⏳ IN PROGRESS

Running system with debug logging to find out:
- Is `self.train_data` None?
- Is there an exception during anchor generation?
- What exactly is failing?

### Step 2: Fix the Root Cause

Once we know why, we can fix it:

**If `train_data` is None**:
- Check `distribute_data()` is called correctly
- Verify data is stored properly

**If exception during generation**:
- Fix the specific error
- May need to handle edge cases

**If anchors generated but not used**:
- Check the `if anchor_prototypes_np is not None` condition
- Verify numpy conversion is working

### Step 3: Implement Additional Fixes (if needed)

If anchor-based assignment alone doesn't fix the issue:
1. Add confidence-based filtering
2. Increase support set size
3. Try disabling prototype updates

## Expected Outcome

Once anchor-based assignment works:

**Before (basic k-means)**:
```
Cluster 0: [N, N, A, N, A]  ← Mixed! Wrong!
Cluster 1: [A, N, A, A, N]  ← Mixed! Wrong!
```

**After (anchor-based)**:
```
Cluster 0 (nearest to Normal anchor):  [N, N, N, N]  ← Pure! ✓
Cluster 1 (nearest to Attack anchor):  [A, A, A, A, A]  ← Pure! ✓
```

**Performance**:
- Base: 81.75%
- TTT: **~84-86%** (instead of 21%)

## Files Modified

1. `coordinators/centralized_coordinator.py`
   - Lines 380, 383-418: Added debug logging

## Current Run

Running with debug logging: `run_with_debug.log`
- Started: 2025-12-15 20:18
- Status: Training phase (waiting for TTT phase)
