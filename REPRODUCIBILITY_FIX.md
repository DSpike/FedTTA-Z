# Result Fluctuation Analysis and Fix

## Problem
Results fluctuate every time `main.py` is run, even though a global seed (42) is set at the top.

## ✅ FIXES IMPLEMENTED

### 1. Added `set_deterministic_seed()` Function
- Created helper function to reset all random seeds consistently
- Located in `main.py` after global seed initialization
- Can be called before critical random operations

### 2. Fixed TTT Query Set Sampling (Line 3874)
- **Before**: `torch.randperm(len(X_test))[:query_size]`
- **After**: Uses `torch.Generator` with fixed seed
- **Impact**: TTT query set is now deterministic

### 3. Fixed Support Set Sampling (Lines 3273, 3283, 4138, 4193, 5524)
- **Before**: `torch.randperm()` without seed
- **After**: Uses `torch.Generator` with fixed seed
- **Impact**: Support sets are now deterministic

### 4. Fixed Support Indices Batch Sampling (Line 2206)
- **Before**: `torch.randperm(len(test_data_subset))`
- **After**: Uses `torch.Generator` with fixed seed
- **Impact**: Batch support sets are deterministic

### 5. Fixed Prototype Computation (centralized_coordinator.py)
- **Before**: `torch.randperm(len(indices))` without seed
- **After**: Uses `torch.Generator` with fixed seed
- **Impact**: Prototype computation is deterministic

## Root Causes

### 1. **Random State Consumption**
- Global seed is set once at the top
- But random state gets consumed during execution
- Multiple `torch.randperm()` calls without resetting seed
- Each run may have different number of operations, leading to different random states

### 2. **Unseeded Random Operations**
Found these unseeded random operations:
- **Line 3874**: `torch.randperm(len(X_test))` - TTT query set sampling
- **Line 3273**: `torch.randperm(len(X_val_tensor))` - Support set sampling
- **Line 3283**: `torch.randperm(len(X_test_filtered))` - Fallback support set
- **Line 4110**: `torch.randperm(len(X_test_tensor))` - Support indices sample
- **Line 2190**: `torch.randperm(len(test_data_subset))` - Support indices batch
- **Line 5496**: `torch.randperm(len(X_test_tensor))` - Support indices
- **Line 5894**: `torch.randperm(len(normal_indices))` - Normal query indices

### 3. **Meta-Task Creation Randomness**
- `create_meta_tasks()` in `transductive_fewshot_model.py` uses `torch.randperm()` multiple times
- No seed reset before meta-task creation
- Different task compositions each run

### 4. **TTT Adaptation Batch Sampling**
- TTT adaptation uses random batch sampling
- `torch.randperm()` used for mini-batching without seed reset
- Different batch orders lead to different gradients

### 5. **Test Set Creation Randomness**
- Stratified sampling uses `np.random.choice()` with local seeds (42, 43, 44)
- But sequence creation and filtering may have different random states
- Post-sequence filtering randomness

## Solution

### Fix 1: Create Seed Management Function
Add a helper function to reset seeds consistently:

```python
def set_deterministic_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### Fix 2: Reset Seed Before Critical Operations
Reset seed before:
1. Test set creation
2. Meta-task creation
3. Support set sampling
4. TTT query set sampling
5. TTT adaptation batch sampling

### Fix 3: Use Generator for PyTorch Randomness
Use `torch.Generator()` with fixed seed for `torch.randperm()`:

```python
generator = torch.Generator()
generator.manual_seed(42)
indices = torch.randperm(n, generator=generator)
```

### Fix 4: Fix TTT Adaptation Randomness
In `coordinators/centralized_coordinator.py`, reset seed before batch sampling:

```python
# Before TTT loop
torch.manual_seed(42)
generator = torch.Generator(device=query_x.device)
generator.manual_seed(42)

# In TTT loop
if use_mini_batch:
    batch_indices = torch.randperm(len(query_x), generator=generator)[:batch_size]
```

## Implementation Priority

1. **High Priority**: Fix TTT query set sampling (line 3874)
2. **High Priority**: Fix support set sampling (lines 3273, 3283)
3. **Medium Priority**: Fix meta-task creation randomness
4. **Medium Priority**: Fix TTT adaptation batch sampling
5. **Low Priority**: Fix all other torch.randperm calls

## Expected Outcome

After fixes:
- ✅ Same test set every run
- ✅ Same support set every run
- ✅ Same TTT query set every run
- ✅ Same meta-tasks every run
- ✅ Same TTT adaptation batches every run
- ✅ Reproducible results within floating-point precision

## Note on Floating-Point Precision

Even with all seeds fixed, small variations (< 0.01%) may occur due to:
- Floating-point arithmetic order
- CPU vs GPU numerical differences
- PyTorch version differences

These are expected and acceptable for research reproducibility.

