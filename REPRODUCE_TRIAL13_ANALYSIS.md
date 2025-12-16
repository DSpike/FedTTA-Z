# Reproducing Trial 13 Results - Analysis

## The Problem

**Optimization Trial 13 Results:**
- Base Model: Accuracy=78.33%, F1=74.26%, AUC-PR=86.89%, **ZDR=91.67%** ⚠️
- TTT Model: Accuracy=80.00%, F1=80.17%, AUC-PR=72.65%, **ZDR=100.00%**
- **ZDR Improvement: +8.33%** ⭐

**Current Run Results (with saved test set):**
- Base Model: Accuracy=80.83%, F1=79.65%, AUC-PR=87.56%, **ZDR=100.00%** ✅
- TTT Model: Accuracy=80.00%, F1=80.17%, AUC-PR=70.95%, **ZDR=100.00%**
- **ZDR Improvement: 0.00%** ❌

**Issue:** Even with saved test set, base model performance is different!

---

## Root Causes

### 1. **Test Set Was Created AFTER Optimization**
- The optimization was run **BEFORE** test set saving was implemented
- Trial 13's actual test set was **never saved**
- The current `test_set_best_trial.pkl` is from a **NEW run**, not from trial 13
- This new test set has different characteristics (easier for base model)

### 2. **Model Training Differences**
Even with the same test set, base model can differ due to:
- **Random seed variations** in federated learning
- **Different client data distributions** (Dirichlet sampling)
- **Model initialization** differences
- **Training randomness** (dropout, batch sampling, etc.)

### 3. **Missing Trial 13 Test Set**
The saved test set shows `trial_number: 'current_run'` not `trial_number: 13`, confirming it's not from the original optimization.

---

## Solutions

### **Option 1: Re-run Optimization (Recommended)**
Re-run the optimization with the save functionality enabled to capture trial 13's actual test set:

```bash
python optimize_hyperparameters.py --n_trials 50
```

This will:
- Save test sets for ALL trials (including trial 13)
- Ensure we have the EXACT test set used in trial 13
- Allow perfect reproducibility

**Pros:**
- ✅ Gets the exact test set from trial 13
- ✅ Ensures perfect reproducibility
- ✅ Can verify optimization results

**Cons:**
- ❌ Takes time (multiple hours)
- ❌ Uses computational resources

---

### **Option 2: Recreate Trial 13 Conditions Exactly**
Manually recreate trial 13's exact conditions:

1. **Use exact hyperparameters** from `best_hyperparameters.json` ✅ (Already done)
2. **Fix ALL random seeds** to ensure identical:
   - Model initialization
   - Client data distribution (Dirichlet sampling)
   - Federated learning training
   - Test set creation

**Challenge:** Even with fixed seeds, differences in:
- Client data distribution (non-deterministic Dirichlet sampling)
- Model initialization across devices
- CUDA randomness

---

### **Option 3: Accept Current Results**
Acknowledge that:
- The optimized hyperparameters work **very well** (100% ZDR)
- The test set is different, so direct comparison is difficult
- Focus on demonstrating that TTT can improve when base model has room for improvement

**This requires:**
- Finding or creating a harder test set where base model doesn't achieve 100% ZDR
- Using a different zero-day attack type that's more challenging

---

## Recommended Action Plan

### **Immediate: Verify What We Have**

1. **Check if trial 13 test set exists** from previous optimization run
   ```bash
   ls saved_test_sets/test_set_trial_13.pkl
   ```

2. **If it doesn't exist**, we have two paths:

   **Path A: Re-run Optimization (Best)**
   - Run optimization again with save functionality
   - This will capture trial 13's actual test set
   - Then reproduce results with exact test set

   **Path B: Create Challenging Test Set (Alternative)**
   - Modify test set creation to be more challenging
   - Ensure base model has room for improvement (not 100% ZDR)
   - This allows TTT to demonstrate improvement

---

## Why Base Model Performance Differs

Even with the same test set, base model can differ because:

### **1. Federated Learning Randomness**
- Client selection (if used)
- Client data distribution (Dirichlet alpha affects this)
- Model aggregation order
- Local training randomness

### **2. Model Initialization**
- Weight initialization differs between runs
- Even with fixed seed, CUDA can introduce variations

### **3. Training Dynamics**
- Dropout layers add randomness
- Batch sampling order
- Gradient accumulation rounding errors

### **4. Test Set Creation**
- Sequence creation randomness
- Post-sequence filtering randomness
- Even with fixed seeds, numerical precision can differ

---

## Conclusion

**The core issue:** We don't have trial 13's actual test set because:
1. Optimization ran BEFORE save functionality was added
2. Current saved test set is from a NEW run, not trial 13
3. Even with same test set, base model can differ due to training randomness

**To truly reproduce trial 13:**
- Need to re-run optimization to capture trial 13's test set
- OR fix ALL sources of randomness (seeds, initialization, training)
- OR accept that perfect reproduction is difficult and focus on relative improvements










