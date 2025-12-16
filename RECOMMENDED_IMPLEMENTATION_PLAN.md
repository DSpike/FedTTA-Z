# Recommended Implementation Plan: Phased Approach

## Phase 1: Fix 1 (Threshold Alignment) - **START HERE** ✅

**Why first:** This addresses the root cause (threshold mismatch)

**Implementation:**
- Use PR-optimized threshold during TTT adaptation (same as evaluation)
- Replace class-specific thresholds with single PR-optimized threshold

**Expected impact:** +3-5% accuracy, +5-10% ZDR

**Time to implement:** ~30 minutes
**Time to test:** 1 run (~5 minutes)

---

## Phase 2: Fix 3 (Individual Model Evaluation) - **Add diagnostics** 🔍

**Why second:** Helps diagnose if ensemble is canceling improvements

**Implementation:**
- Evaluate each TTT variant (pseudo-label, contrastive, self-supervised) separately
- Report individual model metrics
- Compare ensemble vs best individual model

**Expected impact:** 
- Diagnostic value (identifies best variant)
- Might reveal that individual models improve more than ensemble

**Time to implement:** ~20 minutes
**Time to test:** Same run as Phase 1 (no extra time)

---

## Phase 3: Fix 4 (Metric-Optimized TTT) - **If Phase 1 isn't enough** 📈

**When to apply:** Only if Fix 1 gives <2% improvement

**Implementation:**
- Add F1/ZDR terms to TTT loss (requires validation labels)
- Balance unsupervised losses with supervised metric optimization

**Expected impact:** Additional +2-4% if Phase 1 alone isn't sufficient

**Time to implement:** ~45 minutes
**Time to test:** 1 run (~5 minutes)

---

## Phase 4: Fix 5 (Best Individual Model) - **Optional alternative** 🔄

**When to apply:** If ensemble consistently underperforms individual models

**Implementation:**
- Select best individual model instead of ensemble
- Evaluate all variants on validation set
- Use best performer for final evaluation

**Expected impact:** +1-3% if ensemble is indeed canceling improvements

**Time to implement:** ~30 minutes
**Time to test:** Same run as Phase 2

---

## What NOT to apply (yet):

### Fix 2 (Soft Prediction Tracking) - **Skip for now**
- **Reason:** Diagnostic only, doesn't fix performance
- **Apply later:** Only if you need detailed analysis of why TTT isn't working

---

## Recommended Sequence:

```
Step 1: Apply Fix 1 → Run system → Check results
         ↓
         If improvement < 2%:
         ↓
Step 2: Apply Fix 3 → Run system → Check which variant improves most
         ↓
         If ensemble underperforms individual models:
         ↓
Step 3: Apply Fix 5 → Run system → Use best individual model
         ↓
         If still not enough:
         ↓
Step 4: Apply Fix 4 → Run system → Optimize directly for metrics
```

---

## Decision Matrix:

| Current Performance Gap | Recommended Fixes |
|------------------------|-------------------|
| **Large (>5% difference)** | Fix 1 + Fix 3 + Fix 4 |
| **Medium (2-5% difference)** | Fix 1 + Fix 3 |
| **Small (<2% difference)** | Fix 1 only |

---

## My Recommendation:

**Start with Phase 1 only (Fix 1)**

**Reasons:**
1. ✅ Addresses the root cause (threshold mismatch)
2. ✅ Highest expected impact (+3-5% accuracy)
3. ✅ Simplest to implement
4. ✅ Quickest to test

**Then:**
- If Fix 1 gives >3% improvement → **Stop here** (problem solved!)
- If Fix 1 gives 1-3% improvement → **Add Fix 3** (diagnose ensemble issue)
- If Fix 1 gives <1% improvement → **Add Fix 3 + Fix 4** (try both)

---

## Implementation Order (Final Recommendation):

### **Option A: Conservative (Recommended)** ⭐
1. Fix 1 → Test → **If <2% improvement, then Fix 3 → Test → If needed, Fix 4**

### **Option B: Aggressive** 
1. Fix 1 + Fix 3 → Test → **If <2% improvement, then Fix 4**

### **Option C: Comprehensive** 
1. Fix 1 + Fix 3 + Fix 5 → Test → **If <2% improvement, then Fix 4**

---

## Summary:

**Don't apply all fixes at once.** Start with **Fix 1** and measure results. Only add more if needed.

**Quick answer:** Apply **Fix 1 first**, then **Fix 3** (for diagnostics), then **Fix 4** only if Fix 1 isn't enough.



