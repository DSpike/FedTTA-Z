# Hyperparameter Tuning Results Analysis

## 📊 Performance Comparison: Before vs After Tuning

### **Previous Results (Before Hyperparameter Optimization):**
- **Base Model:**
  - Accuracy: 73.80%
  - F1-Score: 72.56%
  - AUC-PR: 75.12%
  - ZDR: 36.08%
  - FAR: 11.56%

- **TTT Model:**
  - Accuracy: 77.41%
  - F1-Score: 81.66%
  - AUC-PR: 91.72%
  - ZDR: 39.18%
  - FAR: 38.78%

---

### **Current Results (After Hyperparameter Optimization):**
- **Base Model:**
  - Accuracy: **79.22%** (+5.42% ⬆️)
  - F1-Score: **82.71%** (+10.15% ⬆️)
  - AUC-PR: 74.79% (-0.33% ⬇️)
  - ZDR: **57.73%** (+21.65% ⬆️) ⭐ **MASSIVE IMPROVEMENT**
  - FAR: ~20.41%

- **TTT Model:**
  - Accuracy: **79.52%** (+2.11% ⬆️)
  - F1-Score: 81.22% (-0.44% ⬇️)
  - AUC-PR: **91.67%** (maintained, excellent)
  - ZDR: **44.33%** (+5.15% ⬆️)
  - FAR: **20.41%** (-18.37% ⬇️) ⭐ **SIGNIFICANT FAR REDUCTION**

---

## ✅ **Key Improvements:**

### **1. Base Model ZDR: HUGE SUCCESS** ⭐⭐⭐
- **ZDR increased from 36.08% → 57.73%** (+21.65 percentage points)
- This is a **60% relative improvement** in zero-day detection!
- The base model is now much better at detecting zero-day attacks
- **This suggests the federated learning improvements (12 rounds, better architecture) helped significantly**

### **2. TTT Model FAR Reduction: EXCELLENT** ⭐⭐
- **FAR reduced from 38.78% → 20.41%** (-18.37 percentage points)
- This is a **47% reduction** in false alarms!
- Much better precision/recall balance
- **The hyperparameter tuning (especially threshold optimization) worked well**

### **3. Overall Accuracy: IMPROVED** ⭐
- Base: 73.80% → 79.22% (+5.42%)
- TTT: 77.41% → 79.52% (+2.11%)
- Both models improved, with base model showing larger gains

### **4. AUC-PR: MAINTAINED EXCELLENCE** ⭐⭐
- TTT AUC-PR: 91.67% (maintained at excellent level)
- Still exceeds SOTA benchmark of 90%

---

## ⚠️ **Areas of Concern:**

### **1. TTT ZDR Still Below Base Model**
- **Base Model ZDR: 57.73%**
- **TTT Model ZDR: 44.33%**
- **Gap: -13.40%** (TTT is worse than base for ZDR)

**Possible Causes:**
1. TTT adaptation may be overfitting to the adaptation set
2. The ZDR-optimized threshold (0.0500) may be too aggressive, causing misclassification
3. Prototype-based inference may not be working optimally for zero-day samples
4. The adaptation process may be reducing sensitivity to zero-day patterns

**This is a critical issue that needs investigation.**

### **2. TTT F1-Score Slight Decrease**
- Previous: 81.66%
- Current: 81.22%
- Change: -0.44% (minimal, but worth noting)

---

## 📈 **Statistical Significance (K-Fold CV):**

### **Base Model:**
- Accuracy: 78.89% ± 4.39%
- F1-Score: ~78.67% ± 2.93%

### **TTT Model:**
- Accuracy: 78.90% ± 3.28%
- F1-Score: 78.69% ± 3.41%

**Observation:** The standard deviations are similar, and the means are nearly identical. This suggests:
- TTT is not providing significant improvement over base in k-fold CV
- The improvements seen in single-run may not be statistically robust
- Need more folds or larger dataset for stronger evidence

---

## 🎯 **Overall Assessment:**

### **✅ What Worked Well:**
1. **Base Model ZDR:** Massive improvement (+21.65%) - the federated learning improvements were highly effective
2. **FAR Reduction:** TTT FAR reduced by 47% - excellent for practical deployment
3. **AUC-PR:** Maintained at excellent level (91.67%)
4. **Overall Accuracy:** Both models improved

### **⚠️ What Needs Attention:**
1. **TTT ZDR Regression:** TTT ZDR (44.33%) is now lower than base ZDR (57.73%)
   - This is the opposite of what we want
   - TTT should improve, not degrade, zero-day detection
   - **Priority: HIGH** - This needs immediate investigation

2. **Statistical Robustness:** K-fold CV shows minimal difference between base and TTT
   - Need to ensure improvements are consistent across folds
   - May need more folds or larger dataset

---

## 💡 **Recommendations:**

### **Immediate Actions:**
1. **Investigate TTT ZDR Regression:**
   - Check if prototype-based inference is working correctly for zero-day samples
   - Verify that ZDR-optimized threshold (0.0500) is being applied correctly
   - Consider adjusting prototype alignment weights or contrastive loss weights
   - May need to increase `ttt_zero_day_ratio` even more (currently 0.65)

2. **Focus on Zero-Day Specific Adaptation:**
   - Increase `ttt_prototype_weight` further (currently 0.6, try 0.8-1.0)
   - Increase `ttt_contrastive_weight` further (currently 0.8, try 1.0-1.2)
   - Lower `ttt_zero_day_candidate_threshold` (currently 0.65, try 0.60)

3. **Verify Threshold Application:**
   - Ensure ZDR-optimized threshold is being used in final evaluation
   - Check if prototype-based inference is interfering with threshold optimization

### **Next Steps:**
1. Run diagnostic to understand why TTT ZDR is lower than base
2. Adjust hyperparameters specifically for zero-day detection
3. Consider reverting to standard inference (not prototype-based) if it's causing issues
4. Re-run with focused zero-day improvements

---

## 📊 **Summary:**

**Overall Assessment: MIXED RESULTS**

✅ **Successes:**
- Base model ZDR improved dramatically (+21.65%)
- TTT FAR reduced significantly (-47%)
- Overall accuracy improved for both models
- AUC-PR maintained at excellent level

⚠️ **Concerns:**
- TTT ZDR is now worse than base model ZDR
- TTT improvements not statistically significant in k-fold CV
- Need to investigate why TTT is degrading zero-day detection

**Verdict:** The hyperparameter tuning helped the **base model significantly**, but the **TTT model needs further optimization** to improve zero-day detection beyond the base model.


