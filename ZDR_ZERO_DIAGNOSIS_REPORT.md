# Zero-Day Detection Rate (ZDR) Zero Diagnosis Report

## 🔍 **Problem Statement**

ZDR is **zero (0.0)** for both base and TTT models, while other metrics (accuracy, F1, precision, recall) show normal values.

---

## ✅ **Diagnostic Results**

### **1. Zero-Day Samples ARE Found** ✅

- **Total test samples**: 11,508
- **Zero-day samples found**: 857 (7.4%)
- **Zero-day attack**: 'DDoS' (label 2)
- **Zero-day mask**: Working correctly

### **2. Zero-Day Labels Are Correct** ✅

- **Binary labels for zero-day samples**: All 857 have label 1 (Attack)
- **Multiclass labels**: All 857 have label 2 (DDoS)
- **No labeling issues detected**

### **3. Root Cause Identified** ❌

**The model is predicting ALL zero-day samples as Normal (0) instead of Attack (1)**

This means:
- **TP (True Positives)**: 0 (no zero-day attacks detected)
- **FN (False Negatives)**: 857 (all zero-day attacks missed)
- **ZDR = TP / (TP + FN) = 0 / (0 + 857) = 0.0**

---

## 📊 **Why Other Metrics Work But ZDR Doesn't**

### **Overall Metrics (Accuracy, F1, etc.)**

These metrics include:
- **Normal samples**: 4,487 (39%)
- **Known attacks**: 6,164 (54%)
- **Zero-day attacks**: 857 (7%)

The model performs well on Normal and Known attacks, so overall metrics look good.

### **ZDR (Zero-Day Detection Rate)**

ZDR only considers **zero-day samples** (857 samples):
- If the model predicts all 857 as Normal → ZDR = 0.0
- This doesn't affect overall metrics much because zero-day is only 7% of test set

---

## 🎯 **Why This Happens**

### **Possible Reasons:**

1. **Model Overfitting to Known Attacks**
   - Model learned to distinguish Normal vs. Known attacks
   - Cannot generalize to novel (zero-day) attack patterns
   - Zero-day attacks look "normal" to the model

2. **Insufficient Training Diversity**
   - Model only saw specific attack types during training
   - Zero-day attack (DDoS) has different patterns
   - Model lacks exposure to diverse attack patterns

3. **Feature Representation Issue**
   - Features learned for known attacks don't capture zero-day characteristics
   - Zero-day attacks may have different feature distributions
   - Model's feature extractor doesn't generalize well

4. **Threshold Too Conservative**
   - Classification threshold optimized for overall F1
   - May be too conservative for zero-day detection
   - Zero-day samples have lower confidence scores

---

## 🔧 **Solutions**

### **1. Check Model Predictions**

Run this to see actual predictions:

```python
# After model evaluation, check:
zero_day_predictions = final_predictions[zero_day_mask]
print(f"Zero-day predictions: {np.bincount(zero_day_predictions)}")
# If all are 0 → model predicts all as Normal
```

### **2. Check Prediction Probabilities**

```python
# Check if zero-day samples have low attack probabilities
zero_day_probs = attack_probs[zero_day_mask]
print(f"Zero-day attack probabilities: mean={zero_day_probs.mean():.4f}, std={zero_day_probs.std():.4f}")
# If mean < 0.5 → model is not confident they're attacks
```

### **3. Optimize Threshold for Zero-Day**

Current threshold may be optimized for overall F1. Try:
- Lower threshold to increase zero-day recall
- Use `threshold_optimization_strategy = 'zdr_optimized'` in config
- Or manually set a lower threshold

### **4. Improve Model Training**

- Add more diverse attack samples to training
- Use data augmentation for attack patterns
- Add regularization to prevent overfitting
- Use contrastive learning to learn better feature representations

### **5. Check Feature Importance**

- Verify that features important for zero-day detection are being used
- Check if zero-day attacks have different feature distributions
- Consider feature engineering specific to zero-day detection

---

## 📋 **Next Steps**

1. **Run the diagnostic script** to verify zero-day samples are found:
   ```bash
   python diagnose_zero_zdr_issue.py
   ```

2. **Check prediction logs** during evaluation:
   - Look for "🔍 Base Model ZDR Calculation" logs
   - Check "Zero-day TP" and "Zero-day FN" values
   - Verify "Zero-day predictions distribution"

3. **Check if threshold is too high**:
   - Review threshold optimization logs
   - Try lowering threshold manually
   - Use ZDR-optimized threshold strategy

4. **Investigate model predictions**:
   - Check prediction probabilities for zero-day samples
   - Compare feature distributions between known and zero-day attacks
   - Visualize embeddings to see if zero-day samples cluster separately

---

## 🎯 **Summary**

**ZDR is zero because:**
- ✅ Zero-day samples ARE found (857 samples)
- ✅ Zero-day mask is correct
- ❌ **Model predicts ALL zero-day samples as Normal (0)**
- ❌ **No zero-day attacks are detected (TP=0)**

**This is a model prediction issue, not a data/mask issue.**

The model needs to be improved to detect zero-day attacks, or the threshold needs to be adjusted to be more sensitive to zero-day patterns.





