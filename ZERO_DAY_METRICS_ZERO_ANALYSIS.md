# Zero-Day Metrics Showing Zero - Root Cause Analysis

## 🔍 **Problem Summary**

The evaluation plot shows **all zero metrics** (0.0) for zero-day detection:

- Accuracy: 0.0
- Precision: 0.0
- Recall: 0.0
- F1-Score: 0.0
- Zero-Day Detection Rate: 0.0

## 📊 **Evidence from Debug Logs**

```
🔍 DEBUG: Zero-day logits mean per class: [-97.18, -149.79]
🔍 DEBUG: Zero-day distances to prototypes: mean=[9.84, 12.23]
🔍 DEBUG: Which prototype is closer? [184, 0] (all 184 closer to prototype 0)
🔍 DEBUG: Zero-day mapped predictions (actual labels): [184, 0]
🔍 DEBUG: Zero-day actual labels: [0, 184]
🔍 DEBUG BASE MODEL - Zero-day confusion matrix: [[0, 0], [184, 0]]
```

## 🎯 **Root Cause**

### **1. Model Performance Issue (Not Code Bug)**

The model is **correctly identifying 184 zero-day samples**, but it's **predicting all of them as Normal (0)** instead of Attack (1).

**Confusion Matrix Interpretation:**

```
[[0, 0],     ← Row 0: Actual Normal samples
 [184, 0]]   ← Row 1: Actual Attack samples (zero-day)
     ↑   ↑
     |   └─ Column 1: Predicted Attack
     └───── Column 0: Predicted Normal

TN = 0, FP = 0, FN = 184, TP = 0
```

**Why Metrics Are Zero:**

- **Precision** = TP/(TP+FP) = 0/(0+0) = 0 (no attacks detected)
- **Recall** = TP/(TP+FN) = 0/(0+184) = 0 (missed all 184 attacks)
- **F1-Score** = 0 (harmonic mean of 0 and 0)
- **Accuracy** = (TP+TN)/(TP+TN+FP+FN) = (0+0)/(0+0+0+184) = 0
- **Zero-Day Detection Rate** = TP/(TP+FN) = 0/184 = 0

### **2. Why Model Predicts All Zero-Day as Normal**

**Prototype-Based Classification:**

- Model uses **prototype-based prediction** (Prototypical Networks style)
- Computes prototypes from support set (validation data)
- Predicts class based on **nearest prototype** (smallest distance)

**The Problem:**

- Zero-day samples have **mean distance 9.84** to Normal prototype
- Zero-day samples have **mean distance 12.23** to Attack prototype
- **All 184 samples are closer to Normal prototype** → All predicted as Normal

**Why This Happens:**

1. **Zero-day attacks are unseen during training** - model never learned their patterns
2. **Embedding space collapse** - zero-day samples may be embedded in a region closer to Normal
3. **Prototype quality** - Attack prototype may not be representative of zero-day attacks
4. **Support set bias** - Support set may have imbalanced Normal/Attack samples (164 Normal, 36 Attack)

## ✅ **What's Working Correctly**

1. ✅ **Zero-day sample identification**: Correctly identifies 184 zero-day samples (25% of test set)
2. ✅ **Evaluation code**: Correctly extracts zero-day samples and computes metrics
3. ✅ **Visualization code**: Correctly displays the metrics (they're zero because model performance is zero)
4. ✅ **Confusion matrix**: Correctly computed and interpreted

## 🔧 **Potential Solutions**

### **Option 1: Improve Model Training (Recommended)**

1. **Better embedding learning**: Train model to better separate Normal vs Attack in embedding space
2. **More diverse support sets**: Ensure support sets have balanced Normal/Attack samples
3. **Better prototype computation**: Use more representative attack samples for prototype computation
4. **Outlier exposure**: Already implemented (5% of other attacks relabeled as normal) - may need tuning

### **Option 2: Adjust Evaluation Strategy**

1. **Use probability-based thresholding**: Instead of nearest prototype, use probability threshold
2. **Calibrate probabilities**: Apply temperature scaling to improve probability estimates
3. **Use confidence scores**: Use distance-based confidence instead of hard argmax

### **Option 3: Improve Support Set Selection**

1. **Ensure balanced support set**: Force 50/50 Normal/Attack split in support set (already implemented)
2. **Use more attack samples**: Increase support set size for better Attack prototype
3. **Use diverse attack types**: Include multiple attack types in support set (if not zero-day)

## 📈 **Expected Behavior**

For a well-trained model on zero-day detection:

- **Zero-Day Detection Rate**: Should be > 50% (ideally > 80%)
- **Recall**: Should be > 50% (detecting at least half of zero-day attacks)
- **Precision**: May be lower (some false positives) but should be > 0
- **F1-Score**: Should be > 0.3 (balanced precision/recall)

## 🎯 **Conclusion**

The **plot showing zeros is CORRECT** - it accurately reflects that the model is **not detecting any zero-day attacks** (all predicted as Normal). This is a **model performance issue**, not a code bug.

**Next Steps:**

1. Investigate why zero-day samples are closer to Normal prototype
2. Check if support set has sufficient Attack samples
3. Verify model training is learning good embeddings
4. Consider adjusting hyperparameters or training strategy



