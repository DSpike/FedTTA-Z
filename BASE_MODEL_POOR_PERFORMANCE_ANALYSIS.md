# Base Model Poor Performance Analysis

## 📊 **Current Performance Summary**

### **Base Model (Before TTT):**
- **Accuracy**: 42.80% ⚠️ **VERY POOR** (worse than random for binary classification)
- **F1-Score**: 26.53% ⚠️ **VERY POOR**
- **Precision**: 52.05% (moderate, but misleading due to low recall)
- **Recall**: 17.80% ⚠️ **CRITICAL ISSUE** - Only catching 18% of attacks!
- **ZDR**: 20.65% (expected to be low, but base model should catch known attacks)

### **TTT Model (After Adaptation):**
- **Accuracy**: 72.55% ✅ (good improvement)
- **F1-Score**: 78.78% ✅ (excellent)
- **Recall**: 87.82% ✅ (excellent - catching most attacks)
- **ZDR**: 88.59% ✅ (excellent zero-day detection)

---

## 🚨 **Critical Finding: Base Model is Fundamentally Broken**

### **The Problem:**

The base model has **42.80% accuracy** and **17.80% recall**, which means:

1. **It's performing WORSE than random guessing** (50% for binary classification)
2. **It's missing 82% of all attacks** (including known attacks it was trained on!)
3. **It cannot serve as a reliable baseline** before TTT adaptation

### **Expected vs. Actual:**

**What SHOULD happen:**
- Base model trained on Normal + 8 known attack types
- Base model should perform reasonably on Normal + Known Attacks (60-80% accuracy)
- Base model should have moderate recall (50-70%) on known attacks
- TTT should then boost zero-day detection specifically

**What IS happening:**
- Base model performs poorly on ALL classes (42.80% accuracy)
- Base model misses most attacks, even known ones (17.80% recall)
- TTT fixes everything (including known attacks it shouldn't need to fix)

---

## 🔍 **Root Cause Analysis**

### **1. Prototype-Based Evaluation Issue** ⚠️ **MOST LIKELY**

**The Problem:**
The base model uses **prototype-based prediction** during evaluation, which requires:
- A support set (validation data) to compute prototypes
- The support set may not represent all known attack types well
- Prototypes computed from validation data may not match test distribution

**From Code** (`main.py` lines 2992-2993):
```python
# Create support set from validation data for prototype computation
support_x = X_val_tensor[support_indices]  # Random 200 samples from validation
support_y = y_val_binary[support_indices]

# Compute prototypes and get prototype-based logits
prototypes, unique_labels = global_model.compute_prototypes(support_x, support_y)
base_logits = global_model.forward_with_prototypes(X_test_filtered, prototypes)
```

**Issues:**
- **Random sampling**: Only 200 random samples from validation set
- **Class imbalance**: Random sampling may not include all attack types equally
- **Distribution mismatch**: Validation set distribution ≠ Test set distribution
- **Prototype quality**: Prototypes may not represent learned embeddings well

### **2. Embedding Quality Issue**

**The Problem:**
If the meta-learned embeddings are poor quality, prototypes will be ineffective:
- TCN feature extractors may not have learned good representations
- Embeddings may not be discriminative enough
- Prototype computation may produce poor class representations

### **3. Training-First Evaluation Issue**

**The Problem:**
The base model is evaluated immediately after federated training, but:
- Meta-learning may require task-specific adaptation (few-shot learning)
- The model expects support sets during inference (meta-learning paradigm)
- Direct evaluation without proper support set setup may fail

### **4. Threshold Issue**

**The Problem:**
Base model uses a **fixed 0.5 threshold** for binary classification:
- Prototype distances converted to probabilities via softmax
- Fixed threshold may not be optimal for distance-based predictions
- No threshold tuning for base model (unlike TTT which optimizes threshold)

**From Code**:
```python
optimal_threshold: fixed_threshold  # Base model uses fixed 0.5 threshold
```

### **5. Distribution Shift Between Training and Test**

**The Problem:**
- Test set may have different distribution than training data
- Meta-learning model may not generalize well to test distribution
- Validation set used for prototypes may not match test set

---

## 🎯 **Why TTT "Fixes" Everything (Including Known Attacks)**

**Critical Insight:**
TTT improves performance on **both known and zero-day attacks** because:

1. **TTT adapts BatchNorm statistics** to test distribution
2. **This helps ALL samples**, not just zero-day
3. **The base model embeddings are actually reasonable**, but statistics are misaligned
4. **TTT corrects the statistical mismatch** that breaks base model performance

**This suggests:**
- The base model's **learned features are good** (TCN, embeddings)
- But the **normalization statistics** are wrong for test data
- TTT fixes this by adapting BN to test distribution

---

## ✅ **Recommended Solutions**

### **Solution 1: Improve Base Model Support Set Selection** ⭐ **HIGH PRIORITY**

**Current**: Random 200 samples from validation set

**Fix**: Use stratified sampling to ensure all classes are represented:
```python
# Use stratified sampling instead of random
from sklearn.model_selection import train_test_split
support_indices = stratified_sample(y_val_tensor, n_samples=200)
```

**Or**: Use more samples (500-1000) to get better prototype estimates

### **Solution 2: Use Task-Specific Support Sets**

**Fix**: For each test sample, use similar samples from validation set as support:
- Use k-NN to find similar validation samples
- Compute prototypes from similar samples only
- This matches the meta-learning training paradigm better

### **Solution 3: Optimize Base Model Threshold**

**Fix**: Tune threshold for base model too:
```python
# Optimize threshold on validation set
best_threshold = optimize_threshold_on_validation(base_model, X_val, y_val)
base_predictions = (base_probabilities[:, 1] > best_threshold).long()
```

### **Solution 4: Evaluate Base Model with Direct Embeddings**

**Fix**: Instead of prototype-based evaluation, use direct model output:
```python
# Direct evaluation (if model has classifier head)
base_logits = global_model(X_test_filtered)  # Direct forward pass
base_predictions = torch.argmax(base_logits, dim=1)
```

**But**: Your model is prototype-based (no classifier), so this requires adding a classifier head

### **Solution 5: Check Meta-Training Quality**

**Fix**: Investigate if meta-training is working correctly:
- Check training loss curves (should decrease)
- Check if embeddings are discriminative (t-SNE visualization)
- Verify prototypes during training match class labels

### **Solution 6: Match Evaluation to Training Paradigm**

**Fix**: During evaluation, simulate the meta-learning setup:
- Create proper support sets (with all classes represented)
- Use the same few-shot setup as during training
- Evaluate in a truly few-shot manner (not with full test set)

---

## 🔬 **Diagnostic Steps**

### **Step 1: Check Confusion Matrix**
```python
# Analyze what base model is actually predicting
cm = base_results['confusion_matrix']
print(f"True Negatives (Normal→Normal): {cm[0][0]}")
print(f"False Positives (Normal→Attack): {cm[0][1]}")
print(f"False Negatives (Attack→Normal): {cm[1][0]}")  # This is likely very high
print(f"True Positives (Attack→Attack): {cm[1][1]}")
```

**Expected Finding**: High FN rate (attacks predicted as normal)

### **Step 2: Check Support Set Quality**
```python
# Verify support set has all classes
unique_support_labels = torch.unique(support_y)
print(f"Support set classes: {unique_support_labels}")
print(f"Expected: [0, 1] (Normal, Attack)")
```

### **Step 3: Check Prototype Quality**
```python
# Check if prototypes are well-separated
prototype_distances = torch.cdist(prototypes, prototypes)
print(f"Prototype distances: {prototype_distances}")
# Should show clear separation between Normal and Attack prototypes
```

### **Step 4: Compare Embeddings**
```python
# Visualize embeddings (t-SNE or PCA)
# Check if Normal and Attack embeddings are separable
from sklearn.manifold import TSNE
embeddings_np = query_embeddings.cpu().numpy()
labels_np = y_test_binary.cpu().numpy()
# Should show clear clusters for Normal vs Attack
```

---

## 📝 **Summary of Findings**

### **Main Issue:**
The base model performs **worse than random** (42.80% accuracy) and misses **82% of attacks** (17.80% recall), even known ones.

### **Root Cause:**
**Prototype-based evaluation** with poor support set quality:
- Random 200 samples may not represent all classes
- Prototypes computed from validation may not match test distribution
- No threshold optimization for base model

### **Why TTT Works:**
TTT adapts BatchNorm statistics to test distribution, fixing the statistical mismatch that breaks base model performance.

### **Impact:**
- Base model cannot serve as a reliable baseline
- Makes it unclear if improvements come from TTT or just fixing broken base model
- For publication, base model should perform reasonably (60-80% accuracy) before TTT

### **Priority Actions:**
1. ⭐ **Fix support set selection** (stratified sampling, more samples)
2. ⭐ **Add threshold optimization** for base model
3. **Investigate meta-training quality** (check embeddings, prototypes)
4. **Match evaluation to training paradigm** (proper few-shot setup)

---

## ✅ **Conclusion**

**The base model's poor performance is a critical issue that needs to be addressed before publication.** While TTT results are excellent (88.59% ZDR), the base model should demonstrate reasonable performance on known attacks first. The prototype-based evaluation setup is likely the main culprit, and fixing it should improve base model performance significantly.










