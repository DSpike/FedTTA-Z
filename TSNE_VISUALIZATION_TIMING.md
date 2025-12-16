# t-SNE Visualization Timing: When Does It Run?

## 📍 **Answer: During Testing (After Training is Complete)**

The t-SNE visualization runs **AFTER meta-learning training** and **DURING the testing/evaluation phase**.

---

## 🔄 **Execution Timeline**

### **Phase 1: Federated Meta-Learning Training** (Rounds 1-15)
```
1. Data Preprocessing
2. Initialize Federated Learning
3. For each round (1 to 15):
   - Clients train locally (meta-learning)
   - Coordinator aggregates models (FedProx)
   - Validation evaluation (monitoring)
4. Training Complete ✅
```

**Model State**: Model is being trained, embeddings are being learned

---

### **Phase 2: Testing/Evaluation** (After Training)
```
1. Base Model Evaluation
   - Uses trained model (self.coordinator.model)
   - Evaluates on TEST set (X_test, y_test)
   - Computes base model metrics
   
2. Embedding Quality Diagnostic ⭐ (HERE!)
   - Uses SAME trained model (self.coordinator.model)
   - Extracts embeddings from TEST data
   - Computes prototypes from VALIDATION data
   - Creates t-SNE visualizations
   
3. TTT Adaptation
   - Adapts model to test distribution
   - Evaluates adapted model
```

**Model State**: Model is fully trained, embeddings are fixed (from meta-learning)

---

## 📊 **What the t-SNE Shows**

### **The Visualization Represents:**

1. **Embeddings learned during meta-learning training**
   - Model was trained through federated meta-learning (15 rounds)
   - Embeddings were learned from training data
   - Model parameters are fixed at this point

2. **Embeddings extracted from TEST data**
   - Test samples are passed through the trained model
   - Model extracts embeddings using learned feature extractors
   - These embeddings show how well the model generalizes to test data

3. **Prototypes computed from VALIDATION data**
   - Support set: 200 random samples from validation set
   - Prototypes = average embeddings of Normal vs Attack from validation
   - Used for prototype-based classification

---

## 🎯 **Key Points**

### **1. Model is Fully Trained**
- ✅ All federated learning rounds completed
- ✅ Model parameters are fixed (no more training)
- ✅ Embeddings reflect what was learned during meta-learning

### **2. Evaluation on Test Data**
- ✅ Uses TEST set (X_test, y_test) - unseen during training
- ✅ Shows generalization capability
- ✅ Reveals if embeddings learned during training work on test data

### **3. Not During Training**
- ❌ Does NOT run during meta-learning training
- ❌ Does NOT show embeddings during training
- ❌ Does NOT use training data for visualization

---

## 📝 **Code Location**

**File**: `main.py`  
**Location**: Line 2698-2725  
**Function**: `generate_performance_visualizations()`  
**Called**: After base model evaluation, before TTT adaptation

```python
# After federated training completes
# After base model evaluation
base_results_no_zeroday = self.evaluate_base_model_only(exclude_zero_day=True)

# THEN embedding quality diagnostic runs
embedding_results = check_embedding_quality(
    self.coordinator.model,  # ← Fully trained model
    X_test, y_test,          # ← Test data (unseen during training)
    X_val, y_val,            # ← Validation data (for prototypes)
    output_dir="embedding_quality_diagnostics"
)
```

---

## 🔍 **What This Means**

### **The t-SNE Visualization Shows:**

1. **Quality of embeddings learned during meta-learning**
   - How well did meta-learning learn discriminative features?
   - Do embeddings separate Normal from Attack?

2. **Generalization to test data**
   - Do embeddings work on unseen test samples?
   - Is there distribution shift between training and test?

3. **Prototype-based classification quality**
   - Are prototypes (from validation) representative?
   - Do test embeddings cluster around correct prototypes?

---

## ⚠️ **Important Distinction**

### **What It's NOT:**

- ❌ **NOT during training**: Doesn't show embeddings as they're being learned
- ❌ **NOT on training data**: Uses test data, not training data
- ❌ **NOT real-time**: Runs once after training completes

### **What It IS:**

- ✅ **After training**: Shows final learned embeddings
- ✅ **On test data**: Shows generalization capability
- ✅ **Diagnostic tool**: Helps understand why base model performs poorly

---

## 🎯 **Why This Timing Matters**

### **For Understanding Base Model Performance:**

The diagnostic runs **after training but before TTT**, which means:

1. **Shows what meta-learning learned**
   - Embeddings reflect 15 rounds of federated meta-learning
   - Reveals if meta-learning is working correctly

2. **Shows generalization issues**
   - If embeddings are poor on test data, meta-learning didn't generalize well
   - Explains why base model has 42.80% accuracy

3. **Baseline for TTT**
   - Shows embeddings before TTT adaptation
   - Can compare with embeddings after TTT (if we add that diagnostic)

---

## 📊 **Summary**

| Aspect | Details |
|--------|---------|
| **When** | After federated meta-learning training completes |
| **Phase** | Testing/Evaluation phase |
| **Model State** | Fully trained, parameters fixed |
| **Data Used** | Test set (X_test) for embeddings, Validation set (X_val) for prototypes |
| **Purpose** | Diagnose embedding quality and explain base model performance |
| **Timing** | After base model evaluation, before TTT adaptation |

---

## ✅ **Conclusion**

The t-SNE visualization runs **DURING TESTING** (after training is complete), not during meta-learning training. It shows:

- ✅ Embeddings learned during meta-learning (as evaluated on test data)
- ✅ How well the trained model generalizes to test distribution
- ✅ Why base model performance is poor (embeddings not discriminative enough)

This is the correct timing for diagnosing why the base model has poor performance! 🎯









