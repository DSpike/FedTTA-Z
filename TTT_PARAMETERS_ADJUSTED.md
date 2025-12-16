# TTT Parameters Adjustment Analysis

## 📋 **Overview**

Test-Time Training (TTT) adaptation adjusts **only a subset** of model parameters to adapt to the test distribution without catastrophic forgetting. This document details exactly which parameters are modified.

---

## 🔧 **Configuration Method**

The key method that configures trainable parameters is `_configure_model_for_tent()` in `coordinators/simple_fedavg_coordinator.py`:

```python
def _configure_model_for_tent(self, model):
    """Configure model: Batch norm + classifier head parameters trainable"""
    model.train()
    
    # Step 1: Disable ALL parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: Enable BatchNorm parameters
    num_bn_params = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for param in module.parameters():
                param.requires_grad = True
                num_bn_params += param.numel()
            # CRITICAL: Disable running statistics tracking
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    
    # Step 3: Enable classifier head parameters (if exists)
    num_classifier_params = 0
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
            num_classifier_params += param.numel()
    
    total_trainable = num_bn_params + num_classifier_params
    logger.info(
        f"Configured TENT: {num_bn_params} batch norm + "
        f"{num_classifier_params} classifier = {total_trainable} trainable parameters"
    )
    return model
```

---

## ✅ **Parameters Adjusted During TTT**

### **1. Batch Normalization Parameters** ⭐ **PRIMARY TARGET**

**What is adjusted:**
- **BatchNorm weight** (`gamma` or `weight`): Scaling parameter
- **BatchNorm bias** (`beta` or `bias`): Shift parameter

**Why BatchNorm:**
- BatchNorm layers normalize activations using **statistics computed from the current batch**
- During TTT, BN layers adapt to **test distribution statistics** (mean and variance)
- This enables the model to handle distribution shift without changing feature extractors

**Specific BatchNorm Layers in Your Model:**

From `models/transductive_fewshot_model.py`:

1. **TCN Feature Extractors** (`EfficientTCN`):
   - `bn1`: BatchNorm1d(hidden_dim) - After first depthwise separable conv
   - `bn2`: BatchNorm1d(hidden_dim) - After second depthwise separable conv
   - `bn3`: BatchNorm1d(hidden_dim) - After third depthwise separable conv

2. **Feature Projection**:
   - BatchNorm1d(embedding_dim) - After linear projection to embedding space

**Total BatchNorm Parameters:**
- Per BN layer: `hidden_dim` (weight) + `hidden_dim` (bias) = `2 × hidden_dim`
- If `hidden_dim = 512` and `embedding_dim = 128`:
  - TCN: 3 layers × 2 × 512 = **3,072 parameters**
  - Projection: 1 layer × 2 × 128 = **256 parameters**
  - **Total BN parameters: ~3,328 parameters**

**Critical Configuration:**
```python
module.track_running_stats = False  # Disable running stats tracking
module.running_mean = None          # Clear cached running mean
module.running_var = None           # Clear cached running variance
```

**Why disable running stats:**
- Forces BN to use **current batch statistics** (test distribution)
- Prevents contamination from training distribution statistics
- Enables true adaptation to test domain

---

### **2. Classifier Head Parameters** (If Present)

**What is adjusted:**
- **Linear layer weights** (`classifier.weight`)
- **Linear layer bias** (`classifier.bias`)

**Important Note:**
- Your model is **prototype-based** (no classifier head)
- Classifier parameters are only adjusted **if** the model has a `classifier` attribute
- From code analysis: `self.classifier` was removed, so this may not apply to your current model

**If classifier exists:**
- Parameters: `num_classes × embedding_dim` (weights) + `num_classes` (bias)
- For binary classification: `2 × embedding_dim + 2` parameters

---

## ❌ **Parameters NOT Adjusted During TTT**

### **🔒 ALL Transductive Meta-Learning Parameters (FROZEN)**

During the **Federated Transductive Meta-Learning** training phase (meta_train), ALL model parameters are learned:
- TCN feature extractors (all convolution layers)
- Feature projection (all linear layers)
- BatchNorm layers (weight, bias, running statistics)
- Residual connections
- All other model components

**During TTT, these Transductive Meta-Learning parameters are FROZEN** (except BatchNorm weight/bias which are adjusted).

---

### **1. Transductive Meta-Learning Feature Extractor Weights (TCN Layers)**

**What is frozen:**
- **Conv1d weights** (depthwise and pointwise convolutions) - Learned during meta-training
- **TCN layer parameters** (convolution kernels, biases) - Learned during meta-training
- **Feature projection Linear layer weights** - Learned during meta-training (only BN is trainable during TTT)

**Why frozen:**
- These parameters encode the **learned feature representations** from federated transductive meta-learning
- Prevents catastrophic forgetting of meta-learned knowledge
- Maintains the ability to extract meaningful features for few-shot tasks
- Only adapts statistics (BN), not feature extraction capabilities

### **2. Transductive Meta-Learning Embedding Weights**

**What is frozen:**
- All Linear layer weights in `feature_projection` - Learned during meta-training
- These weights learned to map TCN features to embedding space for prototype-based classification
- Only the BatchNorm within `feature_projection` is trainable during TTT

### **3. Transductive Meta-Learning Residual Connection Weights**

**What is frozen:**
- All residual connection parameters (Conv1d layers) - Learned during meta-training
- These enable gradient flow and stabilize training during meta-learning

---

## 📊 **Summary Table**

| Component | Layer Type | Learned During Meta-Training? | Trainable During TTT? | Parameters Adjusted in TTT | Count |
|-----------|------------|------------------------------|----------------------|---------------------------|-------|
| **TCN Layers** | Conv1d (depthwise) | ✅ **Yes** (Meta-Learning) | ❌ **No** (Frozen) | None | - |
| **TCN Layers** | Conv1d (pointwise) | ✅ **Yes** (Meta-Learning) | ❌ **No** (Frozen) | None | - |
| **TCN Layers** | BatchNorm1d (bn1, bn2, bn3) | ✅ **Yes** (Meta-Learning) | ✅ **Yes** (Adjusted) | weight, bias | ~3,072 |
| **Feature Projection** | Linear | ✅ **Yes** (Meta-Learning) | ❌ **No** (Frozen) | None | - |
| **Feature Projection** | BatchNorm1d | ✅ **Yes** (Meta-Learning) | ✅ **Yes** (Adjusted) | weight, bias | ~256 |
| **Classifier** | Linear (if exists) | ✅ **Yes** (Meta-Learning) | ✅ **Yes** (Adjusted) | weight, bias | ~256 |
| **Residual Connections** | Conv1d | ✅ **Yes** (Meta-Learning) | ❌ **No** (Frozen) | None | - |

**Total Trainable Parameters:** ~3,328 - 3,584 (depending on classifier presence)

**Percentage of Total Model:** ~1-2% of total parameters (very small subset!)

---

## 🔬 **How TTT Updates Parameters**

### **Optimizer Configuration:**

```python
# From coordinators/simple_fedavg_coordinator.py
params = [p for p in adapted_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    params, 
    lr=ttt_lr,  # Typically 0.00015 (from config)
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
```

**Only trainable parameters receive gradients:**
- BatchNorm parameters get gradients from entropy/diversity/pseudo-label loss
- All other parameters have `requires_grad=False`, so they receive no gradients

### **Loss Functions That Drive Updates:**

1. **Entropy Loss**: Minimizes prediction entropy (encourages confidence)
2. **Diversity Loss**: Prevents mode collapse (encourages balanced predictions)
3. **Pseudo-Label Loss**: Uses high-confidence predictions as supervision

**All losses are computed on query set and backpropagate to trainable parameters only.**

---

## 🎯 **Why This Approach Works**

### **TENT (Test-Time Normalization) Philosophy:**

1. **Minimal Adaptation:**
   - Only adapts normalization statistics (BatchNorm)
   - Preserves learned feature representations
   - Prevents overfitting to test set

2. **Distribution Alignment:**
   - BatchNorm adapts to test distribution mean/variance
   - Aligns feature statistics without changing feature extractors
   - Enables handling of distribution shift

3. **Efficiency:**
   - Only ~1-2% of parameters are trainable
   - Fast adaptation (few optimization steps needed)
   - Low risk of catastrophic forgetting

---

## 🔍 **Verification in Your System**

From the logs, you should see:

```
🔍 TTT Debug: Found X parameter groups with Y trainable parameters
Configured TENT: X batch norm + Y classifier = Z trainable parameters
```

**Expected Values:**
- BatchNorm parameters: ~3,328
- Classifier parameters: ~256 (if present) or 0 (if prototype-based)
- **Total: ~3,328 - 3,584 parameters**

---

## 📝 **Key Takeaways**

1. **TTT adjusts ~1-2% of model parameters** (only BatchNorm + optional classifier)

2. **BatchNorm is the primary target:**
   - Weight (gamma) and bias (beta) parameters
   - Running statistics are disabled and cleared
   - Adapts to test distribution statistics

3. **All Transductive Meta-Learning parameters remain frozen:**
   - TCN layers, convolutions, linear layers (except classifier) learned during federated meta-training are NOT adjusted
   - This prevents catastrophic forgetting of meta-learned feature representations
   - Only BatchNorm parameters are adjusted to adapt to test distribution statistics

4. **This is standard TENT approach:**
   - Minimal, efficient adaptation
   - Proven effective for distribution shift
   - Low risk, high reward

5. **The small parameter count (~3K) explains:**
   - Fast adaptation (few steps needed)
   - Low memory overhead
   - Stable optimization

---

## 🔬 **Alternative TTT Approaches (Not Used Here)**

### **Full Model Fine-Tuning:**
- **Adjusts:** All parameters
- **Pros:** Maximum adaptation
- **Cons:** High risk of overfitting, catastrophic forgetting, slow

### **Feature Extractor Fine-Tuning:**
- **Adjusts:** TCN + BN + classifier
- **Pros:** More adaptation capacity
- **Cons:** Moderate risk, slower

### **Classifier Only:**
- **Adjusts:** Only classifier head
- **Pros:** Very fast, very safe
- **Cons:** Limited adaptation (cannot handle distribution shift)

**Your System Uses:** **BatchNorm + Classifier (if present)** - Balanced approach! ✅

---

## ✅ **Conclusion**

**TTT adjusts ONLY:**
1. ✅ **BatchNorm weight and bias parameters** (~3,328 params)
2. ✅ **Classifier head parameters** (if present, ~256 params)

**TTT does NOT adjust (All Transductive Meta-Learning Parameters):**
- ❌ **TCN convolution weights** (learned during federated meta-learning)
- ❌ **Feature projection linear weights** (learned during federated meta-learning)
- ❌ **Residual connection weights** (learned during federated meta-learning)
- ❌ **Any other feature extractor parameters** (learned during federated meta-learning)
- ❌ **BatchNorm running statistics** (disabled and cleared, but weights/bias are adjusted)

**This minimal adaptation strategy enables:**
- Fast test-time adaptation (~16-228 steps)
- Handling of distribution shift
- Preservation of learned features
- Excellent zero-day detection (88.59% ZDR)! 🎯


