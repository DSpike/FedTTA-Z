# Center Loss and Prototype Margin Loss Implementation

## 🎯 **Objective**

Implement Center Loss and Prototype Margin Loss to improve embedding discriminativeness, addressing the poor embedding separability (silhouette score: 0.0481) that causes low base model performance (42.80% accuracy).

---

## 📋 **Implementation Plan**

### **1. Add CenterLoss Class**

**Location**: `models/transductive_fewshot_model.py` (after FocalLoss class)

**Purpose**: Reduce intra-class variance by pulling embeddings toward learnable class centers

**Key Features**:
- Learnable centers per class (initialized randomly)
- Computes mean squared distance from embeddings to their class centers
- Encourages compact, well-defined clusters

---

### **2. Add Prototype Margin Loss Function**

**Location**: `models/transductive_fewshot_model.py` (as static method in TransductiveLearner class)

**Purpose**: Enforce minimum margin between all pairs of prototypes

**Key Features**:
- Penalizes prototypes that are too close together
- Encourages better inter-class separation
- Configurable margin threshold

---

### **3. Integrate into Meta-Training**

**Location**: `models/transductive_fewshot_model.py` - `meta_train()` method

**Changes**:
- Initialize CenterLoss instance at start of meta_train
- Add center loss to optimizer parameters
- Compute center loss and margin loss in loss computation
- Add to total loss with configurable weights

---

### **4. Add Configuration Parameters**

**Location**: `config.py`

**New Parameters**:
- `use_center_loss: bool = True`
- `center_loss_weight: float = 0.01`
- `use_prototype_margin_loss: bool = True`
- `margin_loss_weight: float = 0.1`
- `prototype_margin: float = 2.0`

---

## ✅ **Expected Impact**

1. **Reduced Intra-Class Variance**
   - Embeddings will cluster more tightly around class centers
   - Smaller spread in t-SNE visualization
   - More compact, well-defined clusters

2. **Better Inter-Class Separation**
   - Prototypes will be forced apart (minimum margin)
   - Less overlap between Normal and Attack embeddings
   - Improved silhouette score (> 0.3 target)

3. **Improved Base Model Performance**
   - Better prototype-based classification accuracy
   - Reduced confusion between classes
   - Target: Base model accuracy 60-80% (from 42.80%)

---

## 📝 **Implementation Status**

- [ ] Add CenterLoss class
- [ ] Add prototype margin loss function
- [ ] Integrate into meta_train method
- [ ] Add configuration parameters
- [ ] Test and verify

---

## 🔍 **Code Integration Points**

### **In `meta_train()` method:**

1. **Initialize Center Loss**:
```python
# After optimizer initialization
device = next(self.parameters()).device
center_loss_fn = CenterLoss(
    num_classes=2,  # Binary: Normal, Attack
    embedding_dim=self.embedding_dim,
    device=device
)
# Add center parameters to optimizer
meta_optimizer = optim.AdamW(
    list(self.parameters()) + list(center_loss_fn.parameters()),
    lr=0.01, weight_decay=1e-4
)
```

2. **Compute Losses**:
```python
# In loss computation loop
center_loss = center_loss_fn(all_embeddings, all_labels)
margin_loss = self._compute_prototype_margin_loss(prototypes, margin=2.0)
total_loss = support_loss + query_loss + 0.01 * center_loss + 0.1 * margin_loss
```

---

## 🚀 **Next Steps**

1. Implement CenterLoss class
2. Add margin loss function
3. Integrate into meta_train
4. Add config parameters
5. Test and verify improvements









