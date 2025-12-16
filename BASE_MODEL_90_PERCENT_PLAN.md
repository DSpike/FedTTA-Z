# 🎯 Base Model 90%+ Performance Plan

## 📊 **Current Status vs Target**

### **Current Base Model Performance**:

- **Accuracy**: 57.20%
- **F1-Score**: 57.60%
- **ZDR**: 54.89%
- **Embedding Separability**: 0.105 (target: >0.3)

### **Target**: **90%+ on Key Metrics** (Accuracy, F1-Score, or ZDR)

**Gap to Close**: ~32-35 percentage points

---

## 🚀 **Aggressive Configuration for 90%+ Base Model**

Based on best-performing configurations and analysis, here's the plan:

---

## 🔴 **TIER 1: HIGH IMPACT CHANGES** (Start Here)

### **1. Dramatically Increase Embedding Quality** ⭐⭐⭐⭐⭐

**Current**: Embedding separability = 0.105 (only 34% of target)

#### **A. Increase Center Loss Weight Aggressively** (Expected: +10-15% F1)

```python
# Current in config.py:
center_loss_weight: float = 0.02

# Recommended for 90%+:
center_loss_weight: float = 0.08-0.10  # 4-5x increase

# Rationale:
# - Embedding separability is critical for base model performance
# - Current 0.02 is still too weak for 90%+ performance
# - Aggressive increase needed to reach target separability (>0.3)
# - Expected: 0.105 → 0.25-0.35 separability
```

#### **B. Increase Margin Loss Weight** (Expected: +5-8% F1)

```python
# Current in config.py:
margin_loss_weight: float = 0.12

# Recommended for 90%+:
margin_loss_weight: float = 0.20-0.25  # 67-108% increase

# Rationale:
# - Better inter-class separation
# - Prototypes further apart = better classification
# - Critical for achieving 90%+ performance
```

#### **C. Increase Prototype Margin** (Expected: +3-5% F1)

```python
# Current in config.py:
prototype_margin: float = 2.5

# Recommended for 90%+:
prototype_margin: float = 4.0-5.0  # 60-100% increase

# Rationale:
# - Larger margin between prototypes
# - Better class separation
# - Essential for high-performance base model
```

---

### **2. Increase Training Intensity** ⭐⭐⭐⭐⭐

#### **A. Dramatically Increase Meta-Training Epochs** (Expected: +8-12% F1)

```python
# Current in config.py:
meta_epochs: int = 20

# Recommended for 90%+:
meta_epochs: int = 40-50  # 2-2.5x increase

# Rationale:
# - More training = better feature learning
# - Embeddings need significant time to converge to high quality
# - 90%+ performance requires extensive training
# - Expected: Better learned representations
```

#### **B. Increase Learning Rate** (Expected: +3-5% F1)

```python
# Current in config.py:
learning_rate: float = 0.0015

# Recommended for 90%+:
learning_rate: float = 0.0025-0.003  # 67-100% increase

# Rationale:
# - Faster convergence to better solutions
# - More aggressive optimization needed
# - Expected: Better optimization trajectory
```

---

### **3. Increase Support Set Size** ⭐⭐⭐⭐

#### **A. Increase k-shot Dramatically** (Expected: +5-8% F1)

```python
# Current: ~130 samples (implicit)
# Recommended for 90%+:
k_shot: int = 200-300  # Much larger support sets

# Rationale:
# - More support samples = more representative prototypes
# - Better prototype-based classification
# - More stable and accurate predictions
# - Critical for 90%+ performance
```

#### **B. Increase Number of Meta-Tasks** (Expected: +2-4% F1)

```python
# Current: ~34 meta-tasks (implicit)
# Recommended for 90%+:
num_meta_tasks: int = 100-150  # 3-4x increase

# Rationale:
# - More diverse tasks = better generalization
# - Better meta-learning
# - Expected: Improved base model quality
```

---

### **4. Increase Model Capacity** ⭐⭐⭐⭐

#### **A. Increase Hidden Dimension** (Expected: +3-6% F1)

```python
# Current in config.py:
hidden_dim: int = 256

# Recommended for 90%+:
hidden_dim: int = 512-768  # 2-3x increase

# Rationale:
# - More model capacity = better feature learning
# - Can capture more complex patterns
# - Expected: Better representations
```

#### **B. Increase Embedding Dimension** (Expected: +2-4% F1)

```python
# Current in config.py:
embedding_dim: int = 128

# Recommended for 90%+:
embedding_dim: int = 256-384  # 2-3x increase

# Rationale:
# - More expressive embeddings
# - Better discriminative power
# - Expected: Improved classification
```

---

### **5. Optimize Transductive Refinement** ⭐⭐⭐

#### **A. Increase Transductive Refinement Iterations** (Expected: +2-3% F1)

```python
# Current in config.py:
transductive_refinement_iterations: int = 10

# Recommended for 90%+:
transductive_refinement_iterations: int = 20-30  # 2-3x increase

# Rationale:
# - More refinement = better prototypes
# - Better use of unlabeled query data
# - Expected: Improved transductive learning
```

#### **B. Increase Transductive Steps** (Expected: +1-2% F1)

```python
# Current in config.py:
transductive_steps: int = 20

# Recommended for 90%+:
transductive_steps: int = 30-40  # 50-100% increase

# Rationale:
# - More steps = more refinement
# - Better prototype updates
# - Expected: Improved meta-learning
```

---

## 📊 **Recommended Configuration for 90%+ Base Model**

### **Complete Configuration Changes**:

```python
# === EMBEDDING QUALITY (Critical for 90%+) ===
center_loss_weight: float = 0.08  # 4x increase (0.02 → 0.08)
margin_loss_weight: float = 0.20  # 67% increase (0.12 → 0.20)
prototype_margin: float = 4.0  # 60% increase (2.5 → 4.0)

# === TRAINING INTENSITY ===
meta_epochs: int = 50  # 2.5x increase (20 → 50)
learning_rate: float = 0.0025  # 67% increase (0.0015 → 0.0025)

# === MODEL CAPACITY ===
hidden_dim: int = 512  # 2x increase (256 → 512)
embedding_dim: int = 256  # 2x increase (128 → 256)

# === SUPPORT SET SIZE ===
# Note: Adjust in meta-task creation
k_shot: int = 250  # Much larger (current ~130 → 250)

# === META-LEARNING ===
num_meta_tasks: int = 100  # 3x increase (current ~34 → 100)
transductive_refinement_iterations: int = 25  # 2.5x increase (10 → 25)
transductive_steps: int = 35  # 75% increase (20 → 35)

# === OPTIMIZATION ===
transductive_lr: float = 0.001  # Higher refinement LR (0.0007 → 0.001)
```

---

## 📈 **Expected Performance Impact**

### **Optimistic Scenario**:

- **Base Accuracy**: 57% → **90-92%** (+33-35pp)
- **Base F1**: 58% → **90-92%** (+32-34pp)
- **Base ZDR**: 55% → **88-90%** (+33-35pp)
- **Embedding Separability**: 0.105 → **0.30-0.35** (3x improvement)

### **Conservative Scenario**:

- **Base Accuracy**: 57% → **85-88%** (+28-31pp)
- **Base F1**: 58% → **85-88%** (+27-30pp)
- **Base ZDR**: 55% → **82-85%** (+27-30pp)
- **Embedding Separability**: 0.105 → **0.25-0.30** (2.4-2.9x improvement)

---

## ⚠️ **Important Considerations**

### **1. Training Time Will Increase**

- 50 epochs vs 20 epochs = **2.5x longer training**
- Larger model capacity = **slower per-epoch**
- More meta-tasks = **more computation**
- **Expected**: 3-4x longer training time

### **2. Memory Requirements**

- Larger model (512 hidden, 256 embedding) = **more GPU memory**
- Larger support sets (250 samples) = **more memory per task**
- **Ensure sufficient GPU memory available**

### **3. Risk of Overfitting**

- Aggressive increases may cause overfitting
- Monitor validation performance
- Consider early stopping if validation plateaus

---

## 🎯 **Implementation Strategy**

### **Phase 1: Aggressive Embedding Quality** (Week 1)

1. Increase `center_loss_weight` to 0.08
2. Increase `margin_loss_weight` to 0.20
3. Increase `prototype_margin` to 4.0
4. **Expected**: Embedding separability 0.105 → 0.25-0.30

### **Phase 2: Increase Training Intensity** (Week 1-2)

1. Increase `meta_epochs` to 50
2. Increase `learning_rate` to 0.0025
3. **Expected**: Better learned representations

### **Phase 3: Increase Model Capacity** (Week 2)

1. Increase `hidden_dim` to 512
2. Increase `embedding_dim` to 256
3. **Expected**: Better feature learning

### **Phase 4: Optimize Meta-Learning** (Week 2-3)

1. Increase `k_shot` to 250
2. Increase `num_meta_tasks` to 100
3. Increase refinement iterations
4. **Expected**: Better prototype-based learning

---

## 📋 **Quick Start Configuration**

For immediate implementation, update `config.py` with these values:

```python
# === CRITICAL FOR 90%+ BASE MODEL ===
center_loss_weight: float = 0.08  # 4x increase
margin_loss_weight: float = 0.20  # 67% increase
prototype_margin: float = 4.0  # 60% increase
meta_epochs: int = 50  # 2.5x increase
learning_rate: float = 0.0025  # 67% increase
hidden_dim: int = 512  # 2x increase
embedding_dim: int = 256  # 2x increase
transductive_refinement_iterations: int = 25  # 2.5x increase
transductive_steps: int = 35  # 75% increase
```

**And adjust in meta-task creation**:

- `k_shot`: 250 (instead of ~130)
- `num_meta_tasks`: 100 (instead of ~34)

---

## 🎯 **Target Metrics**

After implementing these changes, you should achieve:

- ✅ **Base Accuracy**: **90-92%**
- ✅ **Base F1-Score**: **90-92%**
- ✅ **Base ZDR**: **88-90%**
- ✅ **Embedding Separability**: **0.30-0.35**

---

## 💡 **Why This Will Work**

1. **Embedding Quality is Key**: Aggressive center/margin loss will dramatically improve separability
2. **More Training**: 50 epochs gives embeddings time to converge to high quality
3. **Larger Capacity**: Bigger model can learn more complex patterns
4. **Better Prototypes**: Larger support sets create more representative prototypes
5. **More Meta-Learning**: 100 tasks provide better generalization

This aggressive approach prioritizes base model performance over TTT adaptability.








