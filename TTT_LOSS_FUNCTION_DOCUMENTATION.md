# TTT Loss Function Documentation

## 🎯 **TTT Loss Function Components**

The Test-Time Training (TTT) adaptation uses a **composite loss function** with multiple components:

---

## 📊 **Primary Loss Function**

### **Location**: `coordinators/centralized_coordinator.py` (lines 305-340)

### **Formula**:

```python
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss + l2_reg_weight * l2_reg
```

---

## 🔍 **Component Breakdown**

### **1. Entropy Loss (Unsupervised)** ⭐ Primary Component

**Purpose**: Minimize prediction uncertainty (encourage confident predictions)

**Formula**:

```python
probs = F.softmax(logits, dim=1)
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()
```

**Mathematical Form**:

```
H(p) = -Σᵢ pᵢ * log(pᵢ)
entropy_loss = mean(H(p))
```

**Config Parameter**:

- `entropy_weight: float = 0.5740446517340904` (from Optuna optimization)

**Behavior**:

- ✅ Encourages model to make confident predictions
- ✅ Reduces prediction uncertainty
- ⚠️ Can lead to overconfidence if not balanced

---

### **2. Pseudo-Label Loss (Semi-Supervised)** ⭐ Secondary Component

**Purpose**: Use high-confidence predictions as supervision signal

**Formula**:

```python
confidences, pseudo_labels = probs.max(dim=1)
confident_mask = confidences > pseudo_threshold

if confident_mask.sum() > 0:
    pseudo_loss = F.cross_entropy(
        logits[confident_mask],
        pseudo_labels[confident_mask],
        reduction='mean'
    )
else:
    pseudo_loss = 0.0
```

**Mathematical Form**:

```
pseudo_loss = CrossEntropy(logits[confident], pseudo_labels[confident])
```

**Config Parameters**:

- `use_pseudo_labels: bool = True`
- `pseudo_weight: float = 3.0425406933718913` (from Optuna optimization)
- `pseudo_threshold: float = 0.95` (high confidence threshold)
- `pseudo_min_threshold: float = 0.7173803589287694` (minimum threshold)

**Behavior**:

- ✅ Provides supervised signal from confident predictions
- ✅ Helps model learn from test data distribution
- ⚠️ Can propagate errors if pseudo-labels are wrong

---

### **3. L2 Regularization (Prevent Overfitting)** ⭐ Regularization Component

**Purpose**: Penalize large parameter changes (prevent excessive adaptation)

**Formula**:

```python
l2_reg = 0.0
if hasattr(config, 'ttt_l2_reg_weight') and config.ttt_l2_reg_weight > 0:
    for param in adapted_model.parameters():
        if param.requires_grad:
            l2_reg += torch.sum((param - param_initial)**2)
    l2_reg = config.ttt_l2_reg_weight * l2_reg
```

**Mathematical Form**:

```
L2_reg = λ * Σᵢ ||θᵢ - θᵢ₀||²
```

**Config Parameter**:

- `ttt_l2_reg_weight: float = 0.0010257563974185654` (from Optuna optimization)

**Behavior**:

- ✅ Prevents excessive parameter drift
- ✅ Improves generalization (+2-4% improvement observed)
- ✅ Keeps model close to original parameters

---

## 📐 **Complete Loss Formula**

### **Full Expression**:

```
Total Loss = α * Entropy_Loss + β * Pseudo_Label_Loss + λ * L2_Regularization

Where:
- α = entropy_weight = 0.574
- β = pseudo_weight = 3.043
- λ = ttt_l2_reg_weight = 0.001
```

### **Code Implementation**:

```python
# Entropy loss (unsupervised)
entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
entropy_loss = entropy.mean()

# Pseudo-label loss (semi-supervised)
pseudo_loss = F.cross_entropy(logits[confident_mask], pseudo_labels[confident_mask])

# L2 regularization (prevent overfitting)
l2_reg = ttt_l2_reg_weight * sum((param - param_initial)**2 for param in model.parameters())

# Total loss
total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss + l2_reg
```

---

## ⚙️ **Configuration Parameters**

### **From `config.py`**:

```python
# === TENT + PSEUDO-LABELS CONFIGURATION ===
use_pseudo_labels: bool = True
pseudo_threshold: float = 0.95
pseudo_min_threshold: float = 0.7173803589287694
pseudo_weight: float = 3.0425406933718913
entropy_weight: float = 0.5740446517340904

# === TEST-TIME TRAINING (TTT) CONFIGURATION ===
ttt_l2_reg_weight: float = 0.0010257563974185654
```

---

## 🎯 **Loss Function Characteristics**

### **1. Unsupervised Component (Entropy)**

- **Type**: Information-theoretic
- **Goal**: Minimize prediction uncertainty
- **Weight**: ~0.57 (moderate)
- **Impact**: Encourages confident predictions

### **2. Semi-Supervised Component (Pseudo-Labels)**

- **Type**: Cross-entropy loss
- **Goal**: Learn from high-confidence predictions
- **Weight**: ~3.04 (high - dominant component)
- **Impact**: Provides supervised signal from test data

### **3. Regularization Component (L2)**

- **Type**: Parameter regularization
- **Goal**: Prevent excessive adaptation
- **Weight**: ~0.001 (low but important)
- **Impact**: Keeps model stable, improves generalization

---

## 📊 **Loss Weight Analysis**

| Component             | Weight | Relative Importance | Purpose                    |
| --------------------- | ------ | ------------------- | -------------------------- |
| **Entropy Loss**      | 0.574  | Medium              | Confidence maximization    |
| **Pseudo-Label Loss** | 3.043  | **High (dominant)** | Supervised learning signal |
| **L2 Regularization** | 0.001  | Low (but critical)  | Stability & generalization |

**Observation**: Pseudo-label loss has the highest weight, making it the **dominant component** in the loss function.

---

## 🔄 **Adaptive Behavior**

### **Pseudo-Label Threshold Adaptation**:

The system uses **adaptive thresholding** for pseudo-labels:

```python
# Threshold decreases over time (curriculum learning)
current_threshold = max(
    pseudo_min_threshold,  # Minimum: 0.717
    pseudo_threshold - (step / ttt_steps) * (pseudo_threshold - pseudo_min_threshold)
)
```

**Effect**:

- Early steps: Only very confident predictions (threshold = 0.95)
- Later steps: More predictions included (threshold → 0.717)
- **Result**: Curriculum learning approach

---

## ✅ **Summary**

**TTT Loss Function = Entropy Loss + Pseudo-Label Loss + L2 Regularization**

1. **Entropy Loss** (0.574 weight): Minimizes uncertainty
2. **Pseudo-Label Loss** (3.043 weight): **Dominant component** - learns from confident predictions
3. **L2 Regularization** (0.001 weight): Prevents overfitting

**Key Insight**: The loss function is **semi-supervised**, combining:

- **Unsupervised** entropy minimization (TENT-style)
- **Supervised** pseudo-label learning (semi-supervised learning)
- **Regularization** to prevent overfitting

This hybrid approach allows the model to adapt to test data distribution while maintaining stability.
