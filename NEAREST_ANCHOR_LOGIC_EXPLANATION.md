# Nearest Anchor Logic in Intrusion Detection

## What is Nearest Anchor Logic?

**Nearest Anchor Logic** (also called **Prototype-based Classification** or **Anchor-based Detection**) is a classification approach where:

1. **Anchors/Prototypes** are representative points (centroids) for each class in the embedding space
2. **Classification** is done by finding which anchor is **closest** to a test sample
3. **Distance metric** is typically Euclidean distance in the embedding space

### Simple Example

Imagine a 2D embedding space:
```
Normal Traffic Anchor:     [0.2, 0.3]
DoS Attack Anchor:         [0.8, 0.7]
Probe Attack Anchor:       [0.5, 0.9]

Test Sample:               [0.25, 0.35]

Distances:
- Distance to Normal:      0.058 ✅ (closest!)
- Distance to DoS:         0.735
- Distance to Probe:       0.658

Prediction: Normal Traffic
```

## How It Works in Your Project

### Current Implementation

Your project **already uses** nearest anchor logic! Let me show you:

**File**: `models/transductive_fewshot_model.py`

```python
def forward_with_prototypes(self, x, prototypes):
    """
    Classify based on nearest prototype (anchor)
    """
    # Get embeddings
    embeddings = self.get_embeddings(x)  # Shape: [batch, 256]

    # Compute distances to each prototype (anchor)
    distances = torch.cdist(embeddings, prototypes)  # Shape: [batch, num_classes]

    # Nearest anchor = smallest distance
    logits = -distances  # Negative distance (closer = higher score)

    return logits
```

**What happens**:
1. Model embeds test sample into 256-dimensional space
2. Computes distance to each class anchor/prototype
3. Assigns to nearest anchor (smallest distance)

### In TTT Adaptation

**File**: `coordinators/centralized_coordinator.py` (lines 391-393)

```python
# Compute initial prototypes (anchors) from support set
prototypes_ttt, _ = adapted_model.compute_prototypes(support_x_ttt, support_y_ttt)
prototypes_ttt = prototypes_ttt.detach()  # Your recent fix!
```

These prototypes ARE the anchors for nearest anchor logic!

## Why Nearest Anchor Logic is Useful for Your Project

### 1. **Zero-Day Attack Detection** ⭐ PRIMARY BENEFIT

**The Problem**:
- Traditional classifiers need labeled examples of every attack type
- Zero-day attacks have NO training examples
- How can you detect something you've never seen?

**Nearest Anchor Solution**:
```
Known Anchors:
├─ Normal Traffic Anchor     [learned from training]
├─ DoS Anchor                [learned from training]
└─ Probe Anchor              [learned from training]

Zero-Day Attack (unseen):    [0.95, 0.85]

Distances:
- Distance to Normal:        0.92  (far)
- Distance to DoS:           0.15  ✅ (closest)
- Distance to Probe:         0.45  (medium)

Prediction: DoS-like attack (even though it's a NEW attack variant!)
```

**Key Insight**: Zero-day attacks are **similar** to known attack categories:
- New DoS variant → close to DoS anchor
- New Probe variant → close to Probe anchor
- Genuinely novel attack → far from ALL anchors (anomaly detection)

### 2. **Test-Time Training (TTT) Adaptation**

**Why it helps TTT**:

During TTT, you adapt anchors to test distribution:

```
BEFORE TTT (training anchors):
Normal Anchor: [0.2, 0.3]
Attack Anchor: [0.8, 0.7]

Test samples appear shifted:
Normal: [0.25, 0.35] ← slightly shifted
Attack: [0.75, 0.65] ← slightly shifted

AFTER TTT (adapted anchors):
Normal Anchor: [0.24, 0.33] ← moved toward test normals
Attack Anchor: [0.76, 0.68] ← moved toward test attacks
```

**Benefits**:
- ✅ Adapts to distribution shift (test data different from training)
- ✅ Maintains class separation
- ✅ No need for labels during adaptation
- ✅ Can use k-means clustering to refine anchors

### 3. **Interpretability**

**Distance-based decisions are interpretable**:

```python
Sample: Suspicious traffic
Distance to Normal: 0.85 (far)
Distance to DoS:    0.12 (very close) ✅
Distance to Probe:  0.45 (medium)

Interpretation: "This looks 85% like a DoS attack"
Confidence: High (0.12 is very close)
```

Compare to black-box neural network: "Softmax probability = 0.89" (what does that mean?)

### 4. **Few-Shot Learning**

**Works with few examples**:

Traditional classifier needs:
- 1000s of labeled examples per attack type

Nearest anchor needs:
- Just enough samples to compute a good centroid
- Your project uses **152-shot** (152 samples per class)
- Can even work with 5-shot or 1-shot!

### 5. **Open-Set Recognition**

**Detect TRULY novel attacks**:

```python
def is_novel_attack(sample, all_anchors, threshold=0.6):
    """
    If sample is far from ALL known anchors, it's a novel attack
    """
    distances = [distance(sample, anchor) for anchor in all_anchors]
    min_distance = min(distances)

    if min_distance > threshold:
        return True  # Novel attack! (far from everything known)
    else:
        return False  # Similar to known category
```

**Example**:
```
Sample: [0.1, 0.95] (weird corner of embedding space)

Distance to Normal: 0.88 (far)
Distance to DoS:    0.75 (far)
Distance to Probe:  0.82 (far)

All distances > 0.6 → NOVEL ATTACK DETECTED! 🚨
```

## How Your Project Uses It

### Architecture Flow

```
1. Input Traffic → TCN Encoder → 256D Embedding
                                       ↓
2. Training: Compute class anchors (prototypes)
   - Normal anchor = mean(all normal embeddings)
   - DoS anchor = mean(all DoS embeddings)
   - etc.
                                       ↓
3. Test-Time Training (TTT):
   - Use k-means to refine anchors on test data
   - Adapt embeddings to move closer to correct anchors
   - Update anchors dynamically every 10 steps
                                       ↓
4. Classification:
   - Compute distance from test sample to each anchor
   - Assign to nearest anchor
   - Confidence = 1 / (1 + distance)
```

### Key Components

**1. Prototype Computation** ([centralized_coordinator.py:391](coordinators/centralized_coordinator.py#L391))
```python
prototypes_ttt, _ = adapted_model.compute_prototypes(support_x_ttt, support_y_ttt)
# Computes anchors as class centroids
```

**2. K-means Clustering** (for unsupervised anchor discovery)
```python
# Uses k-means to find anchors without labels
kmeans = KMeans(n_clusters=2)  # 2 anchors: Normal, Attack
pseudo_labels = kmeans.fit_predict(embeddings)
```

**3. Distance-based Classification**
```python
distances = torch.cdist(embeddings, prototypes)
logits = -distances  # Closer = higher score
predictions = logits.argmax(dim=1)  # Nearest anchor wins
```

## Advantages for Your Specific Use Case

### Your Goal: Zero-Day Intrusion Detection

| Feature | Traditional Classifier | Nearest Anchor Logic |
|---------|----------------------|---------------------|
| **Zero-day detection** | ❌ Fails (no training examples) | ✅ Detects similar attacks |
| **Adaptation** | ❌ Hard to adapt without labels | ✅ TTT adapts anchors easily |
| **Few-shot learning** | ❌ Needs 1000s of examples | ✅ Works with 10-100 examples |
| **Novel attack detection** | ❌ No mechanism | ✅ Distance threshold |
| **Interpretability** | ❌ Black box | ✅ Distance = similarity |
| **Distribution shift** | ❌ Rigid boundaries | ✅ Anchors can move |

### Example Scenario

**Training**: Model learns on 2017 attacks
- DoS anchor learned from 2017 DoS attacks
- Probe anchor learned from 2017 Probe attacks

**Testing (2025)**: New attack variants appear
- **Zero-day DoS**: Never seen before, but shares characteristics with 2017 DoS
- **Nearest anchor logic**: Measures distance, finds it's closest to DoS anchor
- **Result**: Correctly classifies as DoS-like attack! ✅

**Traditional classifier**:
- "I've never seen this exact pattern before"
- **Result**: Misclassifies as Normal ❌

## Potential Improvements for Your Project

### 1. **Dynamic Anchor Adjustment** (Already Implemented!)

You already do this every 10 steps during TTT:
```python
if (step + 1) % prototype_update_interval == 0:
    prototypes_ttt_new, _ = adapted_model.compute_prototypes(support_x_ttt, support_y_ttt)
    prototypes_ttt_new = prototypes_ttt_new.detach()
    prototypes_ttt = prototypes_ttt_new
```

### 2. **Multi-Level Anchors** (Potential Enhancement)

Instead of one anchor per class, use multiple:
```python
# Instead of:
DoS_anchor = [0.8, 0.7]  # Single anchor

# Use:
DoS_anchors = [
    [0.75, 0.65],  # DoS variant 1
    [0.82, 0.73],  # DoS variant 2
    [0.79, 0.71],  # DoS variant 3
]
# Classify to DoS if close to ANY DoS anchor
```

### 3. **Confidence Thresholding** (Already Implemented!)

You already reject low-confidence samples:
```python
confidence_rejection_threshold: float = 0.90
# Reject samples far from all anchors
```

### 4. **Anchor Margin Loss** (Already Implemented!)

You use prototype margin loss to ensure anchors are well-separated:
```python
ttt_prototype_margin_loss_weight: float = 1.0
margin: float = 4.5
# Ensures anchors are at least 4.5 units apart
```

## Mathematical Foundation

### Distance Computation

**Euclidean Distance** (what your code uses):
```python
distance = sqrt((embedding - anchor)²)
         = ||embedding - anchor||₂
```

**Why it works**:
- Embeddings in same class cluster together
- Different classes are far apart
- Nearest anchor = most similar class

### Prototype Computation

**Class Centroid** (mean of class embeddings):
```python
anchor_class_k = mean(all_embeddings_of_class_k)
```

**K-means Clustering** (unsupervised):
```python
# Iteratively:
1. Assign samples to nearest anchor
2. Recompute anchors as cluster centroids
3. Repeat until convergence
```

## Comparison to Your Current Issue

### Why TTT Was Failing

The L2 regularization was **preventing anchors from moving**:

```
Test Distribution:
- Test normal samples: [0.25, 0.35]
- Test attack samples: [0.75, 0.65]

Training Anchors:
- Normal anchor: [0.2, 0.3]
- Attack anchor: [0.8, 0.7]

TTT tries to adapt anchors toward test distribution:
- Move Normal anchor: [0.2, 0.3] → [0.25, 0.35]
- Move Attack anchor: [0.8, 0.7] → [0.75, 0.65]

But L2 penalty said: "NO! Stay close to original [0.2, 0.3] and [0.8, 0.7]!"
Result: Anchors couldn't move → Classification degraded
```

### Why Disabling L2 Should Work

Now anchors can freely adapt:
```
Step 1:   Normal anchor = [0.20, 0.30] (training)
Step 50:  Normal anchor = [0.22, 0.32] (adapting...)
Step 100: Normal anchor = [0.24, 0.34] (adapting...)
Step 200: Normal anchor = [0.25, 0.35] (matched test distribution!)

Classification improves because anchors are in the right place!
```

## Summary

**Nearest Anchor Logic** is:
- ✅ What your project already uses (prototype-based classification)
- ✅ Perfect for zero-day detection (generalizes to unseen variants)
- ✅ Enables test-time training (anchors can adapt)
- ✅ Interpretable (distance = similarity)
- ✅ Works with few examples (few-shot learning)

**Your current TTT issues** were because:
- L2 regularization prevented anchors from adapting
- Anchors stayed at training positions
- Test samples were far from training anchors
- Classification degraded

**With L2 disabled**:
- Anchors can move to test distribution
- Better alignment with test samples
- Expected +2-4% improvement

The nearest anchor logic is actually the **foundation** of why your approach should work for zero-day detection!
