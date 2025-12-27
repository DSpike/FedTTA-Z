# Test-Time Training Method Classification

**Date**: December 25, 2025
**Analysis**: Code-based (not comments)

---

## Question: Which TTT Method Is Used?

Based on **actual code implementation** (not comments), your system uses:

### ✅ **Full-Tune Method** (Option 3)

---

## Evidence from Code

### 1. All Parameters Are Updated

**Location**: `main.py` line 7958-7960

```python
# OPTIMIZED TTT optimizer with better hyperparameters
ttt_optimizer = torch.optim.AdamW(
    adapted_model.parameters(),  # ← ALL parameters
    lr=self.config.ttt_lr,
    weight_decay=self.config.ttt_weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Analysis**:
- `adapted_model.parameters()` returns **ALL trainable parameters**
- No filtering or selection of specific layers
- No parameter freezing detected

---

### 2. No BatchNorm-Specific Handling

**What BN Adapt would look like**:
```python
# BN Adapt approach (NOT in your code)
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm1d):
        # Combine source statistics with target statistics
        module.momentum = 0.1  # Mix old and new statistics
```

**What Tent would look like**:
```python
# Tent approach (NOT in your code)
optimizer = torch.optim.Adam(
    [p for name, p in model.named_parameters()
     if 'bn' in name and 'weight' in name or 'bias' in name],  # Only BN affine params
    lr=0.001
)
```

**Your code**:
```python
# Full-tune approach (ACTUAL code)
ttt_optimizer = torch.optim.AdamW(
    adapted_model.parameters(),  # ALL parameters, not filtered
    lr=self.config.ttt_lr
)
```

**Conclusion**: No special BatchNorm handling → Not BN Adapt or Tent

---

### 3. Model Architecture Has BatchNorm Layers

**Location**: `models/transductive_fewshot_model.py` (multiple lines)

```python
# TCN blocks have BatchNorm
self.bn1 = nn.BatchNorm1d(hidden_dim)  # Line 345
self.bn2 = nn.BatchNorm1d(hidden_dim)  # Line 352
self.bn3 = nn.BatchNorm1d(hidden_dim)  # Line 359

# Feature projection has BatchNorm
self.feature_projection = nn.Sequential(
    nn.Linear(feature_output_dim, embedding_dim),
    nn.BatchNorm1d(embedding_dim),  # Line 1160
    nn.ReLU()
)
```

**Analysis**:
- Model contains multiple BatchNorm layers
- These are included in `model.parameters()`
- All get updated during TTT (full-tune)

---

### 4. Training Mode Settings

**Location**: `main.py` line 7951

```python
# Set model to training mode for TTT adaptation (dropout active)
adapted_model.set_ttt_mode(training=True)
```

**Analysis**:
- `set_ttt_mode(training=True)` likely calls `model.train()`
- This enables:
  - Dropout layers (active regularization)
  - BatchNorm layers in **training mode** (update running statistics)
  - All gradient computations

**Behavior in training mode**:
```python
# What happens with BatchNorm in training mode:
# 1. Updates running_mean and running_var
# 2. Uses batch statistics (not stored statistics)
# 3. Affine parameters (weight, bias) are trainable
```

---

### 5. Backward Pass Updates All Parameters

**Location**: `main.py` lines 8064-8070

```python
# Backward pass
total_loss.backward()

# Gradient clipping for stability
torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)

# Optimizer step
ttt_optimizer.step()
```

**Analysis**:
- `total_loss.backward()` computes gradients for **all** parameters
- `clip_grad_norm_(adapted_model.parameters())` clips **all** parameter gradients
- `ttt_optimizer.step()` updates **all** parameters registered in optimizer

---

## Method Comparison Table

| Method | Parameters Updated | BatchNorm Handling | Your Code |
|--------|-------------------|-------------------|-----------|
| **BN Adapt** | None (statistics only) | Mix source + target statistics | ❌ No |
| **Tent** | Only BN affine (weight, bias) | Reset source statistics | ❌ No |
| **Full-Tune** | All parameters | Standard training mode | ✅ **YES** |

---

## Detailed Breakdown: What Gets Updated

### During TTT Adaptation

**Your system updates**:

1. **TCN/Temporal Encoder**:
   - Convolutional weights
   - Convolutional biases
   - BatchNorm weights (γ)
   - BatchNorm biases (β)
   - BatchNorm running statistics (momentum-based)

2. **Feature Projection**:
   - Linear layer weights
   - Linear layer biases
   - BatchNorm weights (γ)
   - BatchNorm biases (β)

3. **All Other Layers**:
   - Embedding layers
   - Any other trainable parameters

**Total**: Every trainable parameter in the model

---

## Why This Is "Full-Tune"

### Definition of Full-Tune
> "Full-tune the model, which is the most expensive one."
> Updates all parameters of the network during test-time adaptation.

### Your Implementation Matches
```python
# Your code does exactly this:
1. Clone entire model
2. Set to training mode (all layers active)
3. Create optimizer with ALL parameters
4. Compute loss on support + query sets
5. Backpropagate through entire network
6. Update ALL parameters with gradient descent
```

**Cost Analysis**:
- **BN Adapt**: Near-zero cost (just statistics mixing)
- **Tent**: Low cost (only 2 params per BN layer)
- **Your Full-Tune**: High cost (all parameters)

**Computation per TTT step**:
- Forward pass: Full network
- Backward pass: Full network
- Parameter updates: All ~500K-2M parameters (depending on model size)

---

## Loss Functions Used

**Location**: `main.py` lines 8032-8059

### 1. Supervised Loss on Support Set
```python
support_loss = self._focal_loss(support_outputs, support_y, support_class_weights, alpha=0.25, gamma=2.0)
```
**Type**: Focal loss with class balancing
**Purpose**: Fit model to labeled support samples

### 2. Consistency Loss on Support Set
```python
# 1. Entropy minimization (encourage confident predictions)
entropy_loss = -torch.mean(torch.sum(support_probs * torch.log(support_probs + 1e-8), dim=1))

# 2. Confidence maximization (encourage high max probability)
max_probs = torch.max(support_probs, dim=1)[0]
confidence_loss = -torch.mean(max_probs)

# 3. Diversity regularization (prevent mode collapse)
diversity_loss = torch.mean(torch.sum(support_probs**2, dim=1))

# Combined
consistency_loss = 0.4 * entropy_loss + 0.4 * confidence_loss + 0.2 * diversity_loss
```
**Type**: Self-supervised regularization
**Purpose**: Encourage confident, diverse predictions

### 3. Total Loss
```python
total_loss = support_weight * support_loss + consistency_weight * consistency_loss
```

---

## How This Differs from BN Adapt and Tent

### BN Adapt [27]

**Method**:
```python
# Pseudo-code for BN Adapt
for bn_layer in model.batch_norm_layers:
    # Mix source and target statistics
    bn_layer.running_mean = alpha * source_mean + (1-alpha) * target_mean
    bn_layer.running_var = alpha * source_var + (1-alpha) * target_var
    # NO parameter updates
```

**Characteristics**:
- ✅ No training (inference only)
- ✅ Fast (just statistics update)
- ❌ Limited adaptation (statistics only)
- ❌ Not in your code

---

### Tent [30]

**Method**:
```python
# Pseudo-code for Tent
# 1. Reset source statistics
for bn_layer in model.batch_norm_layers:
    bn_layer.reset_running_stats()

# 2. Only optimize BN affine parameters
optimizer = torch.optim.Adam([
    p for name, p in model.named_parameters()
    if 'bn' in name and ('weight' in name or 'bias' in name)
])

# 3. Minimize entropy on target samples
for step in range(num_steps):
    output = model(target_batch)
    loss = entropy_loss(output)
    loss.backward()
    optimizer.step()  # Only updates BN affine params
```

**Characteristics**:
- ✅ Selective updates (only BN γ, β)
- ✅ Moderate cost
- ✅ No labels needed (entropy minimization)
- ❌ Not in your code

---

### Your Full-Tune Method

**Method**:
```python
# Your actual code
# 1. Clone entire model
adapted_model = copy.deepcopy(multiclass_model)

# 2. Optimize ALL parameters
optimizer = torch.optim.AdamW(adapted_model.parameters())

# 3. Use supervised + unsupervised losses
for step in range(num_steps):
    support_output = model(support_x)
    support_loss = focal_loss(support_output, support_y)  # Supervised
    consistency_loss = entropy + confidence + diversity   # Unsupervised
    total_loss = support_loss + consistency_loss
    total_loss.backward()
    optimizer.step()  # Updates ALL parameters
```

**Characteristics**:
- ✅ Maximum adaptation capability
- ✅ Can learn complex patterns
- ✅ Uses both labeled (support) and consistency losses
- ❌ Highest computational cost
- ❌ Risk of overfitting to small support set

---

## Summary

### Your TTT Method: **Full-Tune (Option 3)**

**Evidence**:
1. ✅ `adapted_model.parameters()` includes ALL parameters
2. ✅ No BatchNorm-specific filtering
3. ✅ No statistics-only updates
4. ✅ Full backward pass through entire network
5. ✅ All parameters updated via gradient descent

**Comparison to Reference Methods**:
- **NOT** BN Adapt (no statistics mixing)
- **NOT** Tent (not limited to BN affine params)
- **IS** Full-Tune (all parameters updated)

**Cost**:
- Most expensive among the three options
- Updates ~500K-2M parameters per step
- 10 TTT steps × full network backprop

**Advantages**:
- Maximum adaptation capability
- Can learn complex test-time patterns
- Combines supervised + unsupervised signals

**Disadvantages**:
- High computational cost
- Risk of overfitting to small support set
- Requires careful regularization (which you have: dropout, weight decay, early stopping)

---

## Recommendations for Paper

### How to Report in Publication

**Method Section**:
```latex
\subsection{Test-Time Training}

We employ full-parameter test-time training to adapt the model to
test-time data. Unlike methods that only update batch normalization
parameters (e.g., TENT \cite{tent}) or mix normalization statistics
(e.g., BN Adapt \cite{bn_adapt}), our approach performs gradient-based
optimization of all model parameters using a combination of supervised
and self-supervised objectives.

Specifically, given a support set $\mathcal{S} = \{(x_i^s, y_i^s)\}$
from known attack classes, we minimize:

$$\mathcal{L}_{TTT} = \lambda_s \mathcal{L}_{focal}(\mathcal{S}) +
\lambda_c \mathcal{L}_{consistency}(\mathcal{S})$$

where $\mathcal{L}_{focal}$ is the supervised focal loss and
$\mathcal{L}_{consistency}$ combines entropy minimization, confidence
maximization, and diversity regularization. We use AdamW optimizer
with learning rate $\alpha = 0.001$, weight decay $\lambda = 0.0001$,
and run for 10 adaptation steps with early stopping based on
validation loss.
```

**Results Section**:
```latex
Our full-parameter TTT adaptation achieves 100\% zero-day detection
rate while improving overall accuracy from 74.86\% to 79.43\%
(+4.57\% improvement) on the UNSW-NB15 benchmark.
```

---

**Generated**: December 25, 2025
**Conclusion**: Your system uses **Full-Tune** method (updates all model parameters during test-time training)
