# Optimized TCN-based Multi-scale Feature Extractor Implementation

## Overview

This implementation provides an optimized TCN-based multi-scale feature extractor for a PyTorch `TransductiveLearner` class, designed for binary classification (normal vs. attack) on the UNSW-NB15 dataset.

## Key Optimizations

### 1. Reduced Complexity

- **TCN Layers**: Reduced from 3 to 2 layers for better efficiency
- **Hidden Dimensions**: Optimized to 64, 32, 128 (total: 224 dimensions)
- **Transductive Steps**: Reduced from 15 to 5 for faster adaptation
- **Sequence Length**: Optimized for 15 timesteps (reduced from 30)

### 2. Architecture Specifications

#### TCN Module (`OptimizedTCN`)

- **Input**: `(batch_size, 15, 40)`
- **Output**: `(batch_size, 15, hidden_dim)`
- **Layers**: 2 temporal blocks with causal dilated convolutions
- **Dilations**: 1, 2, 4 for short-, medium-, long-term patterns
- **Kernel Sizes**: 3 (standard), 5 (for dilation=4 branch)
- **Dropout**: 0.2
- **Residual Connections**: Yes

#### Multi-Scale TCN (`OptimizedMultiScaleTCN`)

- **Three TCN Branches**:
  - Branch 1: `hidden_dim=64` → `(batch_size, 15, 64)`
  - Branch 2: `hidden_dim=32` → `(batch_size, 15, 32)`
  - Branch 3: `hidden_dim=128` → `(batch_size, 15, 128)`
- **Pooling**: Last timestep pooling
- **Output**: `(batch_size, 224)` where 224 = 64 + 32 + 128

#### TransductiveLearner (`OptimizedTCNTransductiveLearner`)

- **Feature Projection**: `nn.Linear(224, 64)`
- **Self-Attention**: `nn.MultiheadAttention(64, num_heads=4)`
- **Classification**: 3-layer MLP with dropout
- **TTT Parameters**:
  - `transductive_steps=5`
  - `transductive_lr=0.0005`
- **ThresholdAgent**: RL-based sample selection

## File Structure

```
models/
├── optimized_tcn_module.py              # Optimized TCN and MultiScaleTCN classes
├── optimized_tcn_transductive_learner.py # Optimized TransductiveLearner class
└── test_optimized_tcn.py               # Comprehensive test suite
```

## Key Features

### 1. Optimized TCN Architecture

- **2-layer design** for reduced computational complexity
- **Causal dilated convolutions** for temporal pattern recognition
- **Residual connections** for stable training
- **Gradient clipping** for training stability

### 2. Multi-Scale Feature Extraction

- **Three parallel TCN branches** with different scales
- **Last timestep pooling** for sequence summarization
- **Concatenated features** for comprehensive representation

### 3. Transductive Learning

- **Test-Time Training (TTT)** with 5 adaptation steps
- **RL-based sample selection** for dynamic thresholding
- **Support and consistency losses** for few-shot learning
- **Feature alignment** for prototype-based adaptation

### 4. Stability Improvements

- **Gradient clipping** (max_norm=1.0)
- **Weight decay** (1e-5)
- **Memory management** with periodic GPU cache clearing
- **Device compatibility** with automatic device detection

## Usage Example

```python
from models.optimized_tcn_transductive_learner import OptimizedTCNTransductiveLearner

# Create model
model = OptimizedTCNTransductiveLearner(
    input_dim=40,
    sequence_length=15,
    hidden_dim=64,
    embedding_dim=64,
    num_classes=2,
    transductive_steps=5,
    transductive_lr=0.0005
)

# Forward pass
x = torch.randn(batch_size, 15, 40)
logits = model(x)  # Shape: (batch_size, 2)

# Transductive adaptation
adapted_logits, metrics = model.transductive_adaptation(
    support_x, support_y, query_x, query_y
)

# Meta-training
training_history = model.meta_train(meta_tasks, meta_epochs=100)
```

## Performance Characteristics

### Computational Efficiency

- **Reduced parameters** due to 2-layer TCN design
- **Faster training** with reduced transductive steps
- **Memory efficient** with optimized sequence length
- **GPU optimized** with proper device management

### Model Specifications

- **Total TCN Output Dimension**: 224 (64 + 32 + 128)
- **Embedding Dimension**: 64
- **Sequence Length**: 15 timesteps
- **Input Dimension**: 40 features
- **Number of Classes**: 2 (binary classification)

## Testing

The implementation includes comprehensive tests covering:

- ✅ TCN module functionality
- ✅ Multi-scale TCN integration
- ✅ TransductiveLearner forward pass
- ✅ Feature extraction
- ✅ Confidence scoring
- ✅ Transductive adaptation
- ✅ Meta-training
- ✅ Device compatibility (CPU/GPU)

## Integration Notes

This optimized implementation is designed to replace the existing TCN branches in the main system while maintaining compatibility with:

- **Existing preprocessing pipeline** (assumes sequence padding for length=15)
- **Blockchain federated learning framework**
- **Test-time training mechanisms**
- **Evaluation and visualization systems**

The optimized design provides better efficiency and stability while maintaining the core functionality for zero-day attack detection in network intrusion detection systems.





