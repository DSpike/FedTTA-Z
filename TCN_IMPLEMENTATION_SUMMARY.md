# TCN-based Multi-Scale Feature Extractor Implementation Summary

## Overview

This implementation provides a TCN-based multi-scale feature extractor and minimally modified preprocessing pipeline for a PyTorch `TransductiveLearner` class, designed for binary classification (normal vs. attack) on the UNSW-NB15 dataset.

## Components Implemented

### 1. TCN Module (`models/tcn_module.py`)

#### `TCN` Class

- **Purpose**: Single TCN with 3 layers and causal dilated convolutions
- **Architecture**:
  - 3 TCN blocks with dilations [1, 2, 4] for short-, medium-, long-term patterns
  - Kernel sizes [3, 3, 5] (larger kernel for long-term patterns)
  - Dropout=0.2, residual connections
- **Input**: `(batch_size, sequence_length=30, input_dim=40)`
- **Output**: `(batch_size, sequence_length=30, hidden_dim=128)`

#### `TCNBlock` Class

- **Purpose**: Individual TCN block with causal dilated convolution
- **Features**: Residual connections, dropout, weight initialization
- **Causal behavior**: Truncates future information to maintain causality

#### `MultiScaleTCN` Class

- **Purpose**: Multi-scale TCN with different hidden dimensions
- **Architecture**: Three TCN branches with hidden_dims=[128, 64, 256]
- **Output**: Concatenated features `(batch_size, 448)` where 448 = 128 + 64 + 256
- **Pooling**: Uses last timestep from each branch

### 2. TCNTransductiveLearner (`models/tcn_transductive_learner.py`)

#### Key Features

- **TCN Integration**: Replaces `nn.Linear` branches with TCN-based multi-scale feature extraction
- **Architecture**:
  - MultiScaleTCN with hidden_dims=[128, 64, 256] → total_dim=448
  - Feature projection: `nn.Linear(448, embedding_dim=64)`
  - Self-attention: `nn.MultiheadAttention(embedding_dim=64, num_heads=4)`
  - Classification head with dropout layers
- **TTT Support**: Retains `ThresholdAgent` for test-time training
- **Parameters**: transductive_steps=15, transductive_lr=0.0005

#### Methods

- `forward()`: Complete forward pass through TCN → projection → attention → classification
- `extract_features()`: Extract features without classification
- `transductive_adaptation()`: Perform TTT adaptation with support/query sets
- `get_confidence_scores()`: Get confidence scores for sample selection
- `select_samples_for_ttt()`: RL-based sample selection for TTT

### 3. Enhanced Preprocessor (`preprocessing/blockchain_federated_unsw_preprocessor.py`)

#### New Method: `create_sequences()`

- **Purpose**: Create sequences from preprocessed data with optional zero-padding
- **Parameters**:
  - `sequence_length=30`: Length of each sequence
  - `stride=1`: Step size for sliding window
  - `zero_pad=True`: Zero-pad short sequences to sequence_length
- **Input**: Preprocessed features `(n_samples, n_features)`
- **Output**: Sequences `(n_sequences, sequence_length, n_features)`

#### Features

- **Sliding Window**: Creates overlapping sequences with configurable stride
- **Zero Padding**: Pads short sequences to ensure consistent length
- **Label Assignment**: Uses label from last timestep of each sequence
- **Flexible Parameters**: Supports different sequence lengths and strides

## Usage Example

```python
from models.tcn_transductive_learner import TCNTransductiveLearner
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

# Initialize preprocessor
preprocessor = UNSWPreprocessor()

# Preprocess data (existing pipeline)
data = preprocessor.preprocess_unsw_dataset(zero_day_attack='Worms')

# Create sequences
X_seq, y_seq = preprocessor.create_sequences(
    data['X_train'], data['y_train'],
    sequence_length=30, zero_pad=True
)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_seq)
y_tensor = torch.LongTensor(y_seq)

# Initialize TCN-based learner
learner = TCNTransductiveLearner(
    input_dim=40,
    sequence_length=30,
    hidden_dim=128,
    embedding_dim=64,
    num_classes=2,
    transductive_steps=15,
    transductive_lr=0.0005
)

# Forward pass
logits = learner(X_tensor)

# Transductive adaptation
support_x, support_y = X_tensor[:20], y_tensor[:20]
query_x, query_y = X_tensor[20:30], y_tensor[20:30]

adapted_logits, metrics = learner.transductive_adaptation(
    support_x, support_y, query_x, query_y
)
```

## Key Specifications Met

✅ **TCN Module**: 3 layers, causal dilated convolutions (dilations 1, 2, 4)  
✅ **Kernel Sizes**: 3 for dilations 1,2 and 5 for dilation 4  
✅ **Dropout**: 0.2 throughout TCN layers  
✅ **Residual Connections**: Implemented in TCNBlock  
✅ **Input/Output**: (batch_size, 30, 40) → (batch_size, 30, hidden_dim)  
✅ **Multi-Scale**: Three TCN branches (128, 64, 256) → 448 total features  
✅ **Feature Projection**: nn.Linear(448, 64)  
✅ **Self-Attention**: nn.MultiheadAttention(64, 4)  
✅ **ThresholdAgent**: Retained for TTT (15 steps, 0.0005 lr)  
✅ **Sequence Creation**: Sliding window with zero-padding  
✅ **Binary Classification**: Normal vs. attack on UNSW-NB15

## Testing

All components have been tested with:

- ✅ TCN module functionality
- ✅ TCNTransductiveLearner forward pass and adaptation
- ✅ Preprocessor sequence creation with padding
- ✅ End-to-end integration

The implementation is ready for integration with the existing blockchain federated learning system.





