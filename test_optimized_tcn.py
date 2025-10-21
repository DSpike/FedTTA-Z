#!/usr/bin/env python3
"""
Test script for optimized TCN-based TransductiveLearner
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.optimized_tcn_transductive_learner import OptimizedTCNTransductiveLearner
from models.optimized_tcn_module import OptimizedTCN, OptimizedMultiScaleTCN

def test_optimized_tcn():
    """Test the OptimizedTCN module"""
    print("Testing OptimizedTCN module...")
    
    # Test parameters
    batch_size = 4
    sequence_length = 15
    input_dim = 40
    hidden_dim = 64
    
    # Create test data
    x = torch.randn(batch_size, sequence_length, input_dim)
    
    # Test single TCN
    tcn = OptimizedTCN(input_dim, hidden_dim, [hidden_dim] * 2, kernel_size=3, dropout=0.2)
    output = tcn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"TCN output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {sequence_length}, {hidden_dim})")
    
    assert output.shape == (batch_size, sequence_length, hidden_dim), f"Expected {(batch_size, sequence_length, hidden_dim)}, got {output.shape}"
    print("✅ OptimizedTCN test passed!")
    
    return tcn

def test_optimized_multi_scale_tcn():
    """Test the OptimizedMultiScaleTCN module"""
    print("\nTesting OptimizedMultiScaleTCN module...")
    
    # Test parameters
    batch_size = 4
    sequence_length = 15
    input_dim = 40
    hidden_dim = 64
    
    # Create test data
    x = torch.randn(batch_size, sequence_length, input_dim)
    
    # Test multi-scale TCN
    multi_tcn = OptimizedMultiScaleTCN(input_dim, sequence_length, hidden_dim=hidden_dim, dropout=0.2)
    output = multi_tcn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"MultiScale TCN output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {hidden_dim + hidden_dim//2 + hidden_dim*2})")
    print(f"Total dimension: {multi_tcn.output_dim}")
    
    expected_dim = hidden_dim + hidden_dim//2 + hidden_dim*2  # 64 + 32 + 128 = 224
    assert output.shape == (batch_size, expected_dim), f"Expected {(batch_size, expected_dim)}, got {output.shape}"
    assert multi_tcn.output_dim == expected_dim, f"Expected {expected_dim}, got {multi_tcn.output_dim}"
    print("✅ OptimizedMultiScaleTCN test passed!")
    
    return multi_tcn

def test_optimized_transductive_learner():
    """Test the OptimizedTCNTransductiveLearner"""
    print("\nTesting OptimizedTCNTransductiveLearner...")
    
    # Test parameters
    batch_size = 4
    sequence_length = 15
    input_dim = 40
    hidden_dim = 64
    embedding_dim = 64
    num_classes = 2
    
    # Create test data
    x = torch.randn(batch_size, sequence_length, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # Create model
    model = OptimizedTCNTransductiveLearner(
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        transductive_steps=5,
        transductive_lr=0.0005
    )
    
    # Test forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    
    assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
    print("✅ Forward pass test passed!")
    
    # Test feature extraction
    features = model.extract_features(x)
    print(f"Extracted features shape: {features.shape}")
    print(f"Expected features shape: ({batch_size}, {embedding_dim})")
    
    assert features.shape == (batch_size, embedding_dim), f"Expected {(batch_size, embedding_dim)}, got {features.shape}"
    print("✅ Feature extraction test passed!")
    
    # Test confidence scores
    confidence = model.get_confidence_scores(x)
    print(f"Confidence scores shape: {confidence.shape}")
    print(f"Expected confidence shape: ({batch_size},)")
    
    assert confidence.shape == (batch_size,), f"Expected {(batch_size,)}, got {confidence.shape}"
    print("✅ Confidence scores test passed!")
    
    # Test transductive adaptation
    print("\nTesting transductive adaptation...")
    support_x = torch.randn(8, sequence_length, input_dim)
    support_y = torch.randint(0, num_classes, (8,))
    query_x = torch.randn(4, sequence_length, input_dim)
    query_y = torch.randint(0, num_classes, (4,))
    
    adapted_logits, metrics = model.transductive_adaptation(query_x)
    
    print(f"Adapted logits shape: {adapted_logits.shape}")
    print(f"Expected adapted logits shape: ({4}, {num_classes})")
    print(f"Adaptation metrics keys: {list(metrics.keys())}")
    
    assert adapted_logits.shape == (4, num_classes), f"Expected {(4, num_classes)}, got {adapted_logits.shape}"
    print("✅ Transductive adaptation test passed!")
    
    # Test meta-training
    print("\nTesting meta-training...")
    meta_tasks = []
    for i in range(3):  # Small number for testing
        meta_task = {
            'support_x': torch.randn(5, sequence_length, input_dim),
            'support_y': torch.randint(0, num_classes, (5,)),
            'query_x': torch.randn(3, sequence_length, input_dim),
            'query_y': torch.randint(0, num_classes, (3,))
        }
        meta_tasks.append(meta_task)
    
    training_history = model.meta_train(meta_tasks, meta_epochs=2)
    
    print(f"Training history keys: {list(training_history.keys())}")
    print(f"Epoch losses length: {len(training_history['epoch_losses'])}")
    print(f"Epoch accuracies length: {len(training_history['epoch_accuracies'])}")
    
    assert 'epoch_losses' in training_history, "Missing epoch_losses in training history"
    assert 'epoch_accuracies' in training_history, "Missing epoch_accuracies in training history"
    print("✅ Meta-training test passed!")
    
    return model

def test_device_compatibility():
    """Test device compatibility"""
    print("\nTesting device compatibility...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Testing on GPU: {device}")
        
        model = OptimizedTCNTransductiveLearner(
            input_dim=40,
            sequence_length=15,
            hidden_dim=64,
            embedding_dim=64,
            num_classes=2
        )
        
        # Move model to GPU
        model = model.to(device)
        print(f"Model device: {model.device}")
        
        # Test forward pass on GPU
        x = torch.randn(2, 15, 40).to(device)
        logits = model(x)
        
        print(f"Input device: {x.device}")
        print(f"Output device: {logits.device}")
        
        assert logits.device.type == device.type, f"Expected {device.type}, got {logits.device.type}"
        print("✅ GPU compatibility test passed!")
    else:
        print("CUDA not available, skipping GPU test")

def main():
    """Run all tests"""
    print("🚀 Starting Optimized TCN Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        tcn = test_optimized_tcn()
        multi_tcn = test_optimized_multi_scale_tcn()
        model = test_optimized_transductive_learner()
        
        # Test device compatibility
        test_device_compatibility()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("\nOptimized TCN Implementation Summary:")
        print(f"  - TCN layers: 2 (reduced from 3)")
        print(f"  - Hidden dimensions: 64, 32, 128")
        print(f"  - Total output dimension: 224")
        print(f"  - Sequence length: 15")
        print(f"  - Input dimension: 40")
        print(f"  - Transductive steps: 5 (reduced from 15)")
        print(f"  - Embedding dimension: 64")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
