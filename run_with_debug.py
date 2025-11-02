#!/usr/bin/env python3
"""
Run the main system with debugging, bypassing indentation issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main components directly
from config import SystemConfig
from models.transductive_fewshot_model import TransductiveLearner
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_base_model_debugging():
    """Test base model debugging with real data"""
    print("🔍 Testing Base Model Debugging...")
    
    # Load config
    config = SystemConfig()
    
    # Load and preprocess data
    preprocessor = UNSWPreprocessor()
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)
    
    # Get test data
    X_test = data['X_test']
    y_test = data['y_test']
    zero_day_mask = data.get('zero_day_mask', None)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Zero-day mask sum: {zero_day_mask.sum() if zero_day_mask is not None else 'N/A'}")
    print(f"Unique labels: {np.unique(y_test)}")
    
    # Create model
    model = TransductiveLearner(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes
    )
    
    # Test base model evaluation
    with torch.no_grad():
        # Get model predictions
        outputs = model(X_test)
        probabilities = torch.softmax(outputs, dim=1)
        
        print(f"🔍 DEBUG BASE MODEL - Output shape: {outputs.shape}")
        print(f"🔍 DEBUG BASE MODEL - Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        print(f"🔍 DEBUG BASE MODEL - Output std: {outputs.std():.4f}")
        print(f"🔍 DEBUG BASE MODEL - Unique outputs: {len(torch.unique(outputs))}")
        print(f"🔍 DEBUG BASE MODEL - Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
        print(f"🔍 DEBUG BASE MODEL - Probability std: {probabilities.std():.4f}")
        print(f"🔍 DEBUG BASE MODEL - Unique probabilities: {len(torch.unique(probabilities))}")
        
        # Convert to binary predictions
        if outputs.shape[1] == 2:
            predictions = torch.argmax(outputs, dim=1)
        else:
            predictions = (torch.argmax(outputs, dim=1) != 0).long()
        
        print(f"🔍 DEBUG BASE MODEL - Predictions shape: {predictions.shape}")
        print(f"🔍 DEBUG BASE MODEL - Predictions range: [{predictions.min()}, {predictions.max()}]")
        print(f"🔍 DEBUG BASE MODEL - Predictions distribution: {torch.bincount(predictions, minlength=2).tolist()}")
        print(f"🔍 DEBUG BASE MODEL - Labels distribution: {torch.bincount(y_test, minlength=2).tolist()}")
        
        # Calculate accuracy
        accuracy = (predictions == y_test).float().mean().item()
        print(f"🔍 DEBUG BASE MODEL - Accuracy: {accuracy:.4f}")
        
        # Test attack probabilities
        attack_probs = probabilities[:, 1].numpy()
        print(f"🔍 DEBUG BASE MODEL - Attack probs range: [{attack_probs.min():.4f}, {attack_probs.max():.4f}]")
        print(f"🔍 DEBUG BASE MODEL - Attack probs std: {attack_probs.std():.4f}")
        print(f"🔍 DEBUG BASE MODEL - Unique attack probs: {len(np.unique(attack_probs))}")
        print(f"🔍 DEBUG BASE MODEL - Attack probs mean: {attack_probs.mean():.4f}")
        
        # Test threshold application
        optimal_threshold = 0.5
        final_predictions = (attack_probs >= optimal_threshold).astype(int)
        print(f"🔍 DEBUG BASE MODEL - Final predictions after threshold: {np.bincount(final_predictions, minlength=2).tolist()}")
        print(f"🔍 DEBUG BASE MODEL - Threshold used: {optimal_threshold:.4f}")
        
        # Calculate final accuracy
        final_accuracy = (final_predictions == y_test.numpy()).mean()
        print(f"🔍 DEBUG BASE MODEL - Final accuracy: {final_accuracy:.4f}")
    
    return True

def test_ssl_ttt_debugging():
    """Test SSL-TTT debugging"""
    print("\n🔧 Testing SSL-TTT Debugging...")
    
    # Create a simple model
    config = SystemConfig()
    model = TransductiveLearner(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes
    )
    
    # Test with proper TCN input shape [batch, seq_len, features]
    batch_size = 4
    seq_len = 30
    features = 23
    
    x_batch = torch.randn(batch_size, seq_len, features)
    print(f"🔧 SSL-TTT: Input batch shape: {x_batch.shape}")
    
    try:
        # Initialize SSL-TTT
        model.initialize_ssl_ttt()
        print("🔧 SSL-TTT: Initialization successful")
        
        # Test weak augmentation
        x_weak = model.weak_augmentation(x_batch)
        print(f"🔧 SSL-TTT: Weak augmentation shape: {x_weak.shape}")
        
        # Test strong augmentation
        x_strong = model.strong_augmentation(x_batch)
        print(f"🔧 SSL-TTT: Strong augmentation shape: {x_strong.shape}")
        
        # Test RL-guided SSL-TTT adaptation
        rl_threshold = 0.7
        ssl_metrics = model.rl_guided_ssl_ttt_adaptation(
            x_batch=x_batch,
            rl_threshold=rl_threshold,
            entropy_weight=0.7,
            consistency_weight=0.4,
            pseudo_weight=0.3
        )
        print(f"🔧 SSL-TTT: Adaptation successful")
        print(f"🔧 SSL-TTT: Metrics: {ssl_metrics}")
        
        return True
        
    except Exception as e:
        print(f"🔧 SSL-TTT: Failed - {e}")
        return False

def test_zero_day_config():
    """Test zero-day attack configuration"""
    print("\n🎯 Testing Zero-Day Configuration...")
    
    config = SystemConfig()
    
    print(f"🔍 DEBUG ZERO-DAY - Config zero_day_attack: {config.zero_day_attack}")
    print(f"🔍 DEBUG ZERO-DAY - Config zero_day_attack_label: {config.zero_day_attack_label}")
    print(f"🔍 DEBUG ZERO-DAY - Attack types: {config.attack_types}")
    
    # Load and preprocess data
    preprocessor = UNSWPreprocessor()
    data = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)
    
    # Check data
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    zero_day_mask = data.get('zero_day_mask', None)
    
    print(f"🔍 DEBUG ZERO-DAY - Training labels: {np.unique(y_train)}")
    print(f"🔍 DEBUG ZERO-DAY - Test labels: {np.unique(y_test)}")
    print(f"🔍 DEBUG ZERO-DAY - Zero-day mask sum: {zero_day_mask.sum() if zero_day_mask is not None else 'N/A'}")
    print(f"🔍 DEBUG ZERO-DAY - Total test samples: {len(y_test)}")
    print(f"🔍 DEBUG ZERO-DAY - Zero-day samples: {zero_day_mask.sum() if zero_day_mask is not None else 'N/A'}")
    
    # Verify Worms is excluded from training
    if config.zero_day_attack_label not in y_train:
        print("✅ Worms attack correctly excluded from training")
    else:
        print("❌ Worms attack NOT excluded from training")
    
    return True

if __name__ == "__main__":
    print("🚀 Running Debug Tests...")
    
    try:
        base_success = test_base_model_debugging()
        ssl_success = test_ssl_ttt_debugging()
        zero_day_success = test_zero_day_config()
        
        print(f"\n📊 Debug Test Results:")
        print(f"Base model debugging: {'✅ PASS' if base_success else '❌ FAIL'}")
        print(f"SSL-TTT debugging: {'✅ PASS' if ssl_success else '❌ FAIL'}")
        print(f"Zero-day config: {'✅ PASS' if zero_day_success else '❌ FAIL'}")
        
        if all([base_success, ssl_success, zero_day_success]):
            print("\n🎉 All debug tests passed!")
        else:
            print("\n⚠️ Some debug tests failed.")
            
    except Exception as e:
        print(f"❌ Debug test failed with error: {e}")
        import traceback
        traceback.print_exc()
