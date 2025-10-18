#!/usr/bin/env python3
"""
Test script to verify Shapley values in the main system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BlockchainFederatedLearningSystem

def test_main_shapley():
    """Test Shapley value calculation in the main system"""
    
    # Create system instance
    system = BlockchainFederatedLearningSystem()
    
    # Test the _get_client_training_accuracy method
    print("Testing _get_client_training_accuracy method:")
    for client_id in ['client_1', 'client_2', 'client_3']:
        accuracy = system._get_client_training_accuracy(client_id)
        print(f"  {client_id}: {accuracy:.4f}")
    
    # Test Shapley value calculation with mock data
    print("\nTesting Shapley value calculation:")
    
    # Create mock client updates
    class MockClientUpdate:
        def __init__(self, client_id, validation_accuracy):
            self.client_id = client_id
            self.validation_accuracy = validation_accuracy
    
    client_updates = [
        MockClientUpdate('client_1', 0.9567),
        MockClientUpdate('client_2', 0.9467),
        MockClientUpdate('client_3', 0.9033)
    ]
    
    # Calculate Shapley values
    shapley_values = system._calculate_shapley_values(client_updates, 0.95)
    
    print("Shapley Value Results:")
    for client_id, shapley_value in shapley_values.items():
        print(f"  {client_id}: Shapley={shapley_value:.4f}")
    
    # Check if values are different
    values = list(shapley_values.values())
    if len(set(values)) > 1:
        print("✅ SUCCESS: Shapley values are different!")
        print(f"   Range: {min(values):.4f} - {max(values):.4f}")
    else:
        print("❌ FAILED: All Shapley values are the same!")
    
    return shapley_values

if __name__ == "__main__":
    test_main_shapley()







