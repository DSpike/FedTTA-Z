#!/usr/bin/env python3
"""
Test script to verify Shapley values are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from incentives.shapley_value_calculator import ShapleyValueCalculator

def test_shapley_values():
    """Test Shapley value calculation with different client performances"""
    
    # Create calculator
    calculator = ShapleyValueCalculator(num_clients=3, evaluation_metric='accuracy')
    
    # Test with different client performances
    global_performance = 0.95
    individual_performances = {
        'client_1': 0.9567,  # Best performing client
        'client_2': 0.9467,  # Medium performing client  
        'client_3': 0.9033   # Lower performing client
    }
    
    client_data_quality = {
        'client_1': 87.5,
        'client_2': 90.0,
        'client_3': 92.5
    }
    
    client_participation = {
        'client_1': 95.0,
        'client_2': 94.0,
        'client_3': 93.0
    }
    
    # Calculate Shapley values
    contributions = calculator.calculate_shapley_values(
        global_performance=global_performance,
        individual_performances=individual_performances,
        client_data_quality=client_data_quality,
        client_participation=client_participation
    )
    
    print("Shapley Value Results:")
    for contrib in contributions:
        print(f"  {contrib.client_id}: Shapley={contrib.shapley_value:.4f}")
    
    # Check if values are different
    shapley_values = [c.shapley_value for c in contributions]
    if len(set(shapley_values)) > 1:
        print("✅ SUCCESS: Shapley values are different!")
        print(f"   Range: {min(shapley_values):.4f} - {max(shapley_values):.4f}")
    else:
        print("❌ FAILED: All Shapley values are the same!")
    
    return contributions

if __name__ == "__main__":
    test_shapley_values()







