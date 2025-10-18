#!/usr/bin/env python3
"""
Debug script to understand why Shapley values are identical
"""

import sys
import os
from collections import namedtuple
import numpy as np

# Add the parent directory to the sys.path to allow importing from 'incentives'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from incentives.shapley_value_calculator import ShapleyValueCalculator

def debug_shapley_calculation():
    """Debug the Shapley value calculation step by step"""
    
    # Create calculator
    calculator = ShapleyValueCalculator(num_clients=3, evaluation_metric='accuracy')
    
    # Use the real performance values from the logs (Round 6)
    global_performance = 0.95
    individual_performances = {
        'client_1': 0.9350,  # Medium performing client
        'client_2': 0.9650,  # Best performing client
        'client_3': 0.9250   # Lower performing client
    }
    
    client_data_quality = {
        'client_1': 87.5,
        'client_2': 90.0,
        'client_3': 92.5
    }
    
    client_participation = {
        'client_1': 1.0,
        'client_2': 1.0,
        'client_3': 1.0
    }
    
    print("=== DEBUGGING SHAPLEY CALCULATION ===")
    print(f"Individual performances: {individual_performances}")
    print(f"Global performance: {global_performance}")
    
    # Test coalition performance calculation
    print("\n=== COALITION PERFORMANCE CALCULATION ===")
    test_coalitions = [
        ['client_1'],
        ['client_2'], 
        ['client_3'],
        ['client_1', 'client_2'],
        ['client_1', 'client_3'],
        ['client_2', 'client_3'],
        ['client_1', 'client_2', 'client_3']
    ]
    
    for coalition in test_coalitions:
        perf = calculator.calculate_coalition_performance(
            coalition, global_performance, individual_performances
        )
        print(f"Coalition {coalition}: {perf:.4f}")
    
    # Test marginal contributions
    print("\n=== MARGINAL CONTRIBUTIONS ===")
    for client_id in ['client_1', 'client_2', 'client_3']:
        print(f"\nClient {client_id}:")
        for coalition in test_coalitions:
            if client_id in coalition:
                # Coalition with client
                coalition_with = coalition
                perf_with = calculator.calculate_coalition_performance(
                    coalition_with, global_performance, individual_performances
                )
                
                # Coalition without client
                coalition_without = [c for c in coalition if c != client_id]
                if coalition_without:
                    perf_without = calculator.calculate_coalition_performance(
                        coalition_without, global_performance, individual_performances
                    )
                else:
                    perf_without = 0.0
                
                marginal = perf_with - perf_without
                print(f"  {coalition} -> {coalition_without}: {perf_with:.4f} - {perf_without:.4f} = {marginal:.4f}")
    
    # Calculate actual Shapley values
    print("\n=== SHAPLEY VALUES ===")
    contributions = calculator.calculate_shapley_values(
        global_performance=global_performance,
        individual_performances=individual_performances,
        client_data_quality=client_data_quality,
        client_participation=client_participation
    )
    
    for contrib in contributions:
        print(f"{contrib.client_id}: Shapley={contrib.shapley_value:.4f}")

if __name__ == "__main__":
    debug_shapley_calculation()







