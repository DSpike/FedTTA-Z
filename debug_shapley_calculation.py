#!/usr/bin/env python3
"""
Debug script to test Shapley value calculation
"""

import sys
import os
import logging

# Add the parent directory to the sys.path to allow importing from 'incentives'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from incentives.shapley_value_calculator import ShapleyValueCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_shapley_calculation():
    """Debug the Shapley value calculation with known values"""
    
    print("=== DEBUGGING SHAPLEY VALUE CALCULATION ===")
    
    # Create calculator
    calculator = ShapleyValueCalculator(num_clients=3)
    
    # Test with clearly different individual performances
    global_performance = 0.95
    individual_performances = {
        'client_1': 0.92,  # Lower performance
        'client_2': 0.88,  # Lowest performance  
        'client_3': 0.90   # Middle performance
    }
    client_data_quality = {
        'client_1': 0.9,
        'client_2': 0.85,
        'client_3': 0.88
    }
    client_participation = {
        'client_1': 1.0,
        'client_2': 0.95,
        'client_3': 1.0
    }
    
    print(f"Individual performances: {individual_performances}")
    print(f"Global performance: {global_performance}")
    
    # Test coalition performance calculation
    print("\n=== TESTING COALITION PERFORMANCE ===")
    for r in range(1, 4):
        from itertools import combinations
        for coalition in combinations(['client_1', 'client_2', 'client_3'], r):
            coalition_list = list(coalition)
            performance = calculator.calculate_coalition_performance(
                coalition_list, global_performance, individual_performances
            )
            print(f"Coalition {coalition_list}: {performance:.4f}")
    
    # Calculate Shapley values
    print("\n=== CALCULATING SHAPLEY VALUES ===")
    contributions = calculator.calculate_shapley_values(
        global_performance, individual_performances, 
        client_data_quality, client_participation
    )
    
    print("Shapley Value Results:")
    for contrib in contributions:
        print(f"{contrib.client_id}: Shapley={contrib.shapley_value:.6f}")
    
    # Calculate token rewards
    token_rewards = calculator.calculate_token_rewards(contributions, total_tokens=3000)
    
    print("\nToken Rewards:")
    for client_id, tokens in token_rewards.items():
        print(f"{client_id}: {tokens} tokens")
    
    # Check if values are different
    shapley_values = [contrib.shapley_value for contrib in contributions]
    if len(set(shapley_values)) == 1:
        print("\n❌ PROBLEM: All Shapley values are identical!")
    else:
        print("\n✅ SUCCESS: Shapley values are different!")

if __name__ == "__main__":
    debug_shapley_calculation()









