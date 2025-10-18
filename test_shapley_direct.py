#!/usr/bin/env python3
"""
Test the Shapley calculation directly without the full system
"""

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from incentives.shapley_value_calculator import ShapleyValueCalculator

def test_shapley_direct():
    """Test Shapley calculation directly"""
    
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
    
    print("=== TESTING SHAPLEY CALCULATION DIRECTLY ===")
    print(f"Individual performances: {individual_performances}")
    print(f"Global performance: {global_performance}")
    
    # Calculate Shapley values
    contributions = calculator.calculate_shapley_values(
        global_performance=global_performance,
        individual_performances=individual_performances,
        client_data_quality=client_data_quality,
        client_participation=client_participation
    )
    
    print(f"\nShapley values:")
    shapley_values = {}
    for contrib in contributions:
        print(f"  {contrib.client_id}: {contrib.shapley_value:.4f}")
        shapley_values[contrib.client_id] = contrib.shapley_value
    
    # Test token distribution
    total_tokens = 1000
    token_rewards = calculator.calculate_token_rewards(contributions, total_tokens)
    
    print(f"\nToken rewards (1000 total tokens):")
    for client_id, reward in token_rewards.items():
        print(f"  {client_id}: {reward} tokens")
    
    # Check if values are different
    shapley_list = list(shapley_values.values())
    token_list = list(token_rewards.values())
    
    print(f"\nAnalysis:")
    if len(set(shapley_list)) > 1:
        print("✅ SUCCESS: Different Shapley values!")
        print(f"   Range: {min(shapley_list):.4f} - {max(shapley_list):.4f}")
    else:
        print("❌ FAILURE: All Shapley values are the same!")
    
    if len(set(token_list)) > 1:
        print("✅ SUCCESS: Different token rewards!")
        print(f"   Range: {min(token_list)} - {max(token_list)} tokens")
    else:
        print("❌ FAILURE: All token rewards are the same!")
    
    # Test the incentive summary logic
    print(f"\n=== TESTING INCENTIVE SUMMARY LOGIC ===")
    
    # Simulate incentive history with Shapley values
    incentive_history = [
        {
            'round_number': 1,
            'total_rewards': 1000,
            'shapley_values': shapley_values
        },
        {
            'round_number': 2,
            'total_rewards': 1200,
            'shapley_values': shapley_values
        }
    ]
    
    # Simulate the get_incentive_summary logic
    client_addresses = [
        '0xCD3a95b26EA98a04934CCf6C766f9406496CA986',
        '0x32cE285CF96cf83226552A9c3427Bd58c0A9AccD', 
        '0x8EbA3b47c80a5E31b4Ea6fED4d5De8ebc93B8d6f'
    ]
    
    participant_rewards = {}
    
    # Initialize rewards for each client
    for address in client_addresses:
        participant_rewards[address] = 0
    
    # Sum up actual rewards from each round
    for record in incentive_history:
        if 'shapley_values' in record:
            shapley_values_round = record['shapley_values']
            total_round_rewards = record['total_rewards']
            
            # Distribute based on Shapley values
            total_shapley = sum(shapley_values_round.values())
            if total_shapley > 0:
                for client_id, shapley_value in shapley_values_round.items():
                    # Find corresponding address
                    client_num = int(client_id.split('_')[1]) - 1
                    if client_num < len(client_addresses):
                        address = client_addresses[client_num]
                        reward_share = (shapley_value / total_shapley) * total_round_rewards
                        participant_rewards[address] += reward_share
    
    print(f"Participant rewards (what gets passed to visualization):")
    for address, reward in participant_rewards.items():
        print(f"  {address}: {reward:.2f} tokens")
    
    # Check if rewards are different
    rewards_list = list(participant_rewards.values())
    if len(set(rewards_list)) > 1:
        print("✅ SUCCESS: Different token amounts for each client!")
        print(f"   Range: {min(rewards_list):.2f} - {max(rewards_list):.2f}")
        print(f"   Difference: {max(rewards_list) - min(rewards_list):.2f} tokens")
    else:
        print("❌ FAILURE: All clients have the same token amount!")

if __name__ == "__main__":
    test_shapley_direct()







