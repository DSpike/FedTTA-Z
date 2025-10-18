#!/usr/bin/env python3
"""
Test script to check what data is being passed to the token distribution visualization
"""

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_visualization_data():
    """Test what data is being passed to the visualization"""
    
    # Simulate incentive history with Shapley values
    incentive_history = [
        {
            'round_number': 1,
            'total_rewards': 1000,
            'num_rewards': 3,
            'timestamp': 1234567890,
            'shapley_values': {
                'client_1': 0.3118,
                'client_2': 0.3305,
                'client_3': 0.3077
            }
        },
        {
            'round_number': 2,
            'total_rewards': 1200,
            'num_rewards': 3,
            'timestamp': 1234567891,
            'shapley_values': {
                'client_1': 0.3150,
                'client_2': 0.3400,
                'client_3': 0.3050
            }
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
            # Use actual Shapley-based rewards if available
            shapley_values = record['shapley_values']
            total_round_rewards = record['total_rewards']
            
            # Distribute based on Shapley values
            total_shapley = sum(shapley_values.values())
            if total_shapley > 0:
                for client_id, shapley_value in shapley_values.items():
                    # Find corresponding address
                    client_num = int(client_id.split('_')[1]) - 1
                    if client_num < len(client_addresses):
                        address = client_addresses[client_num]
                        reward_share = (shapley_value / total_shapley) * total_round_rewards
                        participant_rewards[address] += reward_share
    
    print("=== VISUALIZATION DATA TEST ===")
    print(f"Incentive history records: {len(incentive_history)}")
    print(f"Shapley values in records: {all('shapley_values' in record for record in incentive_history)}")
    print()
    print("Participant rewards (what gets passed to visualization):")
    for address, reward in participant_rewards.items():
        print(f"  {address}: {reward:.2f} tokens")
    
    print()
    print("Analysis:")
    rewards_list = list(participant_rewards.values())
    if len(set(rewards_list)) > 1:
        print("✅ SUCCESS: Different token amounts for each client!")
        print(f"   Range: {min(rewards_list):.2f} - {max(rewards_list):.2f}")
        print(f"   Difference: {max(rewards_list) - min(rewards_list):.2f} tokens")
    else:
        print("❌ FAILURE: All clients have the same token amount!")
    
    # Test the visualization data structure
    incentive_data = {
        'participant_rewards': participant_rewards,
        'total_rewards_distributed': sum(participant_rewards.values())
    }
    
    print()
    print("Visualization data structure:")
    print(f"  participant_rewards: {incentive_data['participant_rewards']}")
    print(f"  total_rewards_distributed: {incentive_data['total_rewards_distributed']}")

if __name__ == "__main__":
    test_visualization_data()







