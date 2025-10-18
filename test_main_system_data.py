#!/usr/bin/env python3
"""
Test script to check the actual data flow in the main system
"""

import sys
import os
import logging

# Add the parent directory to the sys.path to allow importing from 'incentives'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import BlockchainFederatedIncentiveSystem
from coordinators.blockchain_fedavg_coordinator import ClientUpdate

# Configure logging for this test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_main_system_data():
    """Test the actual data flow in the main system"""
    
    print("=== TESTING MAIN SYSTEM DATA FLOW ===")
    
    # Mock the configuration
    config = MagicMock()
    config.num_clients = 3
    config.epochs = 1
    config.rounds = 1
    config.log_level = "INFO"
    config.attack_type = "DoS"
    config.device = "cpu"
    config.enable_incentives = True
    config.ethereum_rpc_url = "http://localhost:7545"
    config.incentive_contract_address = "0x1234567890123456789012345678901234567890"
    config.private_key = "0x0123456789012345678901234567890123456789012345678901234567890123"
    config.aggregator_address = "0xCD3a95b26EA98a04934CCf6C766f9406496CA986"

    # Create a mock system
    system = BlockchainFederatedIncentiveSystem(config)
    system.logger = logger
    
    # Initialize incentive history with mock data that has different Shapley values
    system.incentive_history = [
        {
            'round_number': 1,
            'total_rewards': 1000,
            'shapley_values': {
                'client_1': 0.30,  # Different values
                'client_2': 0.45,  # Different values
                'client_3': 0.25   # Different values
            }
        },
        {
            'round_number': 2,
            'total_rewards': 1200,
            'shapley_values': {
                'client_1': 0.35,  # Different values
                'client_2': 0.40,  # Different values
                'client_3': 0.25   # Different values
            }
        }
    ]
    
    print(f"Incentive history records: {len(system.incentive_history)}")
    for i, record in enumerate(system.incentive_history):
        print(f"  Round {record['round_number']}: Total={record['total_rewards']}, Shapley={record['shapley_values']}")
    
    # Check if incentive_manager is None
    print(f"\nIncentive manager: {system.incentive_manager}")
    print(f"Incentive manager type: {type(system.incentive_manager)}")
    
    # Try to get the incentive summary
    try:
        incentive_summary = system.get_incentive_summary()
        
        print(f"\nIncentive summary:")
        print(f"  Total rounds: {incentive_summary.get('total_rounds', 0)}")
        print(f"  Total rewards distributed: {incentive_summary.get('total_rewards_distributed', 0)}")
        print(f"  Participant rewards: {incentive_summary.get('participant_rewards', {})}")
        
        # Check if participant rewards are different
        participant_rewards = incentive_summary.get('participant_rewards', {})
        if participant_rewards:
            token_amounts = list(participant_rewards.values())
            print(f"\nToken amounts: {token_amounts}")
            
            if len(set(token_amounts)) > 1:
                print("✅ SUCCESS: Different token amounts for each client!")
                print(f"   Range: {min(token_amounts):.2f} - {max(token_amounts):.2f}")
                print(f"   Difference: {max(token_amounts) - min(token_amounts):.2f} tokens")
            else:
                print("❌ FAILURE: All token amounts are the same!")
        else:
            print("❌ FAILURE: No participant rewards found!")
            
    except Exception as e:
        print(f"❌ ERROR getting incentive summary: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from unittest.mock import MagicMock
    test_main_system_data()







