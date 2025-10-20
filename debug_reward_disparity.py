#!/usr/bin/env python3
"""
Debug script to investigate the reward disparity issue
"""

import sys
import os
import logging
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from main import BlockchainFederatedIncentiveSystem, EnhancedSystemConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_reward_disparity():
    print("🔍 Debugging Reward Disparity Issue")
    print("=" * 50)

    # Get configuration
    config = get_config()
    print(f"📋 Configuration:")
    print(f"   num_clients: {config.num_clients}")
    print(f"   enable_incentives: {config.enable_incentives}")
    print()

    # Create enhanced config
    enhanced_config = EnhancedSystemConfig(
        num_clients=config.num_clients,
        num_rounds=1,  # Just one round for debugging
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        enable_incentives=config.enable_incentives,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("🏗️  Creating system...")
    system = BlockchainFederatedIncentiveSystem(enhanced_config)

    print("🔧 Initializing system...")
    success = system.initialize_system()

    if not success:
        print("❌ System initialization failed")
        return False

    print(f"📊 Coordinator client count: {len(system.coordinator.clients)}")
    print(f"📊 Expected client count: {config.num_clients}")

    # Test Shapley value calculation with mock data
    print("\n🧪 Testing Shapley value calculation...")
    
    # Create mock round results
    mock_round_results = {
        'client_updates': [
            type('ClientUpdate', (), {
                'client_id': f'client_{i+1}',
                'validation_accuracy': 0.5 + (i * 0.1),  # Different accuracies
                'training_loss': 0.5 - (i * 0.05),
                'sample_count': 100,
                'model_parameters': {'test': 'data'}
            }) for i in range(config.num_clients)
        ]
    }

    # Test Shapley calculation
    try:
        shapley_values = system._calculate_shapley_values(
            round_num=1,
            round_results=mock_round_results,
            previous_accuracy=0.3,
            current_accuracy=0.7
        )
        
        print(f"📊 Shapley values calculated: {len(shapley_values)} clients")
        for client_id, value in shapley_values.items():
            print(f"   {client_id}: {value:.4f}")
        
        if not shapley_values:
            print("❌ No Shapley values calculated - this explains the fallback!")
            return False
        else:
            print("✅ Shapley values calculated successfully")
            
    except Exception as e:
        print(f"❌ Error calculating Shapley values: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    # Test reward calculation
    print("\n🧪 Testing reward calculation...")
    
    # Create mock client contributions
    mock_contributions = []
    for i in range(config.num_clients):
        contribution = {
            'client_address': f'0x{"1234567890abcdef" * 4}',
            'model_parameters': {'test': 'data'},
            'previous_accuracy': 0.3,
            'current_accuracy': 0.5 + (i * 0.1),
            'data_quality': 80.0 + (i * 5.0),
            'reliability': 85.0 + (i * 2.0)
        }
        mock_contributions.append(contribution)

    # Test reward distribution
    try:
        reward_distributions = system.incentive_manager.process_round_contributions(
            round_number=1,
            client_contributions=mock_contributions,
            shapley_values=shapley_values
        )
        
        print(f"📊 Reward distributions calculated: {len(reward_distributions)} clients")
        for i, reward_dist in enumerate(reward_distributions):
            print(f"   Client {i+1}: {reward_dist.token_amount} tokens")
        
        # Check for disparity
        token_amounts = [rd.token_amount for rd in reward_distributions]
        min_tokens = min(token_amounts)
        max_tokens = max(token_amounts)
        disparity_ratio = max_tokens / min_tokens if min_tokens > 0 else float('inf')
        
        print(f"\n📊 Reward Analysis:")
        print(f"   Min tokens: {min_tokens}")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Disparity ratio: {disparity_ratio:.2f}x")
        
        if disparity_ratio > 10:
            print("❌ High disparity detected - this is the problem!")
        else:
            print("✅ Rewards are reasonably balanced")
            
    except Exception as e:
        print(f"❌ Error calculating rewards: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    import torch
    success = debug_reward_disparity()
    if success:
        print("\n🎉 Debug completed successfully")
    else:
        print("\n❌ Debug failed")
