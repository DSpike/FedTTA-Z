#!/usr/bin/env python3
"""
Debug script to test Shapley calculation exactly as it's done in the main system
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

def debug_main_shapley():
    print("🔍 DEBUGGING SHAPLEY CALCULATION IN MAIN SYSTEM")
    print("=" * 60)

    # Get configuration
    config = get_config()
    print(f"📋 Configuration: {config.num_clients} clients")
    
    # Create enhanced config
    enhanced_config = EnhancedSystemConfig(
        num_clients=config.num_clients,
        num_rounds=1,  # Just 1 round for testing
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        enable_incentives=config.enable_incentives,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("🏗️ Creating system...")
    system = BlockchainFederatedIncentiveSystem(enhanced_config)

    print("🔧 Initializing system...")
    success = system.initialize_system()
    if not success:
        print("❌ System initialization failed")
        return

    print("✅ System initialized successfully")
    
    # Create mock round results with the actual performance data from the recent run
    mock_round_results = {
        'client_updates': [
            type('ClientUpdate', (), {
                'client_id': f'client_{i+1}',
                'validation_accuracy': [0.594, 0.496, 0.556, 0.650, 0.555, 0.526][i],
                'training_loss': [0.4, 0.5, 0.45, 0.35, 0.45, 0.48][i],
                'sample_count': 100,
                'model_parameters': {'test': 'data'}
            }) for i in range(6)
        ]
    }
    
    print("🧪 Testing Shapley calculation with mock data...")
    print(f"Mock client updates: {len(mock_round_results['client_updates'])}")
    for update in mock_round_results['client_updates']:
        print(f"  {update.client_id}: accuracy={update.validation_accuracy}")
    
    # Test the _calculate_shapley_values method directly
    try:
        shapley_values = system._calculate_shapley_values(
            round_num=1,
            round_results=mock_round_results,
            previous_accuracy=0.3,
            current_accuracy=0.3554
        )
        
        print(f"✅ Shapley calculation successful!")
        print(f"📊 Results:")
        for client_id, value in shapley_values.items():
            print(f"  {client_id}: {value:.6f}")
        
        # Check if values are different
        values = list(shapley_values.values())
        unique_values = len(set(values))
        print(f"\n🔍 Analysis:")
        print(f"  Unique Shapley values: {unique_values}")
        print(f"  All values equal: {unique_values == 1}")
        print(f"  Value range: {min(values):.6f} to {max(values):.6f}")
        
        if unique_values == 1:
            print("❌ PROBLEM: All Shapley values are identical!")
        else:
            print("✅ SUCCESS: Shapley values are differentiated!")
            
    except Exception as e:
        print(f"❌ Error in main system Shapley calculation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    debug_main_shapley()
