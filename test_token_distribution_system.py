#!/usr/bin/env python3
"""
Test Token Distribution System
This script tests if token distribution is working correctly
"""

import sys
import os
import logging
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BlockchainFederatedIncentiveSystem
from config import get_config

logger = logging.getLogger(__name__)

def test_token_distribution():
    """Test the token distribution system"""
    
    print("🧪 Testing Token Distribution System")
    print("=" * 50)
    
    try:
        # Initialize the system
        config = get_config()
        system = BlockchainFederatedIncentiveSystem(config)
        
        print("✅ System initialized successfully")
        
        # Simulate some incentive data
        print("📊 Simulating incentive data...")
        
        # Create mock incentive records
        mock_records = [
            {
                'round_number': 1,
                'total_rewards': 1200,
                'num_rewards': 15,
                'individual_rewards': {
                    'client_1': 80,
                    'client_2': 75,
                    'client_3': 90,
                    'client_4': 85,
                    'client_5': 95,
                    'client_6': 70,
                    'client_7': 88,
                    'client_8': 82,
                    'client_9': 76,
                    'client_10': 84,
                    'client_11': 91,
                    'client_12': 77,
                    'client_13': 83,
                    'client_14': 89,
                    'client_15': 86
                },
                'shapley_values': {
                    'client_1': 0.0667,
                    'client_2': 0.0625,
                    'client_3': 0.0750,
                    'client_4': 0.0708,
                    'client_5': 0.0792,
                    'client_6': 0.0583,
                    'client_7': 0.0733,
                    'client_8': 0.0683,
                    'client_9': 0.0633,
                    'client_10': 0.0700,
                    'client_11': 0.0758,
                    'client_12': 0.0642,
                    'client_13': 0.0692,
                    'client_14': 0.0742,
                    'client_15': 0.0717
                },
                'timestamp': time.time()
            },
            {
                'round_number': 2,
                'total_rewards': 1400,
                'num_rewards': 15,
                'individual_rewards': {
                    'client_1': 85,
                    'client_2': 90,
                    'client_3': 88,
                    'client_4': 92,
                    'client_5': 87,
                    'client_6': 89,
                    'client_7': 91,
                    'client_8': 86,
                    'client_9': 88,
                    'client_10': 93,
                    'client_11': 87,
                    'client_12': 89,
                    'client_13': 90,
                    'client_14': 88,
                    'client_15': 91
                },
                'shapley_values': {
                    'client_1': 0.0607,
                    'client_2': 0.0643,
                    'client_3': 0.0629,
                    'client_4': 0.0657,
                    'client_5': 0.0621,
                    'client_6': 0.0636,
                    'client_7': 0.0650,
                    'client_8': 0.0614,
                    'client_9': 0.0629,
                    'client_10': 0.0664,
                    'client_11': 0.0621,
                    'client_12': 0.0636,
                    'client_13': 0.0643,
                    'client_14': 0.0629,
                    'client_15': 0.0650
                },
                'timestamp': time.time()
            }
        ]
        
        # Add mock records to incentive history
        system.incentive_history = mock_records
        
        print(f"✅ Added {len(mock_records)} mock incentive records")
        
        # Test incentive summary
        print("📊 Testing incentive summary...")
        incentive_summary = system.get_incentive_summary()
        
        print(f"   Total Rounds: {incentive_summary['total_rounds']}")
        print(f"   Total Rewards: {incentive_summary['total_rewards_distributed']}")
        print(f"   Participant Rewards: {incentive_summary['participant_rewards']}")
        
        # Check if participant rewards are available
        if incentive_summary['participant_rewards']:
            print("✅ SUCCESS: Individual rewards are available!")
            
            # Test visualization
            print("🎨 Testing token distribution visualization...")
            try:
                from visualization.performance_visualization import PerformanceVisualizer
                visualizer = PerformanceVisualizer(output_dir=".", attack_name="Test")
                
                # Generate token distribution plot
                plot_path = visualizer.plot_token_distribution(incentive_summary, save=True)
                if plot_path:
                    print(f"✅ Token distribution plot generated: {plot_path}")
                    print("🎯 Token distribution visualization is working correctly!")
                else:
                    print("❌ Failed to generate token distribution plot")
                    
            except Exception as e:
                print(f"❌ Error generating visualization: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ FAILURE: Individual rewards are not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing token distribution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_token_distribution()
    if success:
        print("\n🎉 Token distribution system is working correctly!")
    else:
        print("\n❌ Token distribution system has issues")
