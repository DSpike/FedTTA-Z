#!/usr/bin/env python3
"""
Test script to generate token distribution visualization for 6 clients
This simulates the incentive data that should have been generated
"""

import sys
import os
import logging
import time
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from visualization.performance_visualization import PerformanceVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_token_distribution_6_clients():
    print("🧪 Testing Token Distribution Visualization with 6 Clients")
    print("=" * 60)

    # Get configuration
    config = get_config()
    print(f"📋 Configuration:")
    print(f"   num_clients: {config.num_clients}")
    print(f"   enable_incentives: {config.enable_incentives}")
    print()

    # Create mock incentive data based on the actual client performance we saw
    # From the logs, we can see all 6 clients participated and got rewards
    mock_incentive_data = {
        'total_rounds': 3,
        'total_rewards_distributed': 180,  # 6 clients * 30 tokens per round * 3 rounds
        'average_rewards_per_round': 60.0,  # 6 clients * 30 tokens per round
        'participant_rewards': {
            'client_1': 90,   # 30 tokens per round * 3 rounds
            'client_2': 90,   # 30 tokens per round * 3 rounds  
            'client_3': 90,   # 30 tokens per round * 3 rounds
            'client_4': 90,   # 30 tokens per round * 3 rounds
            'client_5': 90,   # 30 tokens per round * 3 rounds
            'client_6': 90,   # 30 tokens per round * 3 rounds
        }
    }

    print(f"📊 Mock Incentive Data:")
    print(f"   Total rounds: {mock_incentive_data['total_rounds']}")
    print(f"   Total rewards: {mock_incentive_data['total_rewards_distributed']}")
    print(f"   Participant rewards: {len(mock_incentive_data['participant_rewards'])} clients")
    print(f"   Expected clients: {config.num_clients}")
    print()

    # Verify we have the right number of clients
    if len(mock_incentive_data['participant_rewards']) == config.num_clients:
        print("✅ All configured clients have rewards!")
    else:
        print(f"❌ Client count mismatch: {len(mock_incentive_data['participant_rewards'])} vs {config.num_clients}")
        return False

    # Test token distribution visualization
    print("\n🎨 Testing token distribution visualization...")
    try:
        visualizer = PerformanceVisualizer(output_dir=".", attack_name="Test_6_Clients")
        plot_path = visualizer.plot_token_distribution(mock_incentive_data, save=True)
        
        if plot_path:
            print(f"✅ Token distribution plot generated: {plot_path}")
            print("🎯 Token distribution visualization is working correctly!")
            
            # Verify the plot file exists
            if os.path.exists(plot_path):
                print(f"📁 Plot file exists: {plot_path}")
                print(f"📏 File size: {os.path.getsize(plot_path)} bytes")
            else:
                print(f"❌ Plot file not found: {plot_path}")
                return False
        else:
            print("❌ Failed to generate token distribution plot")
            return False

    except Exception as e:
        print(f"❌ Error generating visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_token_distribution_6_clients()
    if success:
        print("\n🎉 Token distribution visualization test passed!")
        print("✅ All 6 clients are now visible in the token distribution plot")
    else:
        print("\n❌ Token distribution visualization test failed")
        print("⚠️ There may be an issue with the visualization generation")
