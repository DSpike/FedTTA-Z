#!/usr/bin/env python3
"""
Test script to verify the token distribution visualization fix with 6 clients
This script runs the federated learning system with 6 clients and checks the visualization
"""

import sys
import os
import logging
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from main import BlockchainFederatedIncentiveSystem, EnhancedSystemConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_6_clients_system():
    """Test the federated learning system with 6 clients"""
    
    print("🧪 Testing Federated Learning System with 6 Clients")
    print("=" * 60)
    
    try:
        # Get configuration
        config = get_config()
        print(f"📋 Configuration:")
        print(f"   num_clients: {config.num_clients}")
        print(f"   num_rounds: {config.num_rounds}")
        print(f"   enable_incentives: {config.enable_incentives}")
        print()
        
        # Create enhanced config
        enhanced_config = EnhancedSystemConfig(
            num_clients=config.num_clients,
            num_rounds=config.num_rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            enable_incentives=config.enable_incentives,
        )
        
        print(f"📋 Enhanced Config:")
        print(f"   num_clients: {enhanced_config.num_clients}")
        print(f"   num_rounds: {enhanced_config.num_rounds}")
        print()
        
        # Create system
        print("🏗️  Creating system...")
        system = BlockchainFederatedIncentiveSystem(enhanced_config)
        
        # Initialize system
        print("🔧 Initializing system...")
        success = system.initialize_system()
        
        if not success:
            print("❌ System initialization failed")
            return False
        
        print("✅ System initialized successfully")
        print()
        
        # Check coordinator client count
        if hasattr(system, 'coordinator') and system.coordinator:
            print(f"📊 Coordinator client count: {len(system.coordinator.clients)}")
            print(f"📊 Expected client count: {enhanced_config.num_clients}")
            
            if len(system.coordinator.clients) == enhanced_config.num_clients:
                print("✅ Coordinator has correct number of clients")
            else:
                print(f"❌ Coordinator has {len(system.coordinator.clients)} clients, expected {enhanced_config.num_clients}")
                return False
        else:
            print("❌ No coordinator found")
            return False
        
        print()
        
        # Run federated training
        print("🚀 Running federated training...")
        training_success = system.run_federated_training_with_incentives()
        
        if not training_success:
            print("❌ Federated training failed")
            return False
        
        print("✅ Federated training completed successfully")
        print()
        
        # Check incentive history
        print("📊 Checking incentive history...")
        if hasattr(system, 'incentive_history') and system.incentive_history:
            print(f"   Incentive history records: {len(system.incentive_history)}")
            
            # Count unique clients in incentive history
            all_clients = set()
            for record in system.incentive_history:
                if 'individual_rewards' in record:
                    all_clients.update(record['individual_rewards'].keys())
            
            print(f"   Unique clients in incentive history: {len(all_clients)}")
            print(f"   Client IDs: {sorted(all_clients)}")
            
            if len(all_clients) == enhanced_config.num_clients:
                print("✅ All clients participated in incentives")
            else:
                print(f"❌ Only {len(all_clients)} clients participated, expected {enhanced_config.num_clients}")
                return False
        else:
            print("❌ No incentive history found")
            return False
        
        print()
        
        # Test incentive summary
        print("📊 Testing incentive summary...")
        incentive_summary = system.get_incentive_summary()
        
        print(f"   Total Rounds: {incentive_summary['total_rounds']}")
        print(f"   Total Rewards: {incentive_summary['total_rewards_distributed']}")
        print(f"   Participant Rewards: {len(incentive_summary['participant_rewards'])} clients")
        
        # Check if we have rewards for all clients
        expected_clients = set(f"client_{i+1}" for i in range(enhanced_config.num_clients))
        actual_clients = set(incentive_summary['participant_rewards'].keys())
        
        print(f"   Expected clients: {sorted(expected_clients)}")
        print(f"   Actual clients: {sorted(actual_clients)}")
        
        if expected_clients == actual_clients:
            print("✅ All configured clients have rewards!")
        else:
            missing_clients = expected_clients - actual_clients
            extra_clients = actual_clients - expected_clients
            if missing_clients:
                print(f"❌ Missing clients: {sorted(missing_clients)}")
            if extra_clients:
                print(f"⚠️ Extra clients: {sorted(extra_clients)}")
            return False
        
        print()
        
        # Test visualization
        print("🎨 Testing token distribution visualization...")
        try:
            from visualization.performance_visualization import PerformanceVisualizer
            visualizer = PerformanceVisualizer(output_dir=".", attack_name="Test6Clients")
            
            # Generate token distribution plot
            plot_path = visualizer.plot_token_distribution(incentive_summary, save=True)
            if plot_path:
                print(f"✅ Token distribution plot generated: {plot_path}")
                print("🎯 Token distribution visualization is working correctly!")
                print(f"📊 Visualization shows {len(incentive_summary['participant_rewards'])} clients")
                
                # Verify the plot shows all clients
                if len(incentive_summary['participant_rewards']) == enhanced_config.num_clients:
                    print("✅ Visualization shows all configured clients!")
                    return True
                else:
                    print(f"❌ Visualization shows {len(incentive_summary['participant_rewards'])} clients, expected {enhanced_config.num_clients}")
                    return False
            else:
                print("❌ Failed to generate token distribution plot")
                return False
                
        except Exception as e:
            print(f"❌ Error generating visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Error testing 6 clients system: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_6_clients_system()
    if success:
        print("\n🎉 6-client system test passed!")
        print("✅ Token distribution visualization correctly shows all 6 clients")
    else:
        print("\n❌ 6-client system test failed")
        print("⚠️ Some clients are still missing from the visualization")
