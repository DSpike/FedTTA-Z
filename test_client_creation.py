#!/usr/bin/env python3
"""
Test script to verify client creation
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from main import EnhancedSystemConfig, BlockchainFederatedIncentiveSystem

def test_client_creation():
    """Test client creation with 6 clients"""
    
    print("🧪 Testing Client Creation")
    print("=" * 50)
    
    # Get centralized config
    central_config = get_config()
    print(f"📋 Centralized Config:")
    print(f"   num_clients: {central_config.num_clients}")
    print()
    
    # Create enhanced config
    enhanced_config = EnhancedSystemConfig(
        num_clients=central_config.num_clients,
        num_rounds=central_config.num_rounds,
        local_epochs=central_config.local_epochs,
        learning_rate=central_config.learning_rate,
        enable_incentives=central_config.enable_incentives,
    )
    
    print(f"📋 Enhanced Config:")
    print(f"   num_clients: {enhanced_config.num_clients}")
    print()
    
    # Create system
    print("🏗️  Creating system...")
    system = BlockchainFederatedIncentiveSystem(enhanced_config)
    
    # Initialize system
    print("🔧 Initializing system...")
    success = system.initialize_system()
    
    if success:
        print("✅ System initialized successfully")
        
        # Check coordinator
        if hasattr(system, 'coordinator') and system.coordinator:
            print(f"✅ Coordinator created with {system.coordinator.num_clients} clients")
            print(f"✅ Coordinator has {len(system.coordinator.clients)} client objects")
            
            # List all clients
            print(f"📋 Client IDs:")
            for i, client in enumerate(system.coordinator.clients):
                print(f"   {i+1}. {client.client_id}")
        else:
            print("❌ No coordinator found")
        
        # Check authenticated clients
        if hasattr(system, 'authenticated_clients'):
            print(f"✅ Authenticated clients: {len(system.authenticated_clients)}")
            for client_id in system.authenticated_clients.keys():
                print(f"   - {client_id}")
        else:
            print("❌ No authenticated clients found")
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    test_client_creation()
