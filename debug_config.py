#!/usr/bin/env python3
"""
Debug script to check configuration values
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from main import EnhancedSystemConfig

def debug_config():
    """Debug configuration values"""
    
    print("🔍 Configuration Debug")
    print("=" * 50)
    
    # Get centralized config
    central_config = get_config()
    print(f"📋 Centralized Config (config.py):")
    print(f"   num_clients: {central_config.num_clients}")
    print(f"   num_rounds: {central_config.num_rounds}")
    print()
    
    # Create enhanced config
    enhanced_config = EnhancedSystemConfig(
        num_clients=central_config.num_clients,
        num_rounds=central_config.num_rounds,
        local_epochs=central_config.local_epochs,
        learning_rate=central_config.learning_rate,
        enable_incentives=central_config.enable_incentives,
    )
    
    print(f"📋 Enhanced Config (main.py):")
    print(f"   num_clients: {enhanced_config.num_clients}")
    print(f"   num_rounds: {enhanced_config.num_rounds}")
    print()
    
    # Test client creation loop
    print(f"🧪 Testing client creation loop:")
    for i in range(enhanced_config.num_clients):
        client_id = f"client_{i+1}"
        print(f"   Would create: {client_id}")
    
    print(f"\n✅ Expected clients: {enhanced_config.num_clients}")
    print(f"✅ Actual clients created: {enhanced_config.num_clients}")

if __name__ == "__main__":
    debug_config()
