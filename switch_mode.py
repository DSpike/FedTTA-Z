#!/usr/bin/env python3
"""
Configuration Helper Script
Easily switch between Pure Federated Learning and Blockchain-Enabled modes
"""

import os
import sys

def update_config_file(mode):
    """Update config.py based on the selected mode"""
    
    config_path = "config.py"
    
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} not found!")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    if mode == "pure":
        # Pure federated learning mode
        content = content.replace(
            "enable_incentives: bool = True",
            "enable_incentives: bool = False"
        )
        content = content.replace(
            "enable_incentives: bool = False",
            "enable_incentives: bool = False"
        )
        content = content.replace(
            "enable_ipfs: bool = True",
            "enable_ipfs: bool = False"
        )
        content = content.replace(
            "enable_ipfs: bool = False",
            "enable_ipfs: bool = False"
        )
        content = content.replace(
            "enable_blockchain_recording: bool = True",
            "enable_blockchain_recording: bool = False"
        )
        content = content.replace(
            "enable_blockchain_recording: bool = False",
            "enable_blockchain_recording: bool = False"
        )
        
    elif mode == "blockchain":
        # Blockchain-enabled mode
        content = content.replace(
            "enable_incentives: bool = False",
            "enable_incentives: bool = True"
        )
        content = content.replace(
            "enable_incentives: bool = True",
            "enable_incentives: bool = True"
        )
        content = content.replace(
            "enable_ipfs: bool = False",
            "enable_ipfs: bool = True"
        )
        content = content.replace(
            "enable_ipfs: bool = True",
            "enable_ipfs: bool = True"
        )
        content = content.replace(
            "enable_blockchain_recording: bool = False",
            "enable_blockchain_recording: bool = True"
        )
        content = content.replace(
            "enable_blockchain_recording: bool = True",
            "enable_blockchain_recording: bool = True"
        )
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Main function to handle mode switching"""
    
    if len(sys.argv) != 2:
        print("Usage: python switch_mode.py <mode>")
        print("Modes:")
        print("  pure      - Pure federated learning (no blockchain)")
        print("  blockchain - Full blockchain-enabled federated learning")
        print("\nExample: python switch_mode.py pure")
        return
    
    mode = sys.argv[1].lower()
    
    if mode not in ["pure", "blockchain"]:
        print(f"❌ Error: Invalid mode '{mode}'")
        print("Valid modes: pure, blockchain")
        return
    
    print(f"🔄 Switching to {mode} federated learning mode...")
    
    if update_config_file(mode):
        print(f"✅ Successfully switched to {mode} mode!")
        print(f"\n📋 Current configuration:")
        
        if mode == "pure":
            print("   • Blockchain features: DISABLED")
            print("   • IPFS storage: DISABLED") 
            print("   • Incentive system: DISABLED")
            print("   • Pure federated learning: ENABLED")
        else:
            print("   • Blockchain features: ENABLED")
            print("   • IPFS storage: ENABLED")
            print("   • Incentive system: ENABLED")
            print("   • Full blockchain federated learning: ENABLED")
        
        print(f"\n🚀 You can now run: python main.py")
        
    else:
        print("❌ Failed to update configuration")

if __name__ == "__main__":
    main()

