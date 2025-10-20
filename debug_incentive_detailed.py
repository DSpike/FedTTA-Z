#!/usr/bin/env python3
"""
Detailed debug script to investigate incentive contract initialization failure
"""

import sys
import os
import logging
import json
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_incentive_initialization_detailed():
    """Debug the incentive contract initialization with detailed error reporting"""
    
    print("🔍 Detailed Debugging of Incentive Contract Initialization")
    print("=" * 70)
    
    # Configuration values from EnhancedSystemConfig
    ethereum_rpc_url = "http://localhost:8545"
    incentive_contract_address = "0x02090bbB57546b0bb224880a3b93D2Ffb0dde144"
    private_key = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"
    aggregator_address = "0x4565f36D8E3cBC1c7187ea39Eb613E484411e075"
    
    print(f"📋 Configuration Values:")
    print(f"   RPC URL: {ethereum_rpc_url}")
    print(f"   Contract Address: {incentive_contract_address}")
    print(f"   Private Key: {private_key[:10]}...{private_key[-10:]}")
    print(f"   Aggregator Address: {aggregator_address}")
    print()
    
    # Step 1: Test the exact same initialization as in main.py
    print("🏗️  Step 1: Testing exact main.py initialization sequence")
    try:
        # Load deployed contracts
        with open('deployed_contracts.json', 'r') as f:
            deployed_contracts = json.load(f)
        incentive_abi = deployed_contracts['contracts']['incentive_contract']['abi']
        print(f"   ✅ ABI loaded: {len(incentive_abi)} items")
        
        # Import the exact classes used in main.py
        from blockchain.blockchain_incentive_contract import BlockchainIncentiveContract
        from blockchain.blockchain_incentive_contract import BlockchainIncentiveManager
        
        print(f"   ✅ Classes imported successfully")
        
        # Create incentive contract (exact same as main.py)
        print(f"   🔄 Creating BlockchainIncentiveContract...")
        incentive_contract = BlockchainIncentiveContract(
            rpc_url=ethereum_rpc_url,
            contract_address=incentive_contract_address,
            contract_abi=incentive_abi,
            private_key=private_key,
            aggregator_address=aggregator_address
        )
        print(f"   ✅ BlockchainIncentiveContract created successfully")
        
        # Create incentive manager (exact same as main.py)
        print(f"   🔄 Creating BlockchainIncentiveManager...")
        incentive_manager = BlockchainIncentiveManager(incentive_contract)
        print(f"   ✅ BlockchainIncentiveManager created successfully")
        
        # Test if incentive manager has required methods
        print(f"   🔍 Testing incentive manager methods...")
        required_methods = ['process_round_contributions', 'distribute_rewards']
        for method in required_methods:
            if hasattr(incentive_manager, method):
                print(f"      ✅ Has {method}")
            else:
                print(f"      ❌ Missing {method}")
        
        print(f"   🎉 All tests passed! Incentive system should work.")
        return True
        
    except Exception as e:
        print(f"   ❌ Error during initialization: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Full traceback:")
        traceback.print_exc()
        return False

def test_incentive_processing():
    """Test if incentive processing would work with the created manager"""
    
    print("\n🧪 Step 2: Testing incentive processing simulation")
    try:
        # Create the incentive system components
        from blockchain.blockchain_incentive_contract import BlockchainIncentiveContract
        from blockchain.blockchain_incentive_contract import BlockchainIncentiveManager
        
        with open('deployed_contracts.json', 'r') as f:
            deployed_contracts = json.load(f)
        incentive_abi = deployed_contracts['contracts']['incentive_contract']['abi']
        
        incentive_contract = BlockchainIncentiveContract(
            rpc_url="http://localhost:8545",
            contract_address="0x02090bbB57546b0bb224880a3b93D2Ffb0dde144",
            contract_abi=incentive_abi,
            private_key="0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
            aggregator_address="0x4565f36D8E3cBC1c7187ea39Eb613E484411e075"
        )
        
        incentive_manager = BlockchainIncentiveManager(incentive_contract)
        
        # Test contribution processing
        print(f"   🔄 Testing contribution processing...")
        mock_contributions = [
            {
                'client_address': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
                'model_parameters': {'weight': [1, 2, 3, 4, 5]},
                'previous_accuracy': 0.85,
                'current_accuracy': 0.90,
                'data_quality': 0.9,
                'reliability': 0.85
            }
        ]
        
        shapley_values = {'0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6': 0.1}
        
        # Try to process contributions
        reward_distributions = incentive_manager.process_round_contributions(
            round_number=1,
            client_contributions=mock_contributions,
            shapley_values=shapley_values
        )
        
        if reward_distributions:
            print(f"   ✅ Contribution processing successful: {len(reward_distributions)} rewards")
            for rd in reward_distributions:
                print(f"      - {rd.recipient_address}: {rd.token_amount} tokens")
        else:
            print(f"   ⚠️  No reward distributions generated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error during incentive processing test: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Detailed Incentive Debug")
    print("=" * 50)
    
    # Test initialization
    init_success = debug_incentive_initialization_detailed()
    
    if init_success:
        # Test processing
        process_success = test_incentive_processing()
        
        if process_success:
            print("\n🎉 All tests passed! The incentive system should work correctly.")
        else:
            print("\n⚠️  Initialization works but processing has issues.")
    else:
        print("\n❌ Initialization failed - this is the root cause.")
    
    print("\n" + "=" * 50)
    print("Debug completed.")
