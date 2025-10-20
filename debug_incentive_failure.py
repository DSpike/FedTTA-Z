#!/usr/bin/env python3
"""
Debug script to investigate incentive contract initialization failure
"""

import sys
import os
import logging
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blockchain.blockchain_incentive_contract import BlockchainIncentiveContract
from blockchain.blockchain_incentive_contract import BlockchainIncentiveManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_incentive_contract_initialization():
    """Debug the incentive contract initialization process"""
    
    print("🔍 Debugging Incentive Contract Initialization")
    print("=" * 60)
    
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
    
    # Step 1: Check if deployed_contracts.json exists and has the right structure
    print("📄 Step 1: Checking deployed_contracts.json")
    try:
        with open('deployed_contracts.json', 'r') as f:
            deployed_contracts = json.load(f)
        
        if 'contracts' in deployed_contracts and 'incentive_contract' in deployed_contracts['contracts']:
            incentive_contract_info = deployed_contracts['contracts']['incentive_contract']
            print(f"   ✅ Found incentive_contract section")
            print(f"   Address in file: {incentive_contract_info.get('address', 'NOT_FOUND')}")
            print(f"   ABI length: {len(incentive_contract_info.get('abi', []))}")
            
            # Check if addresses match
            if incentive_contract_info.get('address') == incentive_contract_address:
                print(f"   ✅ Contract addresses match")
            else:
                print(f"   ❌ Contract addresses don't match!")
                print(f"      Config: {incentive_contract_address}")
                print(f"      File:   {incentive_contract_info.get('address')}")
        else:
            print(f"   ❌ incentive_contract section not found in deployed_contracts.json")
            return False
            
    except FileNotFoundError:
        print(f"   ❌ deployed_contracts.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON in deployed_contracts.json: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error reading deployed_contracts.json: {e}")
        return False
    
    print()
    
    # Step 2: Test Web3 connection
    print("🌐 Step 2: Testing Web3 connection")
    try:
        from web3 import Web3
        web3 = Web3(Web3.HTTPProvider(ethereum_rpc_url))
        
        # Test connection
        block_number = web3.eth.block_number
        print(f"   ✅ Connected to Ethereum network")
        print(f"   Current block number: {block_number}")
        print(f"   Is connected: {web3.is_connected()}")
        
        # Check if we can get accounts
        accounts = web3.eth.accounts
        print(f"   Available accounts: {len(accounts)}")
        if accounts:
            print(f"   First account: {accounts[0]}")
            
            # Check balance of first account
            balance = web3.eth.get_balance(accounts[0])
            print(f"   First account balance: {web3.from_wei(balance, 'ether')} ETH")
        
    except Exception as e:
        print(f"   ❌ Web3 connection failed: {e}")
        return False
    
    print()
    
    # Step 3: Test account loading
    print("🔑 Step 3: Testing account loading")
    try:
        from eth_account import Account
        account = Account.from_key(private_key)
        print(f"   ✅ Account loaded successfully")
        print(f"   Account address: {account.address}")
        
        # Check if this account has funds
        balance = web3.eth.get_balance(account.address)
        print(f"   Account balance: {web3.from_wei(balance, 'ether')} ETH")
        
        if balance == 0:
            print(f"   ⚠️  Account has no funds!")
        else:
            print(f"   ✅ Account has funds")
            
    except Exception as e:
        print(f"   ❌ Account loading failed: {e}")
        return False
    
    print()
    
    # Step 4: Test contract loading
    print("📜 Step 4: Testing contract loading")
    try:
        incentive_abi = deployed_contracts['contracts']['incentive_contract']['abi']
        print(f"   ✅ ABI loaded successfully")
        print(f"   ABI functions: {len([item for item in incentive_abi if item.get('type') == 'function'])}")
        
        # Try to create contract instance
        contract = web3.eth.contract(
            address=incentive_contract_address,
            abi=incentive_abi
        )
        print(f"   ✅ Contract instance created")
        
        # Try to call a simple function (if available)
        try:
            # Check if contract has any view functions
            view_functions = [item for item in incentive_abi if item.get('type') == 'function' and item.get('stateMutability') in ['view', 'pure']]
            if view_functions:
                print(f"   ✅ Contract has {len(view_functions)} view functions")
            else:
                print(f"   ⚠️  Contract has no view functions")
        except Exception as e:
            print(f"   ⚠️  Could not call contract functions: {e}")
            
    except Exception as e:
        print(f"   ❌ Contract loading failed: {e}")
        return False
    
    print()
    
    # Step 5: Test BlockchainIncentiveContract initialization
    print("🏗️  Step 5: Testing BlockchainIncentiveContract initialization")
    try:
        incentive_contract = BlockchainIncentiveContract(
            rpc_url=ethereum_rpc_url,
            contract_address=incentive_contract_address,
            contract_abi=incentive_abi,
            private_key=private_key,
            aggregator_address=aggregator_address
        )
        print(f"   ✅ BlockchainIncentiveContract initialized successfully")
        print(f"   Contract address: {incentive_contract.contract_address}")
        print(f"   Account address: {incentive_contract.account.address}")
        
    except Exception as e:
        print(f"   ❌ BlockchainIncentiveContract initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Step 6: Test BlockchainIncentiveManager initialization
    print("👨‍💼 Step 6: Testing BlockchainIncentiveManager initialization")
    try:
        incentive_manager = BlockchainIncentiveManager(incentive_contract)
        print(f"   ✅ BlockchainIncentiveManager initialized successfully")
        
    except Exception as e:
        print(f"   ❌ BlockchainIncentiveManager initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("🎉 All tests passed! Incentive system should work correctly.")
    return True

if __name__ == "__main__":
    success = debug_incentive_contract_initialization()
    if success:
        print("\n✅ Debug completed successfully")
    else:
        print("\n❌ Debug found issues")
