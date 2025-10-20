#!/usr/bin/env python3
"""
Simple ERC20 Token Deployment for Federated Learning System
"""

import json
import time
from web3 import Web3
from solcx import compile_source, set_solc_version

# Set Solidity compiler version
set_solc_version('0.8.19')

# Ganache connection
GANACHE_URL = "http://localhost:8545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

def main():
    print("🚀 Deploying ERC20 Token Contract...")
    
    # Check connection
    if not w3.is_connected():
        print("❌ Failed to connect to Ganache. Make sure it's running on port 8545")
        return
    
    print(f"✅ Connected to Ganache at {GANACHE_URL}")
    print(f"📊 Network ID: {w3.eth.chain_id}")
    print(f"💰 Account balance: {w3.eth.get_balance(w3.eth.accounts[0]) / 10**18:.2f} ETH")
    
    try:
        # Read ERC20 Token contract
        with open('contracts/FederatedLearningToken.sol', 'r') as f:
            token_source = f.read()
        
        # Compile ERC20 Token contract (ignore documentation warnings)
        print("\n📝 Compiling ERC20 Token contract...")
        compiled_sol = compile_source(
            token_source, 
            output_values=['abi', 'bin'],
            solc_version='0.8.19',
            allow_paths='.',
            import_remappings=[]
        )
        
        token_interface = compiled_sol['<stdin>:FederatedLearningToken']
        
        # Deploy ERC20 Token contract
        print("🚀 Deploying ERC20 Token contract...")
        account = w3.eth.accounts[0]
        initial_supply = 10000000  # 10 million tokens (increased for more clients)
        
        # Build contract
        contract = w3.eth.contract(
            abi=token_interface['abi'],
            bytecode=token_interface['bin']
        )
        
        # Build constructor transaction
        constructor_tx = contract.constructor(initial_supply).build_transaction({
            'from': account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(account)
        })
        
        # Deploy contract
        tx_hash = w3.eth.send_transaction(constructor_tx)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        token_address = tx_receipt.contractAddress
        
        print(f"✅ ERC20 Token deployed at: {token_address}")
        
        # Save deployment info
        deployment_info = {
            "network": "ganache",
            "chain_id": w3.eth.chain_id,
            "deployment_time": time.time(),
            "contracts": {
                "FederatedLearningToken": {
                    "address": token_address,
                    "abi": token_interface['abi']
                }
            },
            "accounts": {
                "deployer": account
            },
            "token_info": {
                "name": "Federated Learning Token",
                "symbol": "FLT",
                "decimals": 18,
                "initial_supply": initial_supply
            }
        }
        
        with open('erc20_deployment.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"\n📄 Deployment info saved to: erc20_deployment.json")
        
        # Display summary
        print("\n🎉 ERC20 TOKEN DEPLOYMENT COMPLETE!")
        print("=" * 50)
        print(f"Token Address: {token_address}")
        print(f"Initial Supply: {initial_supply:,} FLT")
        print(f"Deployer Account: {account}")
        print("=" * 50)
        
        # Test token balance
        token_contract = w3.eth.contract(address=token_address, abi=token_interface['abi'])
        token_balance = token_contract.functions.balanceOf(account).call()
        print(f"💰 Deployer token balance: {token_balance / 10**18:,.2f} FLT")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

