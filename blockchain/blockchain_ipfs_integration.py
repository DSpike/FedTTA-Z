#!/usr/bin/env python3
"""
Blockchain and IPFS Integration for Federated Learning
Implements smart contracts for model storage and provenance tracking
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from web3 import Web3
from eth_account import Account
import requests
import base58
# import ipfshttpclient  # Replaced with HTTP API
from concurrent.futures import ThreadPoolExecutor
import threading
from .real_gas_collector import real_gas_collector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for IPFS storage"""
    client_id: str
    round_number: int
    model_type: str
    parameters_count: int
    training_loss: float
    validation_accuracy: float
    timestamp: float
    model_hash: str
    ipfs_cid: Optional[str] = None
    blockchain_tx_hash: Optional[str] = None

@dataclass
class AggregationMetadata:
    """Aggregation metadata for IPFS storage"""
    round_number: int
    num_clients: int
    total_samples: int
    aggregation_time: float
    client_contributions: Dict[str, float]
    model_hash: str
    timestamp: float
    ipfs_cid: Optional[str] = None
    blockchain_tx_hash: Optional[str] = None

class IPFSClient:
    """
    IPFS client for storing and retrieving model data
    """
    
    def __init__(self, ipfs_url: str = "http://localhost:5001"):
        """
        Initialize IPFS client using HTTP API
        
        Args:
            ipfs_url: IPFS API URL
        """
        self.ipfs_url = ipfs_url
        self.connected = False
        
        try:
            # Test connection using HTTP API
            response = requests.post(f'{ipfs_url}/api/v0/version', timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                self.connected = True
                logger.info(f"IPFS client connected to {ipfs_url} (version: {version_info.get('Version', 'Unknown')})")
            else:
                logger.warning(f"Failed to connect to IPFS: HTTP {response.status_code}")
                self.connected = False
        except Exception as e:
            logger.warning(f"Failed to connect to IPFS: {str(e)}")
            self.connected = False
    
    def add_data(self, data: Dict) -> Optional[str]:
        """
        Add data to IPFS
        
        Args:
            data: Data dictionary to store
            
        Returns:
            cid: IPFS content identifier
        """
        if not self.connected:
            logger.warning("IPFS not connected, cannot store data")
            return None
        
        try:
            # For large model data, use compression to reduce memory usage
            import pickle
            import gzip
            
            # Use pickle for better serialization of PyTorch tensors
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress data to reduce size and memory usage
            compressed_data = gzip.compress(serialized_data)
            
            # Add to IPFS using HTTP API with compressed data
            files = {'file': ('model_data.pkl.gz', compressed_data)}
            response = requests.post(f'{self.ipfs_url}/api/v0/add', files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                cid = result['Hash']
                logger.info(f"Data stored on IPFS: {cid}")
                return cid
            else:
                logger.error(f"Failed to add data to IPFS: HTTP {response.status_code}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to store data on IPFS: {str(e)}")
            return None
    
    def get_data(self, cid: str) -> Optional[Dict]:
        """
        Retrieve data from IPFS
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            data: Retrieved data dictionary
        """
        if not self.connected:
            logger.warning("IPFS not connected, cannot retrieve data")
            return None
        
        try:
            # Get data from IPFS using HTTP API
            response = requests.post(f'{self.ipfs_url}/api/v0/cat?arg={cid}', timeout=60)
            
            if response.status_code == 200:
                # Check if data is compressed
                if response.content.startswith(b'\x1f\x8b'):  # gzip magic number
                    # Decompress and deserialize
                    import gzip
                    import pickle
                    decompressed_data = gzip.decompress(response.content)
                    data = pickle.loads(decompressed_data)
                else:
                    # Legacy JSON format
                    json_data = response.text
                    data = json.loads(json_data)
                
                logger.info(f"Data retrieved from IPFS: {cid}")
                return data
            else:
                logger.error(f"Failed to retrieve data from IPFS: HTTP {response.status_code}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data from IPFS: {str(e)}")
            return None
    
    def pin_data(self, cid: str) -> bool:
        """
        Pin data to IPFS to prevent garbage collection
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            success: Whether pinning was successful
        """
        if not self.connected:
            logger.warning("IPFS not connected, cannot pin data")
            return False
        
        try:
            # Pin data using HTTP API
            response = requests.post(f'{self.ipfs_url}/api/v0/pin/add?arg={cid}', timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Data pinned on IPFS: {cid}")
                return True
            else:
                logger.error(f"Failed to pin data on IPFS: HTTP {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to pin data on IPFS: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test IPFS connection"""
        if not self.connected:
            return False
        
        try:
            # Try to get IPFS version using HTTP API
            response = requests.post(f'{self.ipfs_url}/api/v0/version', timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"IPFS version: {version_info.get('Version', 'Unknown')}")
                return True
            else:
                logger.error(f"IPFS connection test failed: HTTP {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"IPFS connection test failed: {str(e)}")
            return False

class EthereumClient:
    """
    Ethereum client for smart contract interactions
    """
    
    def __init__(self, rpc_url: str, private_key: str, contract_address: str, contract_abi: List[Dict]):
        """
        Initialize Ethereum client
        
        Args:
            rpc_url: Ethereum RPC URL
            private_key: Private key for transactions
            contract_address: Smart contract address
            contract_abi: Smart contract ABI
        """
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.contract_address = contract_address
        self.contract_abi = contract_abi
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add middleware for PoA networks
        try:
            from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware
            self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:
            logger.warning("ExtraDataToPOAMiddleware not available")
        
        # Check connection
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum network: {rpc_url}")
        
        # Load account
        self.account = Account.from_key(private_key)
        self.funded_account = None
        
        # Check if account has funds, if not use first Ganache account
        balance = self.web3.eth.get_balance(self.account.address)
        if balance == 0 and len(self.web3.eth.accounts) > 0:
            logger.info(f"Account {self.account.address} has no funds, using first Ganache account")
            # Use first Ganache account directly
            first_account_address = self.web3.eth.accounts[0]
            logger.info(f"Using first Ganache account: {first_account_address}")
            # Set the default account to the first Ganache account
            self.web3.eth.default_account = first_account_address
            # Store the first account address separately
            self.funded_account = first_account_address
        else:
            self.web3.eth.default_account = self.account.address
        
        # Load contract
        if contract_address and contract_address != '0x' + '0' * 40:
            # Load deployed contract ABI from file
            try:
                with open('deployed_contracts.json', 'r') as f:
                    deployment_info = json.load(f)
                    deployed_abi = deployment_info['contracts']['federated_learning']['abi']
                self.contract = self.web3.eth.contract(
                    address=contract_address,
                    abi=deployed_abi
                )
                logger.info(f"Contract initialized with deployed ABI: {contract_address}")
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load deployed ABI: {e}, using default ABI")
                self.contract = self.web3.eth.contract(
                    address=contract_address,
                    abi=contract_abi
                )
                logger.info(f"Contract initialized with default ABI: {contract_address}")
            
        else:
            logger.warning("No contract address provided, blockchain features disabled")
            logger.info("To enable blockchain features, deploy contracts using: python deploy_minimal_contracts.py")
            self.contract = None
        
        logger.info(f"Ethereum client connected to {rpc_url}")
        logger.info(f"Account: {self.account.address}")
        logger.info(f"Contract: {contract_address}")
    
    def submit_model_update(self, model_hash: str, ipfs_cid: str, round_number: int) -> Optional[str]:
        """
        Submit model update to smart contract
        
        Args:
            model_hash: SHA256 hash of model parameters
            ipfs_cid: IPFS content identifier
            round_number: Current round number
            
        Returns:
            tx_hash: Transaction hash
        """
        if not self.contract:
            logger.warning("Contract not available, cannot submit model update")
            return None
        
        try:
            # Check if submitModelUpdate function exists in contract
            if not hasattr(self.contract.functions, 'submitModelUpdate'):
                logger.warning("submitModelUpdate function not available in contract")
                return None
            
            # Convert model hash and IPFS CID to bytes32 format
            model_hash_bytes = self._hash_to_bytes32(model_hash)
            cid_bytes = self._cid_to_bytes32(ipfs_cid)
            
            # Use the funded account for transactions
            funded_account = self.funded_account if hasattr(self, 'funded_account') and self.funded_account else self.web3.eth.default_account
            
            # Build transaction
            tx = self.contract.functions.submitModelUpdate(
                model_hash_bytes,
                cid_bytes,
                round_number
            ).build_transaction({
                'from': funded_account,
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(funded_account)
            })
            
            # For Ganache, we can send unsigned transactions since accounts are unlocked
            tx_hash = self.web3.eth.send_transaction(tx)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)  # 30 second timeout
            
            if receipt.status == 1:
                # Extract real gas usage data
                actual_gas_used = receipt.gasUsed
                block_number = receipt.blockNumber
                gas_price = getattr(receipt, 'effectiveGasPrice', tx.get('gasPrice', 20000000000))
                gas_limit = tx['gas']
                
                # Record real gas usage
                real_gas_collector.record_transaction(
                    tx_hash=tx_hash.hex(),
                    tx_type="Model Update",
                    gas_used=actual_gas_used,
                    gas_limit=gas_limit,
                    gas_price=gas_price,
                    block_number=block_number,
                    round_number=round_number,
                    ipfs_cid=ipfs_cid
                )
                
                logger.info(f"Model update submitted: {tx_hash.hex()}")
                logger.info(f"Real gas used: {actual_gas_used}, Block: {block_number}")
                return tx_hash.hex()
            else:
                logger.error("Transaction failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to submit model update: {str(e)}")
            return None
    
    def submit_aggregation(self, round_number: int, aggregated_hash: str, ipfs_cid: str) -> Optional[str]:
        """
        Submit aggregation result to smart contract
        
        Args:
            round_number: Round number
            aggregated_hash: Hash of aggregated model
            ipfs_cid: IPFS content identifier
            
        Returns:
            tx_hash: Transaction hash
        """
        try:
            # Convert IPFS CID to bytes32 format
            cid_bytes = self._cid_to_bytes32(ipfs_cid)
            
            # Build transaction
            tx = self.contract.functions.submitAggregation(
                round_number,
                aggregated_hash,
                cid_bytes
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)  # 30 second timeout
            
            if receipt.status == 1:
                # Extract real gas usage data
                actual_gas_used = receipt.gasUsed
                block_number = receipt.blockNumber
                gas_price = getattr(receipt, 'effectiveGasPrice', tx.get('gasPrice', 20000000000))
                gas_limit = tx['gas']
                
                # Record real gas usage
                real_gas_collector.record_transaction(
                    tx_hash=tx_hash.hex(),
                    tx_type="Model Aggregation",
                    gas_used=actual_gas_used,
                    gas_limit=gas_limit,
                    gas_price=gas_price,
                    block_number=block_number,
                    round_number=round_number,
                    ipfs_cid=ipfs_cid
                )
                
                logger.info(f"Aggregation submitted: {tx_hash.hex()}")
                logger.info(f"Real gas used: {actual_gas_used}, Block: {block_number}")
                return tx_hash.hex()
            else:
                logger.error("Transaction failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to submit aggregation: {str(e)}")
            return None
    
    def get_model_update(self, round_number: int, client_id: str) -> Optional[Dict]:
        """
        Get model update from smart contract
        
        Args:
            round_number: Round number
            client_id: Client identifier
            
        Returns:
            model_data: Model update data
        """
        try:
            # Call smart contract function
            result = self.contract.functions.getModelUpdate(round_number, client_id).call()
            
            if result:
                model_data = {
                    'model_hash': result[0],
                    'ipfs_cid': self._bytes32_to_cid(result[1]),
                    'timestamp': result[2],
                    'round_number': result[3]
                }
                return model_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model update: {str(e)}")
            return None
    
    def get_aggregation(self, round_number: int) -> Optional[Dict]:
        """
        Get aggregation result from smart contract
        
        Args:
            round_number: Round number
            
        Returns:
            aggregation_data: Aggregation data
        """
        try:
            # Call smart contract function
            result = self.contract.functions.getAggregation(round_number).call()
            
            if result:
                aggregation_data = {
                    'aggregated_hash': result[0],
                    'ipfs_cid': self._bytes32_to_cid(result[1]),
                    'timestamp': result[2],
                    'round_number': result[3]
                }
                return aggregation_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get aggregation: {str(e)}")
            return None
    
    def submit_client_update(self, client_id: str, model_hash: str, ipfs_cid: str) -> Optional[str]:
        """
        Submit client model update to smart contract
        
        Args:
            client_id: Client identifier
            model_hash: SHA256 hash of model parameters
            ipfs_cid: IPFS content identifier
            
        Returns:
            tx_hash: Transaction hash
        """
        if not self.contract:
            logger.warning("Contract not available, cannot submit client update")
            return None
        
        try:
            # Check if submitModelUpdate function exists in contract
            if not hasattr(self.contract.functions, 'submitModelUpdate'):
                logger.warning("submitModelUpdate function not available in contract")
                return None
            
            # Convert model hash and IPFS CID to bytes32 format
            model_hash_bytes = self._hash_to_bytes32(model_hash)
            cid_bytes = self._cid_to_bytes32(ipfs_cid)
            
            # Build transaction (assuming we have a submitClientUpdate function)
            # For now, we'll use the same submitModelUpdate function
            # Use the funded account for transactions
            funded_account = self.funded_account if hasattr(self, 'funded_account') and self.funded_account else self.web3.eth.default_account
            
            tx = self.contract.functions.submitModelUpdate(
                model_hash_bytes,
                cid_bytes,
                0  # Round number for client updates
            ).build_transaction({
                'from': funded_account,
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(funded_account)
            })
            
            # For Ganache, we can send unsigned transactions since accounts are unlocked
            tx_hash = self.web3.eth.send_transaction(tx)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)  # 30 second timeout
            
            if receipt.status == 1:
                # Extract real gas usage data
                actual_gas_used = receipt.gasUsed
                block_number = receipt.blockNumber
                gas_price = getattr(receipt, 'effectiveGasPrice', tx.get('gasPrice', 20000000000))
                gas_limit = tx['gas']
                
                # Record real gas usage
                real_gas_collector.record_transaction(
                    tx_hash=tx_hash.hex(),
                    tx_type="Client Update",
                    gas_used=actual_gas_used,
                    gas_limit=gas_limit,
                    gas_price=gas_price,
                    block_number=block_number,
                    round_number=0,  # Client updates don't have round numbers
                    client_id=client_id,
                    ipfs_cid=ipfs_cid
                )
                
                logger.info(f"Client {client_id} update submitted to blockchain: {tx_hash.hex()}")
                logger.info(f"Real gas used: {actual_gas_used}, Block: {block_number}")
                return tx_hash.hex()
            else:
                logger.error(f"Client {client_id} transaction failed")
                return None
            
        except Exception as e:
            logger.error(f"Failed to submit client update: {str(e)}")
            return None
    
    def _cid_to_bytes32(self, cid: str) -> bytes:
        """Convert IPFS CID to bytes32 format"""
        try:
            # Decode base58 CID to bytes
            cid_bytes = base58.b58decode(cid)
            
            # Pad to 32 bytes
            if len(cid_bytes) < 32:
                cid_bytes = cid_bytes + b'\x00' * (32 - len(cid_bytes))
            elif len(cid_bytes) > 32:
                cid_bytes = cid_bytes[:32]
            
            return cid_bytes
            
        except Exception as e:
            logger.error(f"Failed to convert CID to bytes32: {str(e)}")
            # Return zero bytes as fallback
            return b'\x00' * 32
    
    def _hash_to_bytes32(self, hash_str: str) -> bytes:
        """Convert hash string to bytes32 format"""
        try:
            # If it's already a hex string, convert directly
            if hash_str.startswith('0x'):
                hash_str = hash_str[2:]
            
            # Check if it's a valid hex string
            try:
                hash_bytes = bytes.fromhex(hash_str)
            except ValueError:
                # If not hex, create a hash from the string
                import hashlib
                hash_bytes = hashlib.sha256(hash_str.encode()).digest()
            
            # Pad to 32 bytes
            if len(hash_bytes) < 32:
                hash_bytes = hash_bytes + b'\x00' * (32 - len(hash_bytes))
            elif len(hash_bytes) > 32:
                hash_bytes = hash_bytes[:32]
            
            return hash_bytes
            
        except Exception as e:
            logger.error(f"Failed to convert hash to bytes32: {str(e)}")
            # Return zero bytes as fallback
            return b'\x00' * 32
    
    def _bytes32_to_cid(self, cid_bytes: bytes) -> str:
        """Convert bytes32 to IPFS CID"""
        try:
            # Remove padding zeros
            cid_bytes = cid_bytes.rstrip(b'\x00')
            
            # Encode to base58
            cid = base58.b58encode(cid_bytes).decode('utf-8')
            
            return cid
            
        except Exception as e:
            logger.error(f"Failed to convert bytes32 to CID: {str(e)}")
            return ""

class BlockchainIPFSIntegration:
    """
    Main integration class for blockchain and IPFS
    """
    
    def __init__(self, ethereum_config: Dict, ipfs_config: Dict):
        """
        Initialize blockchain and IPFS integration
        
        Args:
            ethereum_config: Ethereum configuration
            ipfs_config: IPFS configuration
        """
        self.ethereum_config = ethereum_config
        self.ipfs_config = ipfs_config
        
        # Initialize clients
        self.ipfs_client = IPFSClient(ipfs_config['url'])
        self.ethereum_client = EthereumClient(
            ethereum_config['rpc_url'],
            ethereum_config['private_key'],
            ethereum_config['contract_address'],
            ethereum_config['contract_abi']
        )
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        logger.info("Blockchain and IPFS integration initialized")
    
    def store_model_with_provenance(self, model_data: Dict, metadata: ModelMetadata) -> Tuple[Optional[str], Optional[str]]:
        """
        Store model on IPFS and record on blockchain
        
        Args:
            model_data: Model parameters and data
            metadata: Model metadata
            
        Returns:
            ipfs_cid: IPFS content identifier
            tx_hash: Blockchain transaction hash
        """
        logger.info(f"Storing model for client {metadata.client_id}, round {metadata.round_number}")
        
        # Store on IPFS
        ipfs_cid = self.ipfs_client.add_data(model_data)
        if ipfs_cid is None:
            logger.error("Failed to store model on IPFS")
            return None, None
        
        # Pin data to prevent garbage collection
        self.ipfs_client.pin_data(ipfs_cid)
        
        # Record on blockchain
        tx_hash = self.ethereum_client.submit_model_update(
            metadata.model_hash,
            ipfs_cid,
            metadata.round_number
        )
        
        if tx_hash is None:
            logger.error("Failed to record model on blockchain")
            return ipfs_cid, None
        
        logger.info(f"Model stored successfully - IPFS: {ipfs_cid}, Blockchain: {tx_hash}")
        return ipfs_cid, tx_hash
    
    def store_aggregation_with_provenance(self, aggregation_data: Dict, metadata: AggregationMetadata) -> Tuple[Optional[str], Optional[str]]:
        """
        Store aggregation result on IPFS and record on blockchain
        
        Args:
            aggregation_data: Aggregated model data
            metadata: Aggregation metadata
            
        Returns:
            ipfs_cid: IPFS content identifier
            tx_hash: Blockchain transaction hash
        """
        logger.info(f"Storing aggregation for round {metadata.round_number}")
        
        # Store on IPFS
        ipfs_cid = self.ipfs_client.add_data(aggregation_data)
        if ipfs_cid is None:
            logger.error("Failed to store aggregation on IPFS")
            return None, None
        
        # Pin data to prevent garbage collection
        self.ipfs_client.pin_data(ipfs_cid)
        
        # Record on blockchain
        tx_hash = self.ethereum_client.submit_aggregation(
            metadata.round_number,
            metadata.model_hash,
            ipfs_cid
        )
        
        if tx_hash is None:
            logger.error("Failed to record aggregation on blockchain")
            return ipfs_cid, None
        
        logger.info(f"Aggregation stored successfully - IPFS: {ipfs_cid}, Blockchain: {tx_hash}")
        return ipfs_cid, tx_hash
    
    def retrieve_model(self, round_number: int, client_id: str) -> Optional[Dict]:
        """
        Retrieve model from blockchain and IPFS
        
        Args:
            round_number: Round number
            client_id: Client identifier
            
        Returns:
            model_data: Retrieved model data
        """
        logger.info(f"Retrieving model for client {client_id}, round {round_number}")
        
        # Get model metadata from blockchain
        model_metadata = self.ethereum_client.get_model_update(round_number, client_id)
        if model_metadata is None:
            logger.error("Model not found on blockchain")
            return None
        
        # Retrieve model data from IPFS
        model_data = self.ipfs_client.get_data(model_metadata['ipfs_cid'])
        if model_data is None:
            logger.error("Model data not found on IPFS")
            return None
        
        logger.info(f"Model retrieved successfully from IPFS: {model_metadata['ipfs_cid']}")
        return model_data
    
    def retrieve_aggregation(self, round_number: int) -> Optional[Dict]:
        """
        Retrieve aggregation result from blockchain and IPFS
        
        Args:
            round_number: Round number
            
        Returns:
            aggregation_data: Retrieved aggregation data
        """
        logger.info(f"Retrieving aggregation for round {round_number}")
        
        # Get aggregation metadata from blockchain
        aggregation_metadata = self.ethereum_client.get_aggregation(round_number)
        if aggregation_metadata is None:
            logger.error("Aggregation not found on blockchain")
            return None
        
        # Retrieve aggregation data from IPFS
        aggregation_data = self.ipfs_client.get_data(aggregation_metadata['ipfs_cid'])
        if aggregation_data is None:
            logger.error("Aggregation data not found on IPFS")
            return None
        
        logger.info(f"Aggregation retrieved successfully from IPFS: {aggregation_metadata['ipfs_cid']}")
        return aggregation_data
    
    def verify_model_integrity(self, model_data: Dict, expected_hash: str) -> bool:
        """
        Verify model integrity using hash
        
        Args:
            model_data: Model data
            expected_hash: Expected hash
            
        Returns:
            is_valid: Whether model is valid
        """
        try:
            # Compute hash of model data
            model_json = json.dumps(model_data, sort_keys=True, default=str)
            computed_hash = hashlib.sha256(model_json.encode()).hexdigest()
            
            is_valid = computed_hash == expected_hash
            
            if is_valid:
                logger.info("Model integrity verified")
            else:
                logger.error("Model integrity verification failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify model integrity: {str(e)}")
            return False
    
    def get_provenance_chain(self, round_number: int) -> List[Dict]:
        """
        Get provenance chain for a specific round
        
        Args:
            round_number: Round number
            
        Returns:
            provenance_chain: List of provenance records
        """
        logger.info(f"Getting provenance chain for round {round_number}")
        
        provenance_chain = []
        
        # Get aggregation record
        aggregation_metadata = self.ethereum_client.get_aggregation(round_number)
        if aggregation_metadata:
            provenance_chain.append({
                'type': 'aggregation',
                'round_number': round_number,
                'timestamp': aggregation_metadata['timestamp'],
                'ipfs_cid': aggregation_metadata['ipfs_cid'],
                'blockchain_tx': aggregation_metadata.get('tx_hash', 'N/A')
            })
        
        # Get client model updates (this would need to be implemented in the smart contract)
        # For now, we'll return the aggregation record
        
        logger.info(f"Provenance chain retrieved: {len(provenance_chain)} records")
        return provenance_chain
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("Blockchain and IPFS integration cleanup completed")

# Smart Contract ABI for Federated Learning
FEDERATED_LEARNING_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "ipfsCid", "type": "bytes32"},
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "name": "submitModelUpdate",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
            {"internalType": "bytes32", "name": "aggregatedHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "ipfsCid", "type": "bytes32"}
        ],
        "name": "submitAggregation",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"},
            {"internalType": "string", "name": "clientId", "type": "string"}
        ],
        "name": "getModelUpdate",
        "outputs": [
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "ipfsCid", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "name": "getAggregation",
        "outputs": [
            {"internalType": "bytes32", "name": "aggregatedHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "ipfsCid", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "participant", "type": "address"},
            {"internalType": "string", "name": "role", "type": "string"}
        ],
        "name": "registerParticipant",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "client", "type": "address"},
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "ipfsCid", "type": "bytes32"},
            {"internalType": "uint256", "name": "roundNumber", "type": "uint256"}
        ],
        "name": "submitClientUpdate",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

def main():
    """Test the blockchain and IPFS integration"""
    logger.info("Testing Blockchain and IPFS Integration")
    
    # Configuration - Using REAL Ganache blockchain
    ethereum_config = {
        'rpc_url': 'http://localhost:8545',
        'private_key': '0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d',  # Real Ganache private key
        'contract_address': '0x0000000000000000000000000000000000000000',  # Will be deployed to real address
        'contract_abi': FEDERATED_LEARNING_ABI
    }
    
    ipfs_config = {
        'url': 'http://localhost:5001'
    }
    
    try:
        # Initialize integration
        integration = BlockchainIPFSIntegration(ethereum_config, ipfs_config)
        
        # Test IPFS connection
        if integration.ipfs_client.test_connection():
            logger.info("✅ IPFS connection test passed")
        else:
            logger.warning("⚠️ IPFS connection test failed")
        
        # Test model storage
        test_model_data = {
            'parameters': {'weight': [1, 2, 3, 4, 5]},
            'metadata': {'test': True}
        }
        
        test_metadata = ModelMetadata(
            client_id='test_client',
            round_number=1,
            model_type='test_model',
            parameters_count=5,
            training_loss=0.1,
            validation_accuracy=0.95,
            timestamp=time.time(),
            model_hash='test_hash'
        )
        
        # Store model
        ipfs_cid, tx_hash = integration.store_model_with_provenance(test_model_data, test_metadata)
        
        if ipfs_cid:
            logger.info(f"✅ Model stored on IPFS: {ipfs_cid}")
        else:
            logger.warning("⚠️ Model storage failed")
        
        # Cleanup
        integration.cleanup()
        
        logger.info("✅ Blockchain and IPFS integration test completed!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")

if __name__ == "__main__":
    main()
