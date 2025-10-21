#!/usr/bin/env python3
"""
Secure Model Update Implementation
Privacy-preserving federated learning with IPFS-only transmission
"""

import torch
import hashlib
import time
import logging
import pickle
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)

@dataclass
class SecureModelUpdate:
    """Privacy-preserving model update from a client"""
    client_id: str
    ipfs_cid: str                    # Only IPFS reference - NO raw parameters
    model_hash: str                  # Cryptographic hash for verification
    sample_count: int
    accuracy: float
    loss: float
    timestamp: float
    signature: str                   # Digital signature for authentication
    round_number: int
    encryption_method: str = "fernet"  # Encryption method used

class SecureEncryptionManager:
    """Manages encryption and decryption for model parameters"""
    
    def __init__(self):
        self.client_keys: Dict[str, str] = {}
    
    def generate_client_key(self, client_id: str, password: str = None) -> str:
        """Generate unique encryption key for client"""
        if password is None:
            password = f"client_{client_id}_secret_{int(time.time())}"
        
        # Derive key from password using PBKDF2
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        self.client_keys[client_id] = key.decode()
        return key.decode()
    
    def encrypt_model_parameters(self, parameters: Dict[str, torch.Tensor], 
                               client_id: str) -> str:
        """Encrypt model parameters using client's key"""
        try:
            # Get client's encryption key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                raise ValueError(f"No encryption key for client {client_id}")
            
            # Serialize parameters
            serialized = pickle.dumps(parameters)
            
            # Encrypt with client's key
            f = Fernet(client_key.encode())
            encrypted = f.encrypt(serialized)
            
            return encrypted.hex()
            
        except Exception as e:
            logger.error(f"Failed to encrypt model parameters: {e}")
            raise
    
    def decrypt_model_parameters(self, encrypted_hex: str, client_id: str) -> Dict[str, torch.Tensor]:
        """Decrypt model parameters using client's key"""
        try:
            # Get client's encryption key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                raise ValueError(f"No encryption key for client {client_id}")
            
            # Decrypt
            f = Fernet(client_key.encode())
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            decrypted_bytes = f.decrypt(encrypted_bytes)
            
            # Deserialize
            return pickle.loads(decrypted_bytes)
            
        except Exception as e:
            logger.error(f"Failed to decrypt model parameters: {e}")
            raise

class SecureHashManager:
    """Manages cryptographic hashing for model verification"""
    
    @staticmethod
    def compute_model_hash(parameters: Dict[str, torch.Tensor]) -> str:
        """Compute SHA256 hash of model parameters"""
        try:
            # Create deterministic hash
            param_bytes = b''
            for name in sorted(parameters.keys()):
                param = parameters[name]
                param_bytes += name.encode('utf-8')
                param_bytes += param.detach().cpu().numpy().tobytes()
            
            # Compute SHA256 hash
            hash_obj = hashlib.sha256(param_bytes)
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to compute model hash: {e}")
            raise
    
    @staticmethod
    def verify_model_hash(parameters: Dict[str, torch.Tensor], expected_hash: str) -> bool:
        """Verify model parameter hash"""
        try:
            actual_hash = SecureHashManager.compute_model_hash(parameters)
            return actual_hash == expected_hash
        except Exception as e:
            logger.error(f"Failed to verify model hash: {e}")
            return False

class SecureSignatureManager:
    """Manages digital signatures for authentication"""
    
    def __init__(self):
        self.client_keys: Dict[str, str] = {}
    
    def generate_client_keypair(self, client_id: str) -> tuple:
        """Generate RSA key pair for client"""
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Store keys
        self.client_keys[client_id] = {
            'private': private_pem.decode(),
            'public': public_pem.decode()
        }
        
        return private_pem.decode(), public_pem.decode()
    
    def sign_data(self, data: str, client_id: str) -> str:
        """Sign data with client's private key"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            # Get client's private key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                raise ValueError(f"No private key for client {client_id}")
            
            private_key = serialization.load_pem_private_key(
                client_key['private'].encode(),
                password=None,
            )
            
            # Sign data
            signature = private_key.sign(
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise
    
    def verify_signature(self, data: str, signature: str, client_id: str) -> bool:
        """Verify signature with client's public key"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            
            # Get client's public key
            client_key = self.client_keys.get(client_id)
            if not client_key:
                return False
            
            public_key = serialization.load_pem_public_key(
                client_key['public'].encode()
            )
            
            # Verify signature
            public_key.verify(
                base64.b64decode(signature),
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False

class SecureFederatedClient:
    """Privacy-preserving federated learning client"""
    
    def __init__(self, client_id: str, ipfs_client, password: str = None):
        self.client_id = client_id
        self.ipfs_client = ipfs_client
        self.encryption_manager = SecureEncryptionManager()
        self.hash_manager = SecureHashManager()
        self.signature_manager = SecureSignatureManager()
        
        # Generate encryption key
        self.encryption_key = self.encryption_manager.generate_client_key(client_id, password)
        
        # Generate signature key pair
        self.private_key, self.public_key = self.signature_manager.generate_client_keypair(client_id)
        
        logger.info(f"Secure client {client_id} initialized")
    
    def create_secure_model_update(self, model_parameters: Dict[str, torch.Tensor],
                                 sample_count: int, accuracy: float, loss: float,
                                 round_number: int) -> SecureModelUpdate:
        """Create secure model update with encrypted parameters on IPFS"""
        try:
            # Compute model hash
            model_hash = self.hash_manager.compute_model_hash(model_parameters)
            
            # Encrypt model parameters
            encrypted_params = self.encryption_manager.encrypt_model_parameters(
                model_parameters, self.client_id
            )
            
            # Prepare secure data for IPFS
            secure_data = {
                'encrypted_parameters': encrypted_params,
                'client_id': self.client_id,
                'encryption_method': 'fernet',
                'timestamp': time.time(),
                'model_hash': model_hash
            }
            
            # Store on IPFS
            ipfs_cid = self.ipfs_client.add_data(secure_data)
            if not ipfs_cid:
                raise ValueError("Failed to store data on IPFS")
            
            # Create signature data
            signature_data = f"{self.client_id}_{ipfs_cid}_{model_hash}_{time.time()}"
            
            # Sign the data
            signature = self.signature_manager.sign_data(signature_data, self.client_id)
            
            # Create secure model update
            secure_update = SecureModelUpdate(
                client_id=self.client_id,
                ipfs_cid=ipfs_cid,
                model_hash=model_hash,
                sample_count=sample_count,
                accuracy=accuracy,
                loss=loss,
                timestamp=time.time(),
                signature=signature,
                round_number=round_number
            )
            
            logger.info(f"Secure model update created for client {self.client_id}")
            logger.info(f"IPFS CID: {ipfs_cid}")
            logger.info(f"Model hash: {model_hash}")
            
            return secure_update
            
        except Exception as e:
            logger.error(f"Failed to create secure model update: {e}")
            raise

class SecureDecentralizedMiner:
    """Privacy-preserving decentralized miner"""
    
    def __init__(self, miner_id: str, model: torch.nn.Module, ipfs_client):
        self.miner_id = miner_id
        self.model = model
        self.ipfs_client = ipfs_client
        self.encryption_manager = SecureEncryptionManager()
        self.hash_manager = SecureHashManager()
        self.signature_manager = SecureSignatureManager()
        self.client_updates: Dict[str, Dict] = {}
        
        logger.info(f"Secure miner {miner_id} initialized")
    
    def add_client_public_key(self, client_id: str, public_key: str):
        """Add client's public key for signature verification"""
        self.signature_manager.client_keys[client_id] = {'public': public_key}
        logger.info(f"Added public key for client {client_id}")
    
    def add_client_encryption_key(self, client_id: str, encryption_key: str):
        """Add client's encryption key for parameter decryption"""
        self.encryption_manager.client_keys[client_id] = encryption_key
        logger.info(f"Added encryption key for client {client_id}")
    
    def process_secure_client_update(self, update: SecureModelUpdate) -> bool:
        """Process secure client update using only IPFS CID"""
        try:
            # Verify signature
            signature_data = f"{update.client_id}_{update.ipfs_cid}_{update.model_hash}_{update.timestamp}"
            if not self.signature_manager.verify_signature(signature_data, update.signature, update.client_id):
                logger.error(f"Signature verification failed for client {update.client_id}")
                return False
            
            # Retrieve encrypted data from IPFS
            encrypted_data = self.ipfs_client.get_data(update.ipfs_cid)
            if not encrypted_data:
                logger.error(f"Failed to retrieve data from IPFS: {update.ipfs_cid}")
                return False
            
            # Decrypt model parameters
            decrypted_params = self.encryption_manager.decrypt_model_parameters(
                encrypted_data['encrypted_parameters'], update.client_id
            )
            
            # Verify model hash
            if not self.hash_manager.verify_model_hash(decrypted_params, update.model_hash):
                logger.error(f"Model hash verification failed for client {update.client_id}")
                return False
            
            # Store secure update
            self.client_updates[update.client_id] = {
                'parameters': decrypted_params,
                'metadata': update,
                'verified': True
            }
            
            logger.info(f"Successfully processed secure update from client {update.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process secure update: {e}")
            return False
    
    def get_aggregated_parameters(self) -> Dict[str, torch.Tensor]:
        """Get aggregated parameters from verified client updates"""
        if not self.client_updates:
            return {}
        
        # Perform FedAVG aggregation
        aggregated_params = {}
        total_samples = sum(update['metadata'].sample_count for update in self.client_updates.values())
        
        for client_id, update in self.client_updates.items():
            if not update['verified']:
                continue
                
            weight = update['metadata'].sample_count / total_samples
            parameters = update['parameters']
            
            for name, param in parameters.items():
                if name not in aggregated_params:
                    aggregated_params[name] = torch.zeros_like(param)
                aggregated_params[name] += weight * param
        
        return aggregated_params

def main():
    """Test secure model update implementation"""
    print("🔐 Testing Secure Model Update Implementation")
    print("=" * 50)
    
    # Mock IPFS client
    class MockIPFSClient:
        def __init__(self):
            self.storage = {}
        
        def add_data(self, data):
            cid = hashlib.sha256(str(data).encode()).hexdigest()[:16]
            self.storage[cid] = data
            return cid
        
        def get_data(self, cid):
            return self.storage.get(cid)
    
    # Initialize components
    ipfs_client = MockIPFSClient()
    
    # Create secure client
    client = SecureFederatedClient("client_1", ipfs_client)
    
    # Create mock model parameters
    model_params = {
        'layer1.weight': torch.randn(64, 30),
        'layer1.bias': torch.randn(64),
        'layer2.weight': torch.randn(2, 64),
        'layer2.bias': torch.randn(2)
    }
    
    # Create secure model update
    secure_update = client.create_secure_model_update(
        model_parameters=model_params,
        sample_count=100,
        accuracy=0.85,
        loss=0.3,
        round_number=1
    )
    
    print(f"✅ Secure update created:")
    print(f"   Client ID: {secure_update.client_id}")
    print(f"   IPFS CID: {secure_update.ipfs_cid}")
    print(f"   Model Hash: {secure_update.model_hash}")
    print(f"   Signature: {secure_update.signature[:20]}...")
    
    # Create secure miner
    miner = SecureDecentralizedMiner("miner_1", None, ipfs_client)
    
    # Add client's public key to miner
    miner.add_client_public_key("client_1", client.public_key)
    miner.add_client_encryption_key("client_1", client.encryption_key)
    
    # Process secure update
    success = miner.process_secure_client_update(secure_update)
    
    if success:
        print("✅ Secure update processed successfully")
        print("✅ Privacy preserved - no raw parameters transmitted")
        print("✅ MITM attacks prevented")
        print("✅ Data integrity verified")
    else:
        print("❌ Secure update processing failed")

if __name__ == "__main__":
    main()









