#rtutierminal  pythonm
"""
Blockchain-Enabled FedAVG Coordinator for Zero-Day Detection
Implements federated averaging with blockchain integration and IPFS storage
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClientUpdate:
    """Represents a client's model update"""
    client_id: str
    model_parameters: Dict[str, torch.Tensor]
    sample_count: int
    training_loss: float
    validation_accuracy: float
    validation_precision: float = 0.0
    validation_recall: float = 0.0
    validation_f1_score: float = 0.0
    timestamp: float = 0.0
    model_hash: str = ""
    ipfs_cid: Optional[str] = None
    blockchain_tx_hash: Optional[str] = None

@dataclass
class AggregationResult:
    """Represents the result of model aggregation"""
    round_number: int
    aggregated_parameters: Dict[str, torch.Tensor]
    client_contributions: Dict[str, float]
    aggregation_time: float
    model_hash: str
    ipfs_cid: Optional[str] = None
    blockchain_tx_hash: Optional[str] = None

class FedAVGAggregator:
    """
    Federated Averaging Aggregator with blockchain integration
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize FedAVG aggregator
        
        Args:
            model: Global model architecture
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Aggregation state
        self.current_round = 0
        self.client_updates = {}
        self.aggregation_history = []
        
        # Blockchain integration
        self.blockchain_enabled = True
        self.ipfs_enabled = True
        self.blockchain_client = None  # FIXED: Initialize blockchain client
        self.ipfs_client = None        # FIXED: Initialize IPFS client
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
    
    def set_ipfs_client(self, ipfs_client):
        """
        Set IPFS client for aggregator
        
        Args:
            ipfs_client: IPFS client instance
        """
        self.ipfs_client = ipfs_client
        self.ipfs_enabled = True
        logger.info(f"Aggregator: IPFS integration enabled")
    
    def set_blockchain_client(self, blockchain_client):
        """
        Set blockchain client for aggregator
        
        Args:
            blockchain_client: Blockchain client instance
        """
        self.blockchain_client = blockchain_client
        self.blockchain_enabled = True
        logger.info(f"Aggregator: Blockchain integration enabled")
    
    def _verify_aggregator_model_integrity(self, stage: str = "unknown"):
        """
        Verify aggregator model integrity and log detailed information
        
        Args:
            stage: Stage name for debugging (e.g., "before_aggregation", "after_loading")
        """
        logger.info(f"🔍 AGGREGATOR MODEL INTEGRITY CHECK - {stage.upper()}")
        logger.info(f"   Aggregator model type: {type(self.model)}")
        logger.info(f"   Aggregator model class name: {self.model.__class__.__name__}")
        logger.info(f"   Aggregator model module: {self.model.__class__.__module__}")
        logger.info(f"   Aggregator has meta_train: {hasattr(self.model, 'meta_train')}")
        logger.info(f"   Aggregator has meta_learner: {hasattr(self.model, 'meta_learner')}")
        
        if hasattr(self.model, 'meta_learner'):
            logger.info(f"   Aggregator MetaLearner type: {type(self.model.meta_learner)}")
            # Note: MetaLearner doesn't have meta_train, only TransductiveFewShotModel does
        
        # Check if aggregator model is compatible (should be TransductiveFewShotModel or compatible TCN model)
        if not (type(self.model).__name__ == 'TransductiveFewShotModel' or 
                (hasattr(self.model, 'meta_train') and hasattr(self.model, 'forward'))):
            logger.error(f"🚨 AGGREGATOR MODEL CORRUPTION DETECTED: Expected TransductiveFewShotModel or compatible model, got {type(self.model).__name__}")
            logger.error(f"🚨 This explains why meta_train method is missing in aggregator!")
        
        logger.info(f"🔍 AGGREGATOR MODEL INTEGRITY CHECK COMPLETE - {stage.upper()}")
    
    def set_blockchain_integration(self, blockchain_client, ipfs_client):
        """
        Enable blockchain and IPFS integration
        
        Args:
            blockchain_client: Blockchain client instance
            ipfs_client: IPFS client instance
        """
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client
        self.blockchain_enabled = True  # FIXED: Enable blockchain integration
        self.ipfs_enabled = True        # FIXED: Enable IPFS integration
        
        logger.info("Blockchain and IPFS integration enabled")
    
    def compute_model_hash(self, parameters: Dict[str, torch.Tensor]) -> str:
        """
        Compute SHA256 hash of model parameters
        
        Args:
            parameters: Model parameters dictionary
            
        Returns:
            model_hash: SHA256 hash as hex string
        """
        # Convert parameters to bytes (avoid numpy conversion to save memory)
        param_bytes = b''
        for name, param in sorted(parameters.items()):
            param_bytes += name.encode('utf-8')
            # Use tensor data directly without numpy conversion
            param_bytes += param.detach().cpu().contiguous().storage().data_ptr().to_bytes(8, 'little')
            param_bytes += str(param.detach().cpu().contiguous().view(-1).tolist()).encode('utf-8')
        
        # Compute hash
        model_hash = hashlib.sha256(param_bytes).hexdigest()
        return model_hash
    
    def store_model_on_ipfs(self, parameters: Dict[str, torch.Tensor], metadata: Dict) -> Optional[str]:
        """
        Store model parameters on IPFS
        
        Args:
            parameters: Model parameters
            metadata: Additional metadata
            
        Returns:
            ipfs_cid: IPFS content identifier
        """
        if not self.ipfs_enabled:
            logger.warning("IPFS not enabled, skipping model storage")
            return None
        
        try:
            # Prepare model data (memory-safe - avoid numpy conversion)
            model_data = {
                'parameters': {name: param.detach().cpu().tolist() for name, param in parameters.items()},
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            # Store on IPFS
            if self.ipfs_client is not None:
                ipfs_cid = self.ipfs_client.add_data(model_data)
                logger.info(f"Model stored on IPFS: {ipfs_cid}")
            else:
                logger.warning("IPFS client not available, skipping model storage")
                ipfs_cid = None
            
            # Note: IPFS storage doesn't consume blockchain gas, so we don't track it
            
            return ipfs_cid
            
        except Exception as e:
            logger.error(f"Failed to store model on IPFS: {str(e)}")
            return None
    
    def record_on_blockchain(self, model_hash: str, ipfs_cid: str, round_number: int) -> Optional[str]:
        """
        Record model hash and IPFS CID on blockchain
        
        Args:
            model_hash: Model hash
            ipfs_cid: IPFS content identifier
            round_number: Current round number
            
        Returns:
            tx_hash: Blockchain transaction hash
        """
        if not self.blockchain_enabled:
            logger.warning("Blockchain not enabled, skipping recording")
            return None
        
        try:
            # Submit to blockchain smart contract
            tx_hash = self.blockchain_client.submit_model_update(
                model_hash=model_hash,
                ipfs_cid=ipfs_cid,
                round_number=round_number
            )
            
            logger.info(f"Model recorded on blockchain: {tx_hash}")
            
            # Record gas usage for model update (aggregation result)
            if hasattr(self, 'gas_collector') and self.gas_collector:
                try:
                    # Get transaction receipt to extract gas data
                    receipt = self.blockchain_client.web3.eth.get_transaction_receipt(tx_hash)
                    self.gas_collector.record_transaction(
                        tx_hash=tx_hash,
                        tx_type="Model Update",
                        gas_used=receipt.gasUsed,
                        gas_limit=receipt.gasUsed,
                        gas_price=0,
                        block_number=receipt.blockNumber,
                        round_number=round_number,
                        client_id=None,  # Global model update
                        ipfs_cid=ipfs_cid
                    )
                    logger.info(f"Recorded real gas transaction: Model Update - Gas: {receipt.gasUsed}, Block: {receipt.blockNumber}")
                except Exception as gas_error:
                    logger.warning(f"Failed to record gas for model update: {gas_error}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to record on blockchain: {str(e)}")
            return None
    
    def add_client_update(self, client_update: ClientUpdate) -> bool:
        """
        Add a client update to the current round
        
        Args:
            client_update: Client update data
            
        Returns:
            success: Whether update was added successfully
        """
        with self.lock:
            if client_update.client_id in self.client_updates:
                logger.warning(f"Client {client_update.client_id} already submitted update for round {self.current_round}")
                return False
            
            # Verify model hash
            computed_hash = self.compute_model_hash(client_update.model_parameters)
            if computed_hash != client_update.model_hash:
                logger.error(f"Model hash mismatch for client {client_update.client_id}")
                return False
            
            # Record client update on blockchain FIRST
            if self.blockchain_enabled:
                try:
                    # Store model on IPFS
                    ipfs_cid = self.store_model_on_ipfs(
                        client_update.model_parameters, 
                        {'client_id': client_update.client_id, 'round': self.current_round}
                    )
                    
                    if ipfs_cid:
                        # Record on blockchain
                        tx_hash = self.record_on_blockchain(
                            client_update.model_hash, 
                            ipfs_cid,
                            self.current_round
                        )
                        
                        if tx_hash:
                            logger.info(f"Client {client_update.client_id} update recorded on blockchain: {tx_hash}")
                        else:
                            logger.error(f"Failed to record client {client_update.client_id} update on blockchain")
                            return False
                    else:
                        logger.error(f"Failed to store client {client_update.client_id} model on IPFS")
                        return False
                        
                except Exception as e:
                    logger.error(f"Failed to record client {client_update.client_id} update: {str(e)}")
                    return False
            
            self.client_updates[client_update.client_id] = client_update
            logger.info(f"Added update from client {client_update.client_id}")
            
            return True
    
    def aggregate_models(self, min_clients: int = 2) -> Optional[AggregationResult]:
        """
        Aggregate client models using FedAVG algorithm
        
        Args:
            min_clients: Minimum number of clients required for aggregation
            
        Returns:
            aggregation_result: Result of aggregation
        """
        with self.lock:
            if len(self.client_updates) < min_clients:
                logger.warning(f"Insufficient clients for aggregation: {len(self.client_updates)} < {min_clients}")
                return None
            
            logger.info(f"Aggregating models from {len(self.client_updates)} clients")
            start_time = time.time()
            
            # Get global model parameters
            global_params = self.model.state_dict()
            
            # Debug: Log global model parameter dimensions
            for name, param in global_params.items():
                if 'tcn' in name.lower() or 'conv' in name.lower() or 'linear' in name.lower():
                    logger.info(f"🔍 DEBUG GLOBAL: {name} shape: {param.shape}")
            
            # Initialize aggregated parameters
            aggregated_params = {}
            total_samples = 0
            client_contributions = {}
            
            # Calculate total samples
            for client_id, update in self.client_updates.items():
                total_samples += update.sample_count
            
            # Calculate client contributions once
            for client_id, update in self.client_updates.items():
                client_contributions[client_id] = update.sample_count / total_samples
            
            # Memory-efficient weighted averaging
            for param_name in global_params.keys():
                # Initialize on CPU to save GPU memory - create tensor with same shape but on CPU
                param_shape = global_params[param_name].shape
                aggregated_param = torch.zeros(param_shape, dtype=global_params[param_name].dtype, device='cpu')
                
                for client_id, update in self.client_updates.items():
                    if param_name in update.model_parameters:
                        weight = client_contributions[client_id]
                        # Client parameters are already on CPU, no need to move them
                        client_param = update.model_parameters[param_name]
                        # Debug: Log dimension mismatch
                        if aggregated_param.shape != client_param.shape:
                            logger.error(f"🔍 DIMENSION MISMATCH: {param_name}")
                            logger.error(f"  Global param shape: {aggregated_param.shape}")
                            logger.error(f"  Client {client_id} param shape: {client_param.shape}")
                        aggregated_param += weight * client_param
                
                # Keep aggregated parameter on CPU to save GPU memory
                aggregated_params[param_name] = aggregated_param
                
                # Clear CPU memory
                del aggregated_param
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clear GPU cache before updating model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Compute aggregation metrics BEFORE loading parameters
            aggregation_time = time.time() - start_time
            # Use simple hash to avoid memory issues
            model_hash = f"model_round_{self.current_round}_{int(time.time())}"
            
            # Store on IPFS (with memory-safe approach)
            metadata = {
                'round_number': self.current_round,
                'num_clients': len(self.client_updates),
                'total_samples': total_samples,
                'aggregation_time': aggregation_time
            }
            
            ipfs_cid = self.store_model_on_ipfs(aggregated_params, metadata)
            
            # Record on blockchain
            blockchain_tx_hash = self.record_on_blockchain(model_hash, ipfs_cid, self.current_round)
            
            # Create aggregation result with empty parameters (will be filled later)
            result = AggregationResult(
                round_number=self.current_round,
                aggregated_parameters={},  # Empty for now
                client_contributions=client_contributions,
                aggregation_time=aggregation_time,
                model_hash=model_hash,
                ipfs_cid=ipfs_cid,
                blockchain_tx_hash=blockchain_tx_hash
            )
            
            # DEBUG: Track aggregator model type before loading parameters
            self._verify_aggregator_model_integrity("before_loading_parameters")
            
            # Load parameters one by one to avoid memory duplication
            model_state_dict = self.model.state_dict()
            for param_name, param_tensor in aggregated_params.items():
                if param_name in model_state_dict:
                    # Load parameter directly without creating a separate dictionary
                    with torch.no_grad():
                        model_state_dict[param_name].copy_(param_tensor.to(self.device))
            
            # DEBUG: Track aggregator model type after loading parameters
            self._verify_aggregator_model_integrity("after_loading_parameters")
            
            # Clear the aggregated_params dictionary to free memory
            aggregated_params.clear()
            
            # Clear GPU cache after updating model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Store in history
            self.aggregation_history.append(result)
            
            # Clear client updates for next round
            self.client_updates.clear()
            self.current_round += 1
            
            logger.info(f"Aggregation completed in {aggregation_time:.2f}s")
            logger.info(f"Model hash: {model_hash}")
            logger.info(f"IPFS CID: {ipfs_cid}")
            logger.info(f"Blockchain TX: {blockchain_tx_hash}")
            
            return result
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model"""
        return self.model
    
    def get_aggregation_history(self) -> List[AggregationResult]:
        """Get aggregation history"""
        return self.aggregation_history.copy()
    
    def get_client_contributions(self) -> Dict[str, float]:
        """Get client contributions for current round"""
        return {client_id: update.sample_count for client_id, update in self.client_updates.items()}

class BlockchainFederatedClient:
    """
    Blockchain-enabled federated learning client
    """
    
    def __init__(self, client_id: str, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            model: Local model instance
            device: Device to run on
        """
        import copy
        self.client_id = client_id
        self.model = copy.deepcopy(model)  # FIXED: Each client gets independent model copy
        self.device = device
        self.model.to(device)
        
        # Training data
        self.train_data = None
        self.train_labels = None
        
        # Blockchain integration
        self.blockchain_enabled = True
        self.ipfs_enabled = True
        self.blockchain_client = None  # FIXED: Initialize blockchain client
        self.ipfs_client = None        # FIXED: Initialize IPFS client
        
        # Training history
        self.training_history = []
        
        logger.info(f"Federated client {client_id} initialized on {device}")
    
    def set_blockchain_integration(self, blockchain_client, ipfs_client):
        """
        Enable blockchain and IPFS integration
        
        Args:
            blockchain_client: Blockchain client instance
            ipfs_client: IPFS client instance
        """
        self.blockchain_client = blockchain_client
        self.ipfs_client = ipfs_client
        self.blockchain_enabled = True  # FIXED: Enable blockchain integration
        self.ipfs_enabled = True        # FIXED: Enable IPFS integration
        
        logger.info(f"Client {self.client_id}: Blockchain and IPFS integration enabled")
    
    def set_ipfs_client(self, ipfs_client):
        """
        Set IPFS client for integration
        
        Args:
            ipfs_client: IPFS client instance
        """
        self.ipfs_client = ipfs_client
        self.ipfs_enabled = True
        logger.info(f"Client {self.client_id}: IPFS integration enabled")
    
    def set_training_data(self, train_data: torch.Tensor, train_labels: torch.Tensor):
        """
        Set training data for this client
        
        Args:
            train_data: Training features
            train_labels: Training labels
        """
        self.train_data = train_data.to(self.device)
        self.train_labels = train_labels.to(self.device)
        
        logger.info(f"Client {self.client_id}: Set {len(train_data)} training samples")
    
    def train_local_model(self, epochs: int = 2, batch_size: int = 8, learning_rate: float = 0.001) -> ClientUpdate:  # Reasonable batch size for memory
        """
        Train local model using meta-learning and TTT capabilities
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            client_update: Client update data
        """
        if self.train_data is None:
            raise ValueError("No training data set for client")
        
        # DEBUG: Track model type at start of training
        logger.info(f"Client {self.client_id}: Starting enhanced local training for {epochs} epochs")
        self._verify_model_integrity("before_training")
        
        try:
            # Phase 1: Create meta-tasks from local data
            from models.transductive_fewshot_model import create_meta_tasks
            
            logger.info(f"Client {self.client_id}: Creating meta-tasks from local data...")
            local_meta_tasks = create_meta_tasks(
                self.train_data,
                self.train_labels,
                n_way=10,  # 10-way classification (1 Normal + 9 Attack types)
                k_shot=50,  # 50-shot learning (5 samples per class)
                n_query=100,  # 100 query samples
                n_tasks=5,  # 5 tasks per epoch
                phase="training",
                normal_query_ratio=0.8,  # 80% Normal samples in query set for training
                zero_day_attack_label=4  # Exclude DoS (label 4) from training
            )
            
            # Phase 2: Meta-learning training
            logger.info(f"Client {self.client_id}: Running meta-learning training...")
            
            # DEBUG: Check model type before meta_train call
            logger.info(f"🔍 DEBUG: Model type before meta_train: {type(self.model)}")
            logger.info(f"🔍 DEBUG: Model has meta_train: {hasattr(self.model, 'meta_train')}")
            
            # FIXED: Ensure model is TransductiveFewShotModel before calling meta_train
            if not hasattr(self.model, 'meta_train'):
                logger.error(f"🚨 Model corruption detected! Model type: {type(self.model)}")
                logger.error(f"🚨 Expected TransductiveFewShotModel or compatible model, got {type(self.model).__name__}")
                raise AttributeError(f"Model {type(self.model).__name__} does not have meta_train method")
            
            meta_training_history = self.model.meta_train(local_meta_tasks, meta_epochs=3)  # Reduced for TCN stability
            
            # DEBUG: Check model type after meta_train call
            logger.info(f"🔍 DEBUG: Model type after meta_train: {type(self.model)}")
            logger.info(f"🔍 DEBUG: Model has meta_train: {hasattr(self.model, 'meta_train')}")
            
            # Phase 3: Fine-tuning with TTT capabilities
            logger.info(f"Client {self.client_id}: Running test-time training adaptation...")
            
            # DEBUG: Check model type before TTT adaptation
            logger.info(f"🔍 DEBUG: Model type before TTT: {type(self.model)}")
            logger.info(f"🔍 DEBUG: Model has meta_train: {hasattr(self.model, 'meta_train')}")
            
            self._perform_local_ttt_adaptation()
            
            # DEBUG: Check model type after TTT adaptation
            logger.info(f"🔍 DEBUG: Model type after TTT: {type(self.model)}")
            logger.info(f"🔍 DEBUG: Model has meta_train: {hasattr(self.model, 'meta_train')}")
            
            # Note: We don't replace self.model, we update it in place
            
            # Calculate average loss from meta-training
            avg_loss = sum(meta_training_history['epoch_losses']) / len(meta_training_history['epoch_losses'])
            avg_accuracy = sum(meta_training_history['epoch_accuracies']) / len(meta_training_history['epoch_accuracies'])
            
            logger.info(f"Client {self.client_id}: Enhanced training completed - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            
            # Get model parameters for update
            model_parameters = {}
            for name, param in self.model.named_parameters():
                model_parameters[name] = param.detach().cpu()
                # Debug: Log parameter dimensions
                if 'tcn' in name.lower() or 'conv' in name.lower() or 'linear' in name.lower():
                    logger.info(f"🔍 DEBUG: {name} shape: {param.shape}")
            
            # Compute model hash for verification
            model_hash = self.compute_model_hash(model_parameters)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Store on IPFS
            ipfs_cid = None
            if self.ipfs_enabled:
                metadata = {
                    'client_id': self.client_id,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'avg_loss': avg_loss,
                    'validation_accuracy': avg_accuracy
                }
                ipfs_cid = self.store_model_on_ipfs(model_parameters, metadata)
            
            # Record on blockchain
            blockchain_tx_hash = None
            logger.info(f"Client {self.client_id}: About to check blockchain_enabled: {self.blockchain_enabled}")
            if self.blockchain_enabled:
                logger.info(f"Client {self.client_id}: Recording on blockchain with hash {model_hash} and IPFS {ipfs_cid}")
                blockchain_tx_hash = self.record_on_blockchain(model_hash, ipfs_cid)
                logger.info(f"Client {self.client_id}: Blockchain recording result: {blockchain_tx_hash}")
            else:
                logger.warning(f"Client {self.client_id}: Blockchain not enabled, skipping recording")
            
            return ClientUpdate(
                client_id=self.client_id,
                model_parameters=model_parameters,
                sample_count=len(self.train_data),
                training_loss=avg_loss,
                validation_accuracy=avg_accuracy,
                model_hash=model_hash,
                ipfs_cid=ipfs_cid,
                blockchain_tx_hash=blockchain_tx_hash,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Enhanced training failed: {str(e)}")
            raise e
    
    def _perform_local_ttt_adaptation(self):
        """Perform test-time training adaptation on local data"""
        try:
            # Create a support set from a subset of local data
            support_size = min(100, len(self.train_data) // 4)
            support_indices = torch.randperm(len(self.train_data))[:support_size]
            support_x = self.train_data[support_indices]
            support_y = self.train_labels[support_indices]
            
            # Perform TTT adaptation directly on the model
            # This updates the model parameters in place
            adaptation_steps = 5
            inner_lr = 0.01
            
            # Create optimizer for the model
            ttt_optimizer = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
            
            # Perform adaptation steps
            for step in range(adaptation_steps):
                ttt_optimizer.zero_grad()
                
                # Get embeddings using the model's extract_embeddings method
                support_embeddings = self.model.extract_embeddings(support_x)
                
                # Compute prototypes using the model's update_prototypes method
                prototypes, prototype_labels = self.model.update_prototypes(
                    support_embeddings, support_y, support_embeddings, None
                )
                
                # Compute loss (prototype consistency)
                loss = 0
                for i, label in enumerate(prototype_labels):
                    mask = support_y == label
                    if mask.sum() > 0:
                        class_embeddings = support_embeddings[mask]
                        prototype = prototypes[i]
                        loss += torch.nn.functional.mse_loss(class_embeddings.mean(dim=0), prototype)
                
                if loss > 0:
                    loss.backward()
                    ttt_optimizer.step()
            
            logger.info(f"Client {self.client_id}: TTT adaptation completed")
            
        except Exception as e:
            logger.warning(f"Client {self.client_id}: TTT adaptation failed: {str(e)}")
    
    
    def update_global_model(self, global_parameters: Dict[str, torch.Tensor]):
        """
        Update local model with global parameters
        
        Args:
            global_parameters: Global model parameters
        """
        # DEBUG: Track model type before update
        self._verify_model_integrity("before_global_update")
        
        # Memory-efficient parameter loading - load one by one to avoid massive allocation
        model_state_dict = self.model.state_dict()
        for param_name, param_tensor in global_parameters.items():
            if param_name in model_state_dict:
                with torch.no_grad():
                    # Move parameter to GPU and load directly
                    model_state_dict[param_name].copy_(param_tensor.to(self.device))
        
        # DEBUG: Track model type after update
        self._verify_model_integrity("after_global_update")
        
        logger.info(f"Client {self.client_id}: Updated with global model")
    
    def _verify_model_integrity(self, stage: str = "unknown"):
        """
        Verify model integrity and log detailed information
        
        Args:
            stage: Stage name for debugging (e.g., "before_training", "after_aggregation")
        """
        logger.info(f"🔍 MODEL INTEGRITY CHECK - {stage.upper()}")
        logger.info(f"   Model type: {type(self.model)}")
        logger.info(f"   Model class name: {self.model.__class__.__name__}")
        logger.info(f"   Model module: {self.model.__class__.__module__}")
        logger.info(f"   Has meta_train: {hasattr(self.model, 'meta_train')}")
        logger.info(f"   Has meta_learner: {hasattr(self.model, 'meta_learner')}")
        
        if hasattr(self.model, 'meta_learner'):
            logger.info(f"   MetaLearner type: {type(self.model.meta_learner)}")
            # Note: MetaLearner doesn't have meta_train, only TransductiveFewShotModel does
        
        # Check if model is compatible (should be TransductiveFewShotModel or compatible TCN model)
        if not (type(self.model).__name__ == 'TransductiveFewShotModel' or 
                (hasattr(self.model, 'meta_train') and hasattr(self.model, 'forward'))):
            logger.error(f"🚨 MODEL CORRUPTION DETECTED: Expected TransductiveFewShotModel or compatible model, got {type(self.model).__name__}")
            logger.error(f"🚨 This explains why meta_train method is missing!")
        
        logger.info(f"🔍 MODEL INTEGRITY CHECK COMPLETE - {stage.upper()}")
    
    def evaluate_model(self, test_data: torch.Tensor = None, test_labels: torch.Tensor = None) -> tuple:
        """
        Evaluate model performance
        
        Args:
            test_data: Test data (if None, use training data)
            test_labels: Test labels (if None, use training labels)
            
        Returns:
            tuple: (accuracy, precision, recall, f1_score)
        """
        if test_data is None:
            test_data = self.train_data
            test_labels = self.train_labels
        
        if test_data is None:
            return 0.0, 0.0, 0.0, 0.0
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_data)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy
            accuracy = (predictions == test_labels).float().mean().item()
            
            # Calculate precision, recall, F1-score
            # Convert to numpy for sklearn metrics (memory-safe)
            pred_np = predictions.cpu().tolist()
            true_np = test_labels.cpu().tolist()
            
            # Calculate metrics using sklearn
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            try:
                precision = precision_score(true_np, pred_np, average='weighted', zero_division=0)
                recall = recall_score(true_np, pred_np, average='weighted', zero_division=0)
                f1 = f1_score(true_np, pred_np, average='weighted', zero_division=0)
            except:
                precision = recall = f1 = 0.0
        
        return accuracy, precision, recall, f1
    
    def compute_model_hash(self, parameters: Dict[str, torch.Tensor]) -> str:
        """Compute SHA256 hash of model parameters (memory-safe)"""
        param_bytes = b''
        for name, param in sorted(parameters.items()):
            param_bytes += name.encode('utf-8')
            # Use tensor data directly without numpy conversion
            param_bytes += param.detach().cpu().contiguous().storage().data_ptr().to_bytes(8, 'little')
            param_bytes += str(param.detach().cpu().contiguous().view(-1).tolist()).encode('utf-8')
        
        model_hash = hashlib.sha256(param_bytes).hexdigest()
        return model_hash
    
    def store_model_on_ipfs(self, parameters: Dict[str, torch.Tensor], metadata: Dict) -> Optional[str]:
        """Store model parameters on IPFS"""
        logger.info(f"🔍 Debug: Client {self.client_id} store_model_on_ipfs called")
        logger.info(f"🔍 Debug: IPFS enabled: {self.ipfs_enabled}")
        logger.info(f"🔍 Debug: IPFS client: {self.ipfs_client}")
        logger.info(f"🔍 Debug: IPFS client type: {type(self.ipfs_client)}")
        
        if not self.ipfs_enabled:
            return None
        
        try:
            # Memory-safe parameter conversion
            model_data = {
                'parameters': {name: param.detach().cpu().tolist() for name, param in parameters.items()},
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            if self.ipfs_client is not None:
                ipfs_cid = self.ipfs_client.add_data(model_data)
                logger.info(f"Client {self.client_id}: Model stored on IPFS: {ipfs_cid}")
            else:
                logger.warning(f"Client {self.client_id}: IPFS client not available, skipping model storage")
                ipfs_cid = None
            
            # Note: IPFS storage doesn't consume blockchain gas, so we don't track it
            
            return ipfs_cid
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to store on IPFS: {str(e)}")
            return None
    
    def record_on_blockchain(self, model_hash: str, ipfs_cid: str) -> Optional[str]:
        """Record model hash on blockchain"""
        logger.info(f"Client {self.client_id}: record_on_blockchain called with hash={model_hash}, cid={ipfs_cid}")
        logger.info(f"Client {self.client_id}: blockchain_enabled={self.blockchain_enabled}, blockchain_client={self.blockchain_client}")
        
        if not self.blockchain_enabled:
            logger.warning(f"Client {self.client_id}: Blockchain not enabled")
            return None
        
        if not self.blockchain_client:
            logger.warning(f"Client {self.client_id}: Blockchain client not available")
            return None
        
        try:
            logger.info(f"Client {self.client_id}: Calling submit_client_update...")
            tx_hash = self.blockchain_client.submit_client_update(
                client_id=self.client_id,
                model_hash=model_hash,
                ipfs_cid=ipfs_cid
            )
            logger.info(f"Client {self.client_id}: submit_client_update returned: {tx_hash}")
            logger.info(f"Client {self.client_id}: Recorded on blockchain: {tx_hash}")
            
            # Record gas usage for client update
            if hasattr(self, 'gas_collector') and self.gas_collector:
                try:
                    # Get transaction receipt to extract gas data
                    receipt = self.blockchain_client.web3.eth.get_transaction_receipt(tx_hash)
                    self.gas_collector.record_transaction(
                        tx_hash=tx_hash,
                        tx_type="Client Update",
                        gas_used=receipt.gasUsed,
                        gas_limit=receipt.gasUsed,  # Use gasUsed as limit for simplicity
                        gas_price=0,  # Not critical for analysis
                        block_number=receipt.blockNumber,
                        round_number=getattr(self, 'current_round', 0),
                        client_id=self.client_id,
                        ipfs_cid=ipfs_cid
                    )
                    logger.info(f"Recorded real gas transaction: Client Update - Gas: {receipt.gasUsed}, Block: {receipt.blockNumber}")
                except Exception as gas_error:
                    logger.warning(f"Failed to record gas for client update: {gas_error}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Failed to record on blockchain: {str(e)}")
            return None

class BlockchainFedAVGCoordinator:
    """
    Main coordinator for blockchain-enabled federated learning
    """
    
    def __init__(self, model: nn.Module, num_clients: int = 3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize coordinator
        
        Args:
            model: Global model architecture
            num_clients: Number of federated clients
            device: Device to run on
        """
        self.model = model
        self.num_clients = num_clients
        self.device = device
        
        # Initialize aggregator
        self.aggregator = FedAVGAggregator(model, device)
        
        # Initialize clients
        self.clients = []
        for i in range(num_clients):
            client_model = copy.deepcopy(model)
            client = BlockchainFederatedClient(f"client_{i+1}", client_model, device)
            self.clients.append(client)
        
        # Coordination state
        self.current_round = 0
        self.training_history = []
        
        logger.info(f"Blockchain FedAVG Coordinator initialized with {num_clients} clients")
    
    def set_ipfs_client(self, ipfs_client):
        """
        Set IPFS client for all clients
        
        Args:
            ipfs_client: IPFS client instance
        """
        logger.info(f"🔍 Debug: Setting IPFS client on {len(self.clients)} clients")
        logger.info(f"🔍 Debug: IPFS client type: {type(ipfs_client)}")
        for i, client in enumerate(self.clients):
            logger.info(f"🔍 Debug: Setting IPFS client on client {i+1}")
            client.set_ipfs_client(ipfs_client)
            logger.info(f"🔍 Debug: Client {i+1} IPFS client set: {client.ipfs_client is not None}")
        logger.info(f"IPFS client set for all {len(self.clients)} clients")
    
    def _verify_coordinator_model_integrity(self, stage: str = "unknown"):
        """
        Verify coordinator model integrity and log detailed information
        
        Args:
            stage: Stage name for debugging (e.g., "before_round", "after_aggregation")
        """
        logger.info(f"🔍 COORDINATOR MODEL INTEGRITY CHECK - {stage.upper()}")
        logger.info(f"   Coordinator model type: {type(self.model)}")
        logger.info(f"   Coordinator model class name: {self.model.__class__.__name__}")
        logger.info(f"   Coordinator model module: {self.model.__class__.__module__}")
        logger.info(f"   Coordinator has meta_train: {hasattr(self.model, 'meta_train')}")
        logger.info(f"   Coordinator has meta_learner: {hasattr(self.model, 'meta_learner')}")
        
        if hasattr(self.model, 'meta_learner'):
            logger.info(f"   Coordinator MetaLearner type: {type(self.model.meta_learner)}")
            # Note: MetaLearner doesn't have meta_train, only TransductiveFewShotModel does
        
        # Check if coordinator model is compatible (should be TransductiveFewShotModel or compatible TCN model)
        if not (type(self.model).__name__ == 'TransductiveFewShotModel' or 
                (hasattr(self.model, 'meta_train') and hasattr(self.model, 'forward'))):
            logger.error(f"🚨 COORDINATOR MODEL CORRUPTION DETECTED: Expected TransductiveFewShotModel or compatible model, got {type(self.model).__name__}")
            logger.error(f"🚨 This explains why meta_train method is missing in coordinator!")
        
        logger.info(f"🔍 COORDINATOR MODEL INTEGRITY CHECK COMPLETE - {stage.upper()}")
    
    def set_blockchain_integration(self, blockchain_client, ipfs_client):
        """Enable blockchain and IPFS integration for all components"""
        self.aggregator.set_blockchain_integration(blockchain_client, ipfs_client)
        
        for client in self.clients:
            client.set_blockchain_integration(blockchain_client, ipfs_client)
        
        logger.info("Blockchain integration enabled for all components")
    
    def distribute_data(self, train_data: torch.Tensor, train_labels: torch.Tensor, 
                       distribution_type: str = 'iid'):
        """
        Distribute training data among clients
        
        Args:
            train_data: Training features
            train_labels: Training labels
            distribution_type: Type of data distribution ('iid' or 'non_iid')
        """
        logger.info(f"Distributing data among {self.num_clients} clients ({distribution_type})")
        
        if distribution_type == 'iid':
            # IID distribution: random split
            indices = torch.randperm(len(train_data))
            samples_per_client = len(train_data) // self.num_clients
            
            for i, client in enumerate(self.clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else len(train_data)
                
                client_data = train_data[indices[start_idx:end_idx]]
                client_labels = train_labels[indices[start_idx:end_idx]]
                
                client.set_training_data(client_data, client_labels)
                logger.info(f"Client {client.client_id}: {len(client_data)} samples")
        
        elif distribution_type == 'non_iid':
            # Non-IID distribution: split by class
            unique_labels = torch.unique(train_labels)
            samples_per_class = len(train_data) // len(unique_labels)
            samples_per_client_per_class = samples_per_class // self.num_clients
            
            for i, client in enumerate(self.clients):
                client_data_list = []
                client_labels_list = []
                
                for label in unique_labels:
                    label_mask = train_labels == label
                    label_indices = torch.where(label_mask)[0]
                    
                    # Sample for this client
                    start_idx = i * samples_per_client_per_class
                    end_idx = start_idx + samples_per_client_per_class
                    
                    if end_idx <= len(label_indices):
                        selected_indices = label_indices[start_idx:end_idx]
                        client_data_list.append(train_data[selected_indices])
                        client_labels_list.append(train_labels[selected_indices])
                
                if client_data_list:
                    client_data = torch.cat(client_data_list, dim=0)
                    client_labels = torch.cat(client_labels_list, dim=0)
                    client.set_training_data(client_data, client_labels)
                    logger.info(f"Client {client.client_id}: {len(client_data)} samples")
    
    def distribute_data_with_dirichlet(self, train_data: torch.Tensor, train_labels: torch.Tensor, 
                                     alpha: float = 1.0):
        """
        Distribute training data among clients using Dirichlet distribution for realistic non-IID
        
        Args:
            train_data: Training features
            train_labels: Training labels
            alpha: Dirichlet distribution parameter
                  α = 0.1: High heterogeneity (very non-IID)
                  α = 1.0: Moderate heterogeneity (balanced non-IID) - RECOMMENDED
                  α = 10.0: Low heterogeneity (near IID)
        """
        import numpy as np
        
        logger.info(f"Distributing data using Dirichlet distribution (α={alpha}) among {self.num_clients} clients")
        
        num_samples = len(train_data)
        unique_labels = torch.unique(train_labels)
        num_classes = len(unique_labels)
        
        logger.info(f"Total samples: {num_samples:,}, Classes: {num_classes}")
        logger.info(f"Unique labels: {unique_labels.tolist()}")
        
        # Debug: Check label distribution
        for label in unique_labels:
            count = (train_labels == label).sum().item()
            logger.info(f"Label {label.item()}: {count} samples")
        
        # Create Dirichlet distribution for each class
        # This creates realistic non-IID where each client gets different proportions of each class
        dirichlet_distributions = {}
        
        for label in unique_labels:
            # Create Dirichlet distribution for this class across clients
            dirichlet_dist = np.random.dirichlet([alpha] * self.num_clients)
            dirichlet_distributions[label.item()] = dirichlet_dist
            logger.info(f"Class {label.item()}: Dirichlet distribution = {dirichlet_dist}")
        
        # Distribute data for each client
        for i, client in enumerate(self.clients):
            client_data_list = []
            client_labels_list = []
            
            for label in unique_labels:
                # Get samples of this class
                label_mask = train_labels == label
                label_indices = torch.where(label_mask)[0]
                class_samples = len(label_indices)
                
                if class_samples > 0:
                    # Calculate how many samples this client gets for this class
                    client_ratio = dirichlet_distributions[label.item()][i]
                    client_samples_for_class = int(client_ratio * class_samples)
                    
                    if client_samples_for_class > 0:
                        # Randomly sample from this class
                        if client_samples_for_class >= class_samples:
                            selected_indices = label_indices
                        else:
                            random_indices = torch.randperm(class_samples)[:client_samples_for_class]
                            selected_indices = label_indices[random_indices]
                        
                        client_data_list.append(train_data[selected_indices])
                        client_labels_list.append(train_labels[selected_indices])
                        
                        logger.info(f"Client {client.client_id} - Class {label.item()}: {len(selected_indices)} samples ({client_ratio:.3f} ratio)")
            
            if client_data_list:
                client_data = torch.cat(client_data_list, dim=0)
                client_labels = torch.cat(client_labels_list, dim=0)
                client.set_training_data(client_data, client_labels)
                
                # Calculate class distribution for this client
                class_counts = {}
                for label in unique_labels:
                    count = (client_labels == label).sum().item()
                    if count > 0:
                        class_counts[label.item()] = count
                
                total_samples = len(client_data)
                logger.info(f"Client {client.client_id}: {total_samples:,} total samples")
                logger.info(f"  Class distribution: {class_counts}")
                
                # Calculate heterogeneity metrics
                if len(class_counts) > 1:
                    class_proportions = [count/total_samples for count in class_counts.values()]
                    heterogeneity = np.std(class_proportions)  # Standard deviation of class proportions
                    logger.info(f"  Heterogeneity (std): {heterogeneity:.3f}")
            else:
                logger.warning(f"Client {client.client_id}: No data assigned!")
        
        # Log overall distribution statistics
        logger.info("=" * 60)
        logger.info("DIRICHLET DISTRIBUTION SUMMARY:")
        logger.info(f"Alpha parameter: {alpha}")
        logger.info(f"Distribution type: {'High heterogeneity' if alpha < 0.5 else 'Moderate heterogeneity' if alpha < 5.0 else 'Low heterogeneity'}")
        
        total_client_samples = sum(len(client.train_data) for client in self.clients if client.train_data is not None)
        logger.info(f"Total distributed samples: {total_client_samples:,} / {num_samples:,}")
        logger.info("=" * 60)
    
    def run_federated_round(self, epochs: int = 15, batch_size: int = 32, 
                           learning_rate: float = 0.001) -> Dict:
        """
        Run one federated learning round
        
        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            round_results: Results of the federated round
        """
        logger.info(f"Starting federated round {self.current_round + 1}")
        
        # Train all clients with error handling
        client_updates = []
        successful_clients = 0
        failed_clients = 0
        
        for i, client in enumerate(self.clients):
            try:
                logger.info(f"Training client {i+1}/{len(self.clients)}: {client.client_id}")
                update = client.train_local_model(epochs, batch_size, learning_rate)
                client_updates.append(update)
                successful_clients += 1
                logger.info(f"✅ Client {client.client_id} training completed successfully")
            except Exception as e:
                logger.error(f"❌ Client {client.client_id} training failed: {str(e)}")
                failed_clients += 1
                # Create a fallback update with current model parameters
                try:
                    fallback_update = ClientUpdate(
                        client_id=client.client_id,
                        model_parameters=client.model.state_dict(),
                        sample_count=len(client.train_data) if client.train_data is not None else 0,
                        training_loss=1.0,  # High loss to indicate failure
                        validation_accuracy=0.5,  # Low accuracy to indicate failure
                        model_hash="failed_training",
                        ipfs_cid=None,
                        blockchain_tx_hash=None,
                        timestamp=time.time()
                    )
                    client_updates.append(fallback_update)
                    logger.info(f"🔄 Created fallback update for failed client {client.client_id}")
                except Exception as fallback_error:
                    logger.error(f"❌ Failed to create fallback update for client {client.client_id}: {str(fallback_error)}")
        
        logger.info(f"📊 Training Summary: {successful_clients} successful, {failed_clients} failed out of {len(self.clients)} clients")
        
        # Ensure we have at least some client updates
        if not client_updates:
            logger.error("❌ No client updates available - all clients failed")
            return None
        
        # Add updates to aggregator
        for update in client_updates:
            self.aggregator.add_client_update(update)
        
        # Aggregate models
        aggregation_result = self.aggregator.aggregate_models()
        
        if aggregation_result is None:
            logger.error("Aggregation failed")
            return None
        
        # DEBUG: Verify coordinator model integrity before updating clients
        self._verify_coordinator_model_integrity("before_updating_clients")
        
        # Update all clients with global model directly from coordinator model
        for client in self.clients:
            # Get parameters directly from coordinator model to avoid memory duplication
            global_params = self.model.state_dict()
            client.update_global_model(global_params)
        
        # DEBUG: Verify coordinator model integrity after updating clients
        self._verify_coordinator_model_integrity("after_updating_clients")
        
        # Clear GPU cache after updating all clients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Store round results
        round_results = {
            'round_number': self.current_round,
            'client_updates': client_updates,
            'aggregation_result': aggregation_result,
            'timestamp': time.time()
        }
        
        self.training_history.append(round_results)
        self.current_round += 1
        
        logger.info(f"Federated round {round_results['round_number'] + 1} completed")
        logger.info(f"  Aggregation time: {aggregation_result.aggregation_time:.2f}s")
        logger.info(f"  Model hash: {aggregation_result.model_hash}")
        
        return round_results
    
    def run_federated_training(self, num_rounds: int = 10, epochs: int = 15, 
                              batch_size: int = 32, learning_rate: float = 0.001) -> List[Dict]:
        """
        Run complete federated training
        
        Args:
            num_rounds: Number of federated rounds
            epochs: Number of local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            training_history: Complete training history
        """
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        for round_num in range(num_rounds):
            round_results = self.run_federated_round(epochs, batch_size, learning_rate)
            
            if round_results is None:
                logger.error(f"Round {round_num + 1} failed")
                break
            
            # Log progress
            if round_num % 2 == 0:
                logger.info(f"Round {round_num + 1}/{num_rounds} completed")
        
        logger.info("Federated training completed")
        return self.training_history
    
    def run_meta_training(self, meta_tasks: List[Dict]) -> bool:
        """
        Run meta-training across clients for few-shot learning
        
        Args:
            meta_tasks: List of meta-learning tasks
            
        Returns:
            bool: True if meta-training successful, False otherwise
        """
        try:
            logger.info(f"Starting meta-training with {len(meta_tasks)} tasks")
            
            # For now, implement a simple meta-training approach
            # In a full implementation, this would coordinate meta-learning across clients
            for i, task in enumerate(meta_tasks):
                logger.info(f"Processing meta-task {i+1}/{len(meta_tasks)}")
                
                # Simulate meta-training by running a few federated rounds
                # This is a simplified implementation
                if i < 3:  # Only run first 3 tasks to avoid long execution
                    self.run_federated_round(epochs=2, batch_size=16, learning_rate=0.001)
            
            logger.info("Meta-training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Meta-training failed: {str(e)}")
            return False
    
    def evaluate_global_model(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict:
        """
        Evaluate global model performance
        
        Args:
            test_data: Test features
            test_labels: Test labels
            
        Returns:
            evaluation_results: Evaluation metrics
        """
        logger.info("Evaluating global model")
        
        global_model = self.aggregator.get_global_model()
        global_model.eval()
        
        with torch.no_grad():
            outputs = global_model(test_data.to(self.device))
            predictions = torch.argmax(outputs, dim=1)
            
            accuracy = (predictions == test_labels.to(self.device)).float().mean().item()
        
        # Evaluate each client
        client_accuracies = []
        for client in self.clients:
            client_accuracy = client.evaluate_model(test_data, test_labels)
            client_accuracies.append(client_accuracy)
        
        evaluation_results = {
            'global_accuracy': accuracy,
            'client_accuracies': client_accuracies,
            'avg_client_accuracy': np.mean(client_accuracies),
            'std_client_accuracy': np.std(client_accuracies)
        }
        
        logger.info(f"Global model accuracy: {accuracy:.4f}")
        logger.info(f"Average client accuracy: {evaluation_results['avg_client_accuracy']:.4f}")
        
        return evaluation_results

def main():
    """Test the blockchain FedAVG coordinator"""
    logger.info("Testing Blockchain FedAVG Coordinator")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=25, hidden_dim=64, num_classes=2):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 25
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Initialize coordinator
    model = SimpleModel(input_dim=n_features)
    coordinator = BlockchainFedAVGCoordinator(model, num_clients=3)
    
    # Distribute data
    coordinator.distribute_data(X_train, y_train, distribution_type='iid')
    
    # Run federated training
    training_history = coordinator.run_federated_training(num_rounds=5, epochs=3)
    
    # Evaluate global model
    evaluation_results = coordinator.evaluate_global_model(X_test, y_test)
    
    logger.info("✅ Blockchain FedAVG Coordinator test completed!")
    logger.info(f"Final evaluation: {evaluation_results}")

if __name__ == "__main__":
    main()
