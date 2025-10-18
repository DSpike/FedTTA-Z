"""
Simplified Federated Averaging Coordinator - Memory Optimized
"""
import torch
import torch.nn as nn
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimpleClientUpdate:
    """Simple client update without blockchain complexity"""
    client_id: str
    model_parameters: Dict[str, torch.Tensor]
    sample_count: int
    training_loss: float
    timestamp: float

class SimpleFedAVGCoordinator:
    """
    Simplified FedAVG Coordinator with aggressive memory management
    """
    
    def __init__(self, model: nn.Module, num_clients: int = 3, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.num_clients = num_clients
        self.current_round = 0
        self.clients = []
        
        # Initialize clients
        for i in range(num_clients):
            client = SimpleFederatedClient(f"client_{i+1}", model, device)
            self.clients.append(client)
        
        logger.info(f"Simple FedAVG Coordinator initialized with {num_clients} clients")
    
    def distribute_data(self, train_data: torch.Tensor, train_labels: torch.Tensor):
        """Distribute data among clients using simple splitting"""
        chunk_size = len(train_data) // self.num_clients
        
        for i, client in enumerate(self.clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_clients - 1 else len(train_data)
            
            client_data = train_data[start_idx:end_idx]
            client_labels = train_labels[start_idx:end_idx]
            
            client.set_training_data(client_data, client_labels)
            logger.info(f"Client {client.client_id}: {len(client_data)} samples")
    
    def run_federated_round(self, epochs: int = 2) -> Dict:
        """Run one federated learning round with minimal memory usage"""
        logger.info(f"Starting federated round {self.current_round + 1}")
        
        # Train all clients
        client_updates = []
        for client in self.clients:
            update = client.train_local_model(epochs)
            client_updates.append(update)
            
            # Clear GPU cache after each client
            torch.cuda.empty_cache()
        
        # Aggregate models directly without storing intermediate results
        self._aggregate_models_direct(client_updates)
        
        # Update clients with global model
        for client in self.clients:
            client.update_global_model(self.model.state_dict())
            torch.cuda.empty_cache()
        
        self.current_round += 1
        
        logger.info(f"Round {self.current_round} completed")
        return {
            'round': self.current_round,
            'client_updates': len(client_updates),
            'timestamp': time.time()
        }
    
    def _aggregate_models_direct(self, client_updates: List[SimpleClientUpdate]):
        """Direct aggregation without intermediate storage"""
        logger.info("Aggregating models directly")
        
        # Calculate total samples
        total_samples = sum(update.sample_count for update in client_updates)
        
        # Get global model parameters
        global_params = dict(self.model.named_parameters())
        
        # Update each parameter directly
        for param_name, global_param in global_params.items():
            # Initialize accumulator
            accumulator = torch.zeros_like(global_param.data)
            
            # Sum weighted client parameters
            for update in client_updates:
                if param_name in update.model_parameters:
                    weight = update.sample_count / total_samples
                    client_param = update.model_parameters[param_name].to(self.device)
                    accumulator += weight * client_param
                    
                    # Clear immediately
                    del client_param
            
            # Update global parameter
            with torch.no_grad():
                global_param.data.copy_(accumulator)
            
            # Clear accumulator
            del accumulator
            torch.cuda.empty_cache()
        
        logger.info("Direct aggregation completed")

class SimpleFederatedClient:
    """Simplified federated client with minimal memory usage"""
    
    def __init__(self, client_id: str, model: nn.Module, device: str = 'cuda'):
        import copy
        self.client_id = client_id
        self.device = device
        self.model = copy.deepcopy(model).to(device)  # FIXED: Each client gets independent model copy
        self.train_data = None
        self.train_labels = None
        
        
    def set_training_data(self, train_data: torch.Tensor, train_labels: torch.Tensor):
        """Set training data"""
        self.train_data = train_data.to(self.device)
        self.train_labels = train_labels.to(self.device)
    
    def train_local_model(self, epochs: int = 2) -> SimpleClientUpdate:
        """Train local model using meta-learning and TTT capabilities"""
        logger.info(f"Client {self.client_id}: Starting enhanced training for {epochs} epochs")
        
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
            meta_training_history = self.model.meta_train(local_meta_tasks, meta_epochs=epochs)
            
            # Phase 3: Fine-tuning with TTT capabilities
            logger.info(f"Client {self.client_id}: Running test-time training adaptation...")
            self._perform_local_ttt_adaptation()
            
            # Note: We don't replace self.model, we update it in place
            
            # Get model parameters (on CPU to save GPU memory)
            model_parameters = {}
            for name, param in self.model.named_parameters():
                model_parameters[name] = param.detach().cpu()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Calculate average loss from meta-training
            avg_loss = sum(meta_training_history['epoch_losses']) / len(meta_training_history['epoch_losses'])
            avg_accuracy = sum(meta_training_history['epoch_accuracies']) / len(meta_training_history['epoch_accuracies'])
            
            logger.info(f"Client {self.client_id}: Enhanced training completed - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            
            return SimpleClientUpdate(
                client_id=self.client_id,
                model_parameters=model_parameters,
                sample_count=len(self.train_data),
                training_loss=avg_loss,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Client {self.client_id}: Enhanced training failed, falling back to basic training: {str(e)}")
            return self._fallback_basic_training(epochs)
    
    def _perform_local_ttt_adaptation(self):
        """Perform test-time training adaptation on local data"""
        try:
            # Create a support set from a subset of local data
            support_size = min(100, len(self.train_data) // 4)
            support_indices = torch.randperm(len(self.train_data))[:support_size]
            support_x = self.train_data[support_indices]
            support_y = self.train_labels[support_indices]
            
            # Perform TTT adaptation directly on the model's meta_learner
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
    
    def _fallback_basic_training(self, epochs: int) -> SimpleClientUpdate:
        """Fallback to basic training if enhanced training fails"""
        logger.info(f"Client {self.client_id}: Using fallback basic training")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            batch_size = 32
            
            for i in range(0, len(self.train_data), batch_size):
                batch_data = self.train_data[i:i+batch_size]
                batch_labels = self.train_labels[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Clear cache every 100 batches
                if i % (batch_size * 100) == 0:
                    torch.cuda.empty_cache()
        
        # Get model parameters (on CPU to save GPU memory)
        model_parameters = {}
        for name, param in self.model.named_parameters():
            model_parameters[name] = param.detach().cpu()
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        logger.info(f"Client {self.client_id}: Fallback training completed")
        
        return SimpleClientUpdate(
            client_id=self.client_id,
            model_parameters=model_parameters,
            sample_count=len(self.train_data),
            training_loss=total_loss / (len(self.train_data) // 32),
            timestamp=time.time()
        )
    
    def update_global_model(self, global_parameters: Dict[str, torch.Tensor]):
        """Update local model with global parameters"""
        model_state_dict = self.model.state_dict()
        
        for param_name, param_tensor in global_parameters.items():
            if param_name in model_state_dict:
                with torch.no_grad():
                    model_state_dict[param_name].copy_(param_tensor.to(self.device))
        
        torch.cuda.empty_cache()
        logger.info(f"Client {self.client_id}: Updated with global model")
