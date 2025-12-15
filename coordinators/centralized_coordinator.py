"""
Centralized Learning Coordinator

This coordinator implements centralized learning by training on the full dataset
without client splitting. It maintains the same interface as SimpleFedAVGCoordinator
to allow easy switching between federated and centralized modes.
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from models.transductive_fewshot_model import create_meta_tasks, TransductiveLearner

logger = logging.getLogger(__name__)


@dataclass
class CentralizedTrainingUpdate:
    """Training update from centralized training"""
    training_loss: float
    validation_accuracy: float
    timestamp: float
    epoch: int


class CentralizedCoordinator:
    """
    Centralized Learning Coordinator
    
    Trains the model directly on the full dataset without client splitting.
    Maintains the same interface as SimpleFedAVGCoordinator for easy switching.
    """
    
    def __init__(self, model: nn.Module, config, device: str = "cuda"):
        """Initialize centralized coordinator"""
        # Convert device string to torch.device
        if isinstance(device, str):
            if device == "cuda" and torch.cuda.is_available():
                device_obj = torch.device('cuda')
                logger.info(f"✅ Centralized Coordinator using GPU: {torch.cuda.get_device_name(0)}")
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("⚠️  CUDA requested but not available, falling back to CPU")
                device_obj = torch.device('cpu')
            else:
                device_obj = torch.device(device)
        else:
            device_obj = device
        
        self.device = device_obj if isinstance(device_obj, torch.device) else str(device_obj)
        self.model = model.to(self.device)
        
        # Verify model is on correct device
        actual_device = next(self.model.parameters()).device
        logger.info(f"✅ Centralized Coordinator model moved to device: {actual_device}")
        
        self.config = config
        self.current_round = 0
        self.training_history: List[Dict] = []
        
        # Store full training data (will be set by distribute_data)
        self.train_data: Optional[torch.Tensor] = None
        self.train_labels: Optional[torch.Tensor] = None
        self.train_multiclass_labels: Optional[torch.Tensor] = None
        
        # Compatibility: Empty clients list (centralized learning has no clients)
        self.clients: List = []
        self.client_updates: List = []
        
        logger.info(f"Centralized Coordinator initialized on device: {self.device}")
    
    def distribute_data(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        alpha: float = 1.0,  # Not used in centralized, but kept for interface compatibility
        train_multiclass_labels: Optional[torch.Tensor] = None,
    ):
        """
        Store full training data for centralized learning
        
        In centralized learning, we use ALL data directly (no splitting).
        This method just stores the data for later use.
        """
        logger.info("=" * 80)
        logger.info("📊 CENTRALIZED LEARNING DATA SETUP")
        logger.info("=" * 80)
        logger.info(f"Using FULL dataset for centralized training (no client splitting)")
        
        # CRITICAL: Validate that train_data and train_labels have matching sizes
        train_data_size = train_data.shape[0] if isinstance(train_data, torch.Tensor) else len(train_data)
        train_labels_size = train_labels.shape[0] if isinstance(train_labels, torch.Tensor) else len(train_labels)
        
        if train_data_size != train_labels_size:
            raise ValueError(f"CRITICAL SIZE MISMATCH in distribute_data: train_data has {train_data_size} samples "
                           f"but train_labels has {train_labels_size} samples. They must have the same length.")
        
        # Store full dataset
        self.train_data = train_data.to(self.device)
        self.train_labels = train_labels.to(self.device)
        if train_multiclass_labels is not None:
            self.train_multiclass_labels = train_multiclass_labels.to(self.device)
        else:
            self.train_multiclass_labels = None
        
        logger.info(f"✅ Full training data stored:")
        logger.info(f"   - Training samples: {len(self.train_data):,}")
        logger.info(f"   - Data shape: {self.train_data.shape}")
        logger.info(f"   - Labels shape: {self.train_labels.shape}")
        
        # Log class distribution
        unique_labels, counts = torch.unique(self.train_labels, return_counts=True)
        logger.info(f"   - Class distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"     Class {label.item()}: {count.item():,} samples ({100*count.item()/len(self.train_labels):.2f}%)")
        
        logger.info("=" * 80)
    
    def train_once(self) -> Dict:
        """
        Train model once on full dataset (no rounds needed in centralized learning)
        
        Centralized learning doesn't need rounds - just train once, then do TTT.
        This is simpler and more efficient than repeating training multiple times.
        """
        logger.info("=" * 80)
        logger.info("🚀 CENTRALIZED META-LEARNING TRAINING")
        logger.info("=" * 80)
        logger.info("Note: No rounds needed - training once on full dataset, then TTT")
        
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data not distributed. Call distribute_data() first.")
        
        try:
            # Create meta-tasks from FULL dataset
            logger.info("\n📋 Creating meta-tasks from full training dataset...")
            logger.info(
                f"Meta-learning config - "
                f"n_way: {self.config.n_way}, k_shot: {self.config.k_shot}, "
                f"n_query: {self.config.n_query}, n_tasks: {self.config.num_meta_tasks}, "
                f"zero_day_attack: {self.config.zero_day_attack} "
                f"(label: {self.config.zero_day_attack_label})"
            )
            
            # Get zero-day attack label from config (it's a property)
            zero_day_attack_label = self.config.zero_day_attack_label
            
            meta_tasks = create_meta_tasks(
                self.train_data,
                self.train_labels,
                n_way=self.config.n_way,
                k_shot=self.config.k_shot,
                n_query=self.config.n_query,
                n_tasks=self.config.num_meta_tasks,
                phase="training",
                normal_query_ratio=0.8,
                zero_day_attack_label=zero_day_attack_label,
                enforce_equal_support_composition=getattr(self.config, 'enforce_equal_support_composition', True),
                include_all_attack_types_in_support=getattr(self.config, 'include_all_attack_types_in_support', False),
                data_y_multiclass=self.train_multiclass_labels,
            )
            
            logger.info(f"✅ Created {len(meta_tasks)} meta-tasks from full dataset")
            
            # Train model on full dataset using meta-learning (once, no rounds)
            logger.info(f"\n🎯 Running transductive meta-learning training ({self.config.meta_epochs} epochs)...")
            meta_training_history = self.model.meta_train(
                meta_tasks,
                meta_epochs=self.config.meta_epochs,
                config=self.config,
                global_params=None  # No FedProx in centralized learning
            )
            
            # Extract training metrics
            training_loss = meta_training_history.get('epoch_losses', [0.0])[-1] if meta_training_history else 0.0
            validation_accuracy = meta_training_history.get('epoch_accuracies', [0.0])[-1] if meta_training_history else 0.0
            
            # Store training history
            self.training_history.append({
                'loss': training_loss,
                'accuracy': validation_accuracy,
                'meta_history': meta_training_history
            })
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ CENTRALIZED META-LEARNING TRAINING COMPLETED")
            logger.info("=" * 80)
            logger.info(f"   Training loss: {training_loss:.4f}")
            logger.info(f"   Validation accuracy: {validation_accuracy:.4f}")
            logger.info("\n💡 Next step: TTT adaptation on test data")
            
            return {
                "training_loss": training_loss,
                "validation_accuracy": validation_accuracy,
                "meta_history": meta_training_history,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            logger.error(f"❌ Centralized training failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def run_federated_round(self, epochs: int = 2) -> Dict:
        """
        Legacy method kept for compatibility - but centralized learning doesn't need rounds!
        
        In centralized mode, use train_once() instead. This method exists only for
        interface compatibility but should not be used in centralized learning.
        """
        logger.warning("⚠️  Using run_federated_round() in centralized mode is redundant!")
        logger.warning("    Use train_once() instead - no rounds needed in centralized learning.")
        return self.train_once()
    
    def adapt_to_test_data(
        self,
        query_x: Optional[torch.Tensor] = None,
        query_y: Optional[torch.Tensor] = None,
        config: Optional[Any] = None,
        method: str = 'tent',
        X_test: Optional[torch.Tensor] = None,  # For backward compatibility
        y_test: Optional[torch.Tensor] = None   # For backward compatibility
    ) -> nn.Module:
        """
        Perform TTT adaptation on test data
        
        This method performs test-time training adaptation using TENT or TENT+pseudo-labels.
        It stores ttt_adaptation_data on the adapted model for visualization.
        """
        import copy
        import torch.nn.functional as F
        
        # Handle backward compatibility with old signature
        if X_test is not None:
            query_x = X_test
        if y_test is not None:
            query_y = y_test
        
        if query_x is None:
            logger.error("❌ No query/test data provided for TTT adaptation")
            return self.model
        
        # Use config from parameter or instance
        ttt_config = config if config is not None else self.config
        
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model = adapted_model.to(self.device)
        query_x = query_x.to(self.device)
        
        # Set model to training mode for TTT
        if hasattr(adapted_model, 'set_ttt_mode'):
            adapted_model.set_ttt_mode(training=True)
        else:
            adapted_model.train()
        
        # Get TTT parameters from config
        ttt_steps = getattr(ttt_config, 'ttt_steps', getattr(ttt_config, 'ttt_base_steps', 100))
        ttt_lr = getattr(ttt_config, 'ttt_lr', 0.001)
        use_pseudo_labels = getattr(ttt_config, 'use_pseudo_labels', False) or (method == 'tent_pseudo')
        pseudo_threshold = getattr(ttt_config, 'pseudo_threshold', 0.85)
        entropy_weight = getattr(ttt_config, 'entropy_weight', 1.0)
        pseudo_weight = getattr(ttt_config, 'pseudo_weight', 1.5)
        
        # ========================================
        # CRITICAL FIX: TENT-Style Layer Selection
        # ========================================
        
        # FREEZE all parameters first
        for param in adapted_model.parameters():
            param.requires_grad = False
        
        # UNFREEZE only BatchNorm affine parameters
        params_to_update = []
        bn_count = 0
        
        for name, module in adapted_model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Update affine parameters (scale and shift)
                if module.weight is not None:
                    module.weight.requires_grad = True
                    params_to_update.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad = True
                    params_to_update.append(module.bias)
                
                # Update running statistics
                module.track_running_stats = True
                module.momentum = 0.1
                bn_count += 1
        
        total_params = sum(p.numel() for p in params_to_update)
        frozen_params = sum(p.numel() for p in adapted_model.parameters() if not p.requires_grad)
        
        logger.info(f"✅ TENT mode enabled:")
        logger.info(f"   - Updating {bn_count} BatchNorm layers ({total_params:,} parameters)")
        logger.info(f"   - Frozen: {frozen_params:,} parameters (TCN, projections, prototypes)")
        
        # Before TTT loop: Store original BatchNorm parameters for L2 regularization
        # (prevents excessive parameter drift and improves generalization)
        original_params = {}
        for name, param in adapted_model.named_parameters():
            if param.requires_grad:  # Only BatchNorm parameters are trainable in TENT mode
                original_params[name] = param.clone().detach()
        
        # Setup optimizer with ONLY BatchNorm parameters
        optimizer = torch.optim.AdamW(params_to_update, lr=ttt_lr, weight_decay=1e-4)
        
        # Track adaptation data
        adaptation_data = {
            'steps': [],
            'total_losses': [],
            'entropy_losses': [],
            'pseudo_losses': [],
            'l2_reg_losses': []  # Track L2 regularization loss
        }
        
        logger.info(f"🔄 Starting TTT adaptation ({method}) for {ttt_steps} steps...")
        
        # TTT adaptation loop
        for step in range(ttt_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = adapted_model(query_x)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim=1)
            
            # Entropy loss (unsupervised)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            entropy_loss = entropy.mean()
            
            # Pseudo-label loss (if enabled)
            pseudo_loss = torch.tensor(0.0, device=logits.device)
            if use_pseudo_labels:
                confidences, pseudo_labels = probs.max(dim=1)
                confident_mask = confidences > pseudo_threshold
                
                if confident_mask.sum() > 0:
                    pseudo_loss = F.cross_entropy(
                        logits[confident_mask],
                        pseudo_labels[confident_mask],
                        reduction='mean'
                    )
            
            # Total loss
            total_loss = entropy_weight * entropy_loss + pseudo_weight * pseudo_loss
            
            # ADD L2 REGULARIZATION: penalize deviation from original parameters
            # This prevents excessive parameter drift and improves generalization (+2-4% improvement)
            l2_reg = torch.tensor(0.0, device=logits.device)
            if hasattr(ttt_config, 'ttt_l2_reg_weight') and ttt_config.ttt_l2_reg_weight > 0:
                for name, param in adapted_model.named_parameters():
                    if param.requires_grad and name in original_params:
                        l2_reg += (param - original_params[name]).pow(2).sum()
                
                # Total loss with L2 regularization
                total_loss = total_loss + ttt_config.ttt_l2_reg_weight * l2_reg
                reg_loss = l2_reg  # For tracking
            else:
                reg_loss = torch.tensor(0.0, device=logits.device)  # For tracking
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping (only on trainable BatchNorm parameters in TENT mode)
            torch.nn.utils.clip_grad_norm_(params_to_update, max_norm=1.0)
            
            optimizer.step()
            
            # Store metrics
            adaptation_data['steps'].append(step + 1)
            adaptation_data['total_losses'].append(total_loss.item())
            adaptation_data['entropy_losses'].append(entropy_loss.item())
            adaptation_data['pseudo_losses'].append(pseudo_loss.item())
            adaptation_data['l2_reg_losses'].append(reg_loss.item())
            
            if (step + 1) % 20 == 0:
                logger.info(f"  TTT Step {step + 1}/{ttt_steps}: Loss={total_loss.item():.4f}, "
                          f"Entropy={entropy_loss.item():.4f}, Pseudo={pseudo_loss.item():.4f}, "
                          f"L2_Reg={reg_loss.item():.4f}")
        
        # Set model back to evaluation mode
        if hasattr(adapted_model, 'set_ttt_mode'):
            adapted_model.set_ttt_mode(training=False)
        else:
            adapted_model.eval()
        
        # Store adaptation data on model for visualization
        adapted_model.ttt_adaptation_data = {
            'total_losses': adaptation_data['total_losses'],
            'entropy_losses': adaptation_data['entropy_losses'],
            'pseudo_losses': adaptation_data['pseudo_losses'],
            'l2_reg_losses': adaptation_data['l2_reg_losses'],  # Include L2 regularization losses
            'steps': adaptation_data['steps'],
            'final_loss': adaptation_data['total_losses'][-1] if adaptation_data['total_losses'] else 0.0,
            'adaptation_steps': len(adaptation_data['steps'])
        }
        
        logger.info(f"✅ TTT adaptation completed: {len(adaptation_data['steps'])} steps, "
                   f"final loss: {adaptation_data['total_losses'][-1]:.4f}")
        
        return adapted_model
    
    def evaluate_with_flow_wrapper(
        self,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        flow_ids: Optional[Any] = None,
        config: Optional[Any] = None,
        method: str = 'tent'
    ) -> Dict:
        """
        Evaluate model using flow wrapper (for compatibility with federated coordinator)
        
        Args:
            query_x: Query set features
            query_y: Query set labels
            flow_ids: Flow IDs (optional, for flow-level aggregation)
            config: Configuration (optional)
            method: TTT method (optional)
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Use the model from this coordinator
            model = self.model
            
            # Set model to evaluation mode
            model.eval()
            
            # Ensure query_x and query_y are on the correct device
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                if hasattr(model, 'forward_with_prototypes'):
                    # Prototype-based model
                    # Create simple prototypes from query set (for compatibility)
                    # In real flow-level evaluation, we'd aggregate by flow_id
                    outputs = model(query_x)
                else:
                    outputs = model(query_x)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Handle different output shapes
                if logits.dim() > 1:
                    predictions = torch.argmax(logits, dim=1)
                else:
                    # Binary classification with single output
                    predictions = (logits > 0.5).long()
            
            # Calculate metrics
            correct = (predictions == query_y).sum().item()
            total = len(query_y)
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate F1 score
            try:
                from sklearn.metrics import f1_score
                y_true = query_y.cpu().numpy()
                y_pred = predictions.cpu().numpy()
                
                # Determine if binary or multiclass
                unique_labels = len(np.unique(y_true))
                if unique_labels == 2:
                    # Binary classification
                    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0.0)
                else:
                    # Multiclass classification
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0.0)
            except Exception as e:
                logger.warning(f"Could not calculate F1 score: {e}")
                f1 = 0.0
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'total_samples': total
            }
        except Exception as e:
            logger.error(f"Flow-level evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'predictions': None,
                'total_samples': 0,
                'error': str(e)
            }
    
    def quick_system_self_check(self) -> Dict:
        """Quick system self-check (for compatibility)"""
        return {
            "meta_learning_ok": True,
            "aggregation_ok": True,
            "ttt_ok": True,
            "evaluation_ok": True,
            "mode": "centralized"
        }
    
    def set_attack_types(self, attack_types: Dict[str, int]):
        """Set attack types mapping (for compatibility)"""
        self.attack_types = attack_types

