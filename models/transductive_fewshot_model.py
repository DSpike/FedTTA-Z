#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, matthews_corrcoef
import copy
import math
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


        

class EmbeddingUtils:
    """
    Centralized utility class for embedding extraction and processing
    Eliminates redundancy across different model classes
    """
    
    @staticmethod
    def extract_embeddings(feature_extractors, feature_projection, self_attention, x):
        """
        Unified method for extracting and normalizing features with self-attention
        
        Args:
            feature_extractors: TCN-based multi-scale feature extractors (OptimizedMultiScaleTCN)
            feature_projection: Feature projection layer
            self_attention: Self-attention mechanism
            x: Input features (batch_size, sequence_length, input_dim)
            
        Returns:
            Normalized embeddings with self-attention applied
        """
        # Extract features using TCN-based multi-scale extractor
        # TCN expects input shape: (batch_size, sequence_length, input_dim)
        # Our input is already in the correct format: (batch_size, sequence_length, input_dim)
        
        # Extract multi-scale features using TCN
        combined_features = feature_extractors(x)  # (batch_size, tcn_output_dim)
        
        # Project to embedding space
        embeddings = feature_projection(combined_features)
        
        # Apply layer normalization
        embeddings = F.layer_norm(embeddings, embeddings.size()[1:])
        
        # Apply self-attention
        attended_embeddings, _ = self_attention(embeddings, embeddings, embeddings)
        
        return attended_embeddings

class PrototypeUtils:
    """
    Centralized utility class for prototype computation and updates
    Eliminates redundancy in prototype calculation across classes
    """
    
    @staticmethod
    def compute_prototypes(support_embeddings, support_y):
        """
        Compute class prototypes from support set embeddings
        
        Args:
            support_embeddings: Embeddings of support samples
            support_y: Labels of support samples
            
        Returns:
            prototypes: Class prototypes
            unique_labels: Unique class labels
        """
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            mask = support_y == label
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes), unique_labels
    
    @staticmethod
    def update_prototypes(test_embeddings, test_predictions, num_clusters=2, support_weight=0.0, test_weight=1.0):
        """
        ✅ UNSUPERVISED PROTOTYPE UPDATE: Uses k-means clustering on test embeddings only
        
        Args:
            test_embeddings: Test set embeddings
            test_predictions: Test set predictions (soft assignments)
            num_clusters: Number of clusters for k-means (default: 2 for binary classification)
            support_weight: Weight for support set contribution (default: 0.0 - no support data)
            test_weight: Weight for test set contribution (default: 1.0 - only test data)
            
        Returns:
            updated_prototypes: Updated class prototypes from k-means clustering
            cluster_labels: Cluster labels from k-means
        """
        from sklearn.cluster import KMeans
        import numpy as np
        
        # Convert to numpy for sklearn
        test_embeddings_np = test_embeddings.detach().cpu().numpy()
        
        # Perform k-means clustering on test embeddings
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(test_embeddings_np)
        cluster_labels = torch.tensor(cluster_labels, device=test_embeddings.device)
        
        # Calculate prototypes from k-means clusters
        updated_prototypes = []
        unique_clusters = torch.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            # Get embeddings for this cluster
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() > 0:
                cluster_embeddings = test_embeddings[cluster_mask]
                cluster_prototype = cluster_embeddings.mean(dim=0)
            else:
                raise ValueError("Empty cluster found during prototype computation")
            
            updated_prototypes.append(cluster_prototype)
        
        return torch.stack(updated_prototypes), unique_clusters

class LossUtils:
    """
    Centralized utility class for loss computation
    Breaks down complex loss calculation into reusable components
    """
    
    @staticmethod
    def compute_support_loss(support_embeddings, support_y, classifier):
        """Compute classification loss on support set with class weighting"""
        support_logits = classifier(support_embeddings)
        
        # Calculate class weights for imbalanced data
        class_counts = torch.bincount(support_y)
        total_samples = len(support_y)
        
        # Create weights for all possible classes (0-9) to match model output
        num_classes = support_logits.size(1)  # Get number of classes from model output
        class_weights = torch.ones(num_classes, device=support_y.device)
        
        # Set weights for classes present in the batch
        for class_id in range(num_classes):
            if class_id < len(class_counts) and class_counts[class_id] > 0:
                class_weights[class_id] = total_samples / (len(class_counts) * class_counts[class_id].float())
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        # Apply class weights
        return F.cross_entropy(support_logits, support_y, weight=class_weights)
    
    @staticmethod
    def compute_consistency_loss(test_embeddings, test_predictions, classifier):
        """Compute consistency loss on test set"""
        test_logits = classifier(test_embeddings)
        return F.kl_div(
            F.log_softmax(test_logits, dim=1),
            test_predictions,
            reduction='batchmean'
        )
    
    @staticmethod
    def compute_smoothness_loss(embeddings, temperature=0.1):
        """Compute graph smoothness loss"""
        # Compute pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t())
        similarities = similarities / temperature
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarities, dim=1)
        
        # Compute smoothness loss (encourage similar samples to have similar embeddings)
        smoothness_loss = torch.mean(torch.sum(attention_weights * torch.norm(embeddings.unsqueeze(1) - embeddings.unsqueeze(0), dim=2), dim=1))
        
        return smoothness_loss
    
    @staticmethod
    def compute_total_loss(support_embeddings, support_y, test_embeddings, test_predictions, 
                          prototypes, classifier, consistency_weight=0.1, smoothness_weight=0.01):
        """
        Compute total loss combining all components
        
        Args:
            support_embeddings: Support set embeddings
            support_y: Support set labels
            test_embeddings: Test set embeddings
            test_predictions: Test set predictions
            prototypes: Class prototypes
            classifier: Classification layer
            consistency_weight: Weight for consistency loss
            smoothness_weight: Weight for smoothness loss
            
        Returns:
            total_loss: Combined loss value
        """
        # Support loss
        support_loss = LossUtils.compute_support_loss(support_embeddings, support_y, classifier)
        
        # Consistency loss
        consistency_loss = LossUtils.compute_consistency_loss(test_embeddings, test_predictions, classifier)
        
        # Graph smoothness loss
        all_embeddings = torch.cat([support_embeddings, test_embeddings], dim=0)
        smoothness_loss = LossUtils.compute_smoothness_loss(all_embeddings)
        
        # Combine losses
        total_loss = support_loss + consistency_weight * consistency_loss + smoothness_weight * smoothness_loss
        
        return total_loss

class PredictionUtils:
    """
    Centralized utility class for prediction updates
    Eliminates redundancy in prediction logic across classes
    """
    
    @staticmethod
    def update_predictions_by_distance(test_embeddings, prototypes, temperature=2.0):
        """
        Update predictions using distance-based approach
        
        Args:
            test_embeddings: Test set embeddings
            prototypes: Class prototypes
            temperature: Temperature for softmax scaling
            
        Returns:
            predictions: Updated predictions
        """
        # Compute distances to prototypes
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        
        # Convert distances to probabilities with temperature scaling
        logits = -distances / temperature
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    @staticmethod
    def update_predictions_with_confidence(test_embeddings, prototypes, temperature=2.0):
        """
        Update predictions with confidence weighting
        
        Args:
            test_embeddings: Test set embeddings
            prototypes: Class prototypes
            temperature: Temperature for softmax scaling
            
        Returns:
            weighted_predictions: Confidence-weighted predictions
        """
        # Compute distances to prototypes
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        
        # Convert distances to probabilities with temperature scaling
        logits = -distances / temperature
        probabilities = F.softmax(logits, dim=1)
        
        # Apply confidence weighting
        confidence = torch.max(probabilities, dim=1)[0]
        confidence_weights = confidence.unsqueeze(1)
        
        # Weighted predictions
        weighted_predictions = probabilities * confidence_weights
        
        return weighted_predictions
    
class LoggingUtils:
    """
    Centralized utility class for standardized logging
    Eliminates redundancy in logging messages across classes
    """
    
    @staticmethod
    def log_ttt_step(step, loss, lr, consistency_weight, augmentation_type=None):
        """Log TTT step information with standardized format"""
        if augmentation_type:
            logger.info(f"TTT Step {step}: Applied {augmentation_type}")
        logger.info(f"Enhanced TTT Step {step}: Loss = {loss:.4f}, LR = {lr:.6f}, Consistency Weight = {consistency_weight:.4f}")
    
    @staticmethod
    def log_early_stopping(step, patience, best_loss, best_acc):
        """Log early stopping information"""
        logger.info(f"Early stopping at TTT step {step} (patience: {patience}, best_loss: {best_loss:.4f}, best_acc: {best_acc:.4f})")
    
    @staticmethod
    def log_adaptation_completion(steps, final_lr, dropout_layers):
        """Log TTT adaptation completion"""
        logger.info(f"✅ Enhanced test-time training adaptation completed in {steps} steps")
        logger.info(f"Final learning rate: {final_lr:.6f}")
        logger.info(f"TTT adaptation completed with dropout regularization: {dropout_layers} dropout layers")
    
    @staticmethod
    def log_model_mode(mode, dropout_layers):
        """Log model mode changes"""
        if mode == "training":
            logger.info(f"Model set to training mode for TTT adaptation (dropout active)")
            logger.info(f"TTT adaptation started with dropout regularization (p=0.3): {dropout_layers} dropout layers active")
        else:
            logger.info(f"Model set to evaluation mode for predictions (dropout disabled)")
            logger.info(f"TTT model evaluation started in evaluation mode (dropout disabled): {dropout_layers} dropout layers")

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25 for normal class)
        gamma: Focusing parameter (default: 2)
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, weight=None):
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        
        # Compute probability of true class
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TransductiveLearner(nn.Module):
    """
    Optimized Transductive Learning for Zero-Day Detection
    Streamlined implementation with unified methods for better maintainability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3, sequence_length: int = 1):
        super(TransductiveLearner, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes  # Now supports 10 classes for UNSW-NB15
        self.support_weight = support_weight
        self.test_weight = test_weight
        
        # TTT parameters (will be updated from config)
        self.ttt_threshold = 0.5
        
        # TTT parameters (will be updated from config)
        self.ttt_lr = 0.0005
        self.ttt_steps = 100
        
        # Multi-scale TCN feature extractors for temporal pattern recognition
        from .optimized_tcn_module import OptimizedMultiScaleTCN
        
        self.feature_extractors = OptimizedMultiScaleTCN(
            input_dim=input_dim,
            sequence_length=sequence_length,  # Use configurable sequence length
            hidden_dim=hidden_dim,
            dropout=0.2
        )
        
        # Feature projection to embedding space (TCN output: hidden_dim + hidden_dim//2 + hidden_dim*2)
        tcn_output_dim = hidden_dim + (hidden_dim // 2) + (hidden_dim * 2)
        self.feature_projection = nn.Sequential(
            nn.Linear(tcn_output_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),  # Added for TENT compatibility
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Enhanced classification network for better handling of imbalanced data
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Attention mechanism for global context
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=2)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Enhanced transductive learning parameters for better convergence
        self.transductive_lr = 0.01  # Increased learning rate for faster convergence
        self.transductive_steps = 50  # Increased steps for better adaptation
        
        # Meta-learner compatibility (for TTT adaptation)
        # Note: meta_learner will be set after initialization to avoid recursion
        
        # Initialize weights for better learning on imbalanced data
        self._initialize_weights()
    
    @property
    def meta_learner(self):
        """Meta-learner compatibility property"""
        return self
    
    
    
    def get_confidence_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates confidence scores for the input samples.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            confidence_scores: Confidence scores for each sample
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        
        return confidence_scores

    def get_confidence_and_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates confidence scores and returns class probabilities.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            confidence_scores: Confidence scores for each sample
            probabilities: Probabilities per class (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
        return confidence_scores, probabilities
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the model
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            embeddings: Feature embeddings of shape (batch_size, embedding_dim)
        """
        return self.extract_embeddings(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the model (alias for extract_embeddings for TTT compatibility)
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            features: Feature embeddings of shape (batch_size, embedding_dim)
        """
        return self.extract_embeddings(x)
    
    
    def get_dropout_status(self):
        """
        Get current dropout status of the model
        Returns:
            dropout_layers: List of dropout layers and their probabilities
        """
        dropout_layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(f"{name}: p={module.p}")
        return dropout_layers
    
        
    def forward(self, x):
        """
        Forward pass: Extract embeddings and get logits
        """
        embeddings = self.extract_embeddings(x)
        logits = self.classifier(embeddings)
        return logits
    
    def extract_embeddings(self, x):
        """
        Unified method for extracting and normalizing features with self-attention
        Now uses centralized utility for consistency
        """
        return EmbeddingUtils.extract_embeddings(
            self.feature_extractors, 
            self.feature_projection, 
            self.self_attention, 
            x
        )
    

    
    def _initialize_weights(self):
        """
        Initialize weights for better learning on imbalanced data
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    
    def transductive_optimization(self, support_x, support_y, test_x, test_y=None):
        """
        Main transductive optimization method
        """
        device = next(self.parameters()).device
        
        # Move data to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        test_x = test_x.to(device)
        
        # Compute support and test embeddings
        support_embeddings = self.extract_embeddings(support_x)
        test_embeddings = self.extract_embeddings(test_x)
        
        # Compute prototypes
        prototypes, unique_labels = self.update_prototypes(support_embeddings, support_y, test_embeddings, None)
        
        # Initialize test predictions
        test_predictions = self.update_test_predictions(test_embeddings, prototypes)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.parameters(), lr=self.transductive_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
        
        best_loss = float('inf')
        patience_counter = 0
        
        # Transductive optimization loop
        for step in range(self.transductive_steps):
            optimizer.zero_grad()
            
            # Recompute embeddings (they change during optimization)
            support_embeddings = self.extract_embeddings(support_x)
            test_embeddings = self.extract_embeddings(test_x)
            
            # Update prototypes
            prototypes = self.update_prototypes(support_embeddings, support_y, test_embeddings, test_predictions)
            
            # Compute total loss
            total_loss = self.compute_loss(support_embeddings, support_y, test_embeddings, test_predictions, prototypes)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(total_loss)
            
            # Update test predictions
            test_predictions = self.update_test_predictions(test_embeddings, prototypes)
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 8:
                logger.info(f"Early stopping at step {step} (patience: 8, best_loss: {best_loss:.4f})")
                break
            
            if step % 5 == 0:
                logger.info(f"Transductive step {step}: Loss = {total_loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        return test_predictions, prototypes, unique_labels
    



    # true_ttt_adaptation method removed - now handled by FedAVG coordinator    
    
    def update_prototypes(self, support_embeddings, support_y, test_embeddings, test_predictions=None, num_clusters=2):
        """
        ✅ UNSUPERVISED prototype update method using k-means clustering
        Now uses centralized utility for consistency with unsupervised approach
        """
        return PrototypeUtils.update_prototypes(
            test_embeddings, test_predictions, num_clusters=num_clusters,
            support_weight=0.0, test_weight=1.0
        )
    
    def compute_loss(self, support_embeddings, support_y, test_embeddings, test_predictions, prototypes):
        """
        Unified loss computation method combining all loss components
        Now uses centralized utility for consistency
        """
        return LossUtils.compute_total_loss(
            support_embeddings, support_y, test_embeddings, test_predictions, 
            prototypes, self.classifier, consistency_weight=0.1, smoothness_weight=0.01
        )
    
    def update_test_predictions(self, test_embeddings, prototypes):
        """
        Unified method for updating test predictions using distance and confidence
        Now uses centralized utility for consistency
        """
        return PredictionUtils.update_predictions_with_confidence(test_embeddings, prototypes)
    
    def meta_train(self, meta_tasks: List[Dict], meta_epochs: int = 100):
        """
        Meta-train the model on multiple tasks
        
        Args:
            meta_tasks: List of meta-learning tasks
            meta_epochs: Number of meta-training epochs
            
        Returns:
            training_history: Training metrics
        """
        logger.info(f"Starting transductive meta-training for {meta_epochs} epochs")
        
        training_history = {
            'epoch_losses': [],
            'epoch_accuracies': []
        }
        
        # Enhanced optimizer for better convergence on imbalanced data
        meta_optimizer = optim.AdamW(self.parameters(), lr=0.01, weight_decay=1e-4)
        
        
        # Initialize focal loss function
        focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        for epoch in range(meta_epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            # Sample tasks for this epoch
            np.random.shuffle(meta_tasks)
            
            for task in meta_tasks:
                # Move tensors to the same device as the model
                device = next(self.parameters()).device
                support_x = task['support_x'].to(device)
                support_y = task['support_y'].to(device)
                query_x = task['query_x'].to(device)
                query_y = task['query_y'].to(device)
                
                # Forward pass on support set
                support_logits = self(support_x)
                
                # Calculate class weights for support set
                support_class_counts = torch.bincount(support_y)
                support_total = len(support_y)
                num_classes = support_logits.size(1)
                
                # Create stronger weights for class imbalance handling
                support_class_weights = torch.ones(num_classes, device=support_y.device)
                for class_id in range(num_classes):
                    if class_id < len(support_class_counts) and support_class_counts[class_id] > 0:
                        # Use inverse frequency with square root to reduce extreme weights
                        support_class_weights[class_id] = torch.sqrt(support_total / support_class_counts[class_id].float())
                    else:
                        # Give very high weight to missing classes to encourage learning
                        support_class_weights[class_id] = support_total * 2.0
                
                # Normalize weights but keep them strong
                support_class_weights = support_class_weights / support_class_weights.sum() * num_classes * 2.0
                
                # Use focal loss for better handling of hard examples
                support_loss = focal_loss_fn(support_logits, support_y, weight=support_class_weights)
                
                # Forward pass on query set
                query_logits = self(query_x)
                
                # Calculate stronger class weights for query set
                query_class_counts = torch.bincount(query_y)
                query_total = len(query_y)
                query_class_weights = torch.ones(num_classes, device=query_y.device)
                for class_id in range(num_classes):
                    if class_id < len(query_class_counts) and query_class_counts[class_id] > 0:
                        # Use inverse frequency with square root to reduce extreme weights
                        query_class_weights[class_id] = torch.sqrt(query_total / query_class_counts[class_id].float())
                    else:
                        # Give very high weight to missing classes to encourage learning
                        query_class_weights[class_id] = query_total * 2.0
                
                # Normalize weights but keep them strong
                query_class_weights = query_class_weights / query_class_weights.sum() * num_classes * 2.0
                
                # Use focal loss for better handling of hard examples
                query_loss = focal_loss_fn(query_logits, query_y, weight=query_class_weights)
                
                # Total loss
                total_loss = support_loss + query_loss
                
                # Compute accuracy
                predictions = torch.argmax(query_logits, dim=1)
                accuracy = (predictions == query_y).float().mean().item()
                
                # Backward pass
                meta_optimizer.zero_grad()
                total_loss.backward()
                meta_optimizer.step()
                
                epoch_losses.append(total_loss.item())
                epoch_accuracies.append(accuracy)
            
            # Record epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)
            
            training_history['epoch_losses'].append(avg_loss)
            training_history['epoch_accuracies'].append(avg_accuracy)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        logger.info("Transductive meta-training completed")
        return training_history
   # TTT methods removed - now handled by FedAVG coordinator
class MetaLearner(nn.Module):
    """
    Meta-Learning model for few-shot adaptation with transductive learning
    Learns to quickly adapt to new tasks with minimal examples
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3, sequence_length: int = 1):
        super(MetaLearner, self).__init__()
        
        self.transductive_net = TransductiveLearner(input_dim, hidden_dim, embedding_dim, num_classes, support_weight, test_weight, sequence_length)
        self.meta_optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Meta-learning parameters
        self.inner_lr = 0.01
        self.inner_steps = 5
        
    def forward(self, x):
        return self.transductive_net(x)
    
    def get_embeddings(self, x):
        """
        Extract embeddings from the transductive network
        Now uses centralized utility for consistency
        """
        return self.transductive_net.extract_embeddings(x)
    
    def meta_update(self, support_x, support_y, query_x, query_y):
        """
        Perform meta-update using support and query sets with transductive learning
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            query_y: Query set labels
            
        Returns:
            loss: Meta-learning loss
        """
        # Get embeddings
        support_embeddings = self.transductive_net.extract_embeddings(support_x)
        query_embeddings = self.transductive_net.extract_embeddings(query_x)
        
        # Compute prototypes
        prototypes, prototype_labels = self.transductive_net.update_prototypes(
            support_embeddings, support_y, query_embeddings, None
        )
        
        # Classify query samples using distance-based classification
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        predictions = prototype_labels[torch.argmin(distances, dim=1)]
        
        # Compute loss using Focal Loss for better class imbalance handling
        logits = -distances
        focal_loss = FocalLoss(alpha=1, gamma=2, reduction='mean')
        loss = focal_loss(logits, query_y)
        
        return loss, predictions
    

class TransductiveFewShotModel(nn.Module):
    """
    Transductive Few-Shot Model for Zero-Day Detection
    Combines meta-learning with test-time training for rapid adaptation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3, sequence_length: int = 1):
        super(TransductiveFewShotModel, self).__init__()
        
        self.meta_learner = MetaLearner(input_dim, hidden_dim, embedding_dim, num_classes, support_weight, test_weight, sequence_length)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Zero-day detection parameters
        self.anomaly_threshold = 0.5
        
    def forward(self, x):
        return self.meta_learner(x)
    
    def get_embeddings(self, x):
        """
        Extract embeddings from the model
        Now uses centralized utility for consistency
        """
        return self.meta_learner.get_embeddings(x)
    
    
    def get_dropout_status(self):
        """
        Get current dropout status for logging
        
        Returns:
            dict: Status of dropout layers in the model
        """
        dropout_status = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                dropout_status[name] = {
                    'p': module.p,
                    'training': module.training
                }
        return dropout_status
    
    def compute_confidence(self, embeddings, prototypes, prototype_labels):
        """
        Compute confidence scores for predictions
        
        Args:
            embeddings: Sample embeddings
            prototypes: Class prototypes
            prototype_labels: Prototype labels
            
        Returns:
            confidence: Confidence scores
        """
        distances = torch.cdist(embeddings, prototypes, p=2)
        min_distances = torch.min(distances, dim=1)[0]
        max_distances = torch.max(distances, dim=1)[0]
        
        # Confidence based on distance ratio
        confidence = 1.0 - (min_distances / (max_distances + 1e-8))
        return confidence
    

    
    # Removed duplicate private loss helpers; active path uses LossUtils
    
    # Removed wrapper meta_train; use TransductiveLearner.meta_train instead
    
    # Removed evaluate_zero_day_detection: adaptation is handled by the coordinator

    def validate_data_splits(self, train_x, train_y, val_x, val_y, test_x, test_y):
        """
        Validate that data splits don't have overlap to prevent data leakage
        
        Args:
            train_x, train_y: Training data
            val_x, val_y: Validation data  
            test_x, test_y: Test data
            
        Returns:
            is_valid: Boolean indicating if splits are valid
            overlap_info: Dictionary with overlap details
        """
        logger.info("🔍 Validating data splits to prevent data leakage...")
        
        overlap_info = {
            'train_val_overlap': 0,
            'train_test_overlap': 0,
            'val_test_overlap': 0,
            'total_overlaps': 0
        }
        
        # Convert to numpy for comparison
        train_x_np = train_x.detach().cpu().numpy() if torch.is_tensor(train_x) else train_x
        val_x_np = val_x.detach().cpu().numpy() if torch.is_tensor(val_x) else val_x
        test_x_np = test_x.detach().cpu().numpy() if torch.is_tensor(test_x) else test_x
        
        # Check for exact duplicates between splits
        def find_overlaps(data1, data2, name1, name2):
            overlaps = 0
            for i, sample1 in enumerate(data1):
                for j, sample2 in enumerate(data2):
                    if np.array_equal(sample1, sample2):
                        overlaps += 1
                        logger.warning(f"Overlap found: {name1}[{i}] == {name2}[{j}]")
            return overlaps
        
        # Check all pairwise overlaps
        overlap_info['train_val_overlap'] = find_overlaps(train_x_np, val_x_np, 'train', 'val')
        overlap_info['train_test_overlap'] = find_overlaps(train_x_np, test_x_np, 'train', 'test')
        overlap_info['val_test_overlap'] = find_overlaps(val_x_np, test_x_np, 'val', 'test')
        
        overlap_info['total_overlaps'] = (overlap_info['train_val_overlap'] + 
                                        overlap_info['train_test_overlap'] + 
                                        overlap_info['val_test_overlap'])
        
        is_valid = overlap_info['total_overlaps'] == 0
        
        if is_valid:
            logger.info("✅ Data splits are valid - no overlaps detected")
        else:
            logger.error(f"❌ Data leakage detected! Total overlaps: {overlap_info['total_overlaps']}")
            logger.error(f"  Train-Val overlaps: {overlap_info['train_val_overlap']}")
            logger.error(f"  Train-Test overlaps: {overlap_info['train_test_overlap']}")
            logger.error(f"  Val-Test overlaps: {overlap_info['val_test_overlap']}")
        
        return is_valid, overlap_info

def create_meta_tasks(data_x, data_y, n_way: int = 2, k_shot: int = 5, n_query: int = 15, n_tasks: int = 100, 
                     phase: str = "training", normal_query_ratio: float = 0.8, zero_day_attack_label: int = None):
    """
    Create meta-learning tasks for few-shot learning with controlled query set distribution
    
    Args:
        data_x: Input data
        data_y: Labels
        n_way: Number of classes per task
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        n_tasks: Number of tasks to create
        phase: Phase of learning ("training", "validation", "testing")
        normal_query_ratio: Ratio of normal samples in query set (0.8 for training/validation, 0.9 for testing)
        zero_day_attack_label: Label of zero-day attack to exclude from training (None for testing phase)
        
    Returns:
        meta_tasks: List of meta-learning tasks
    """
    logger.info(f"Creating {n_tasks} meta-learning tasks ({n_way}-way, {k_shot}-shot) for {phase} phase")
    logger.info(f"Query set will have {normal_query_ratio*100:.0f}% Normal samples")
    if zero_day_attack_label is not None:
        logger.info(f"Excluding zero-day attack (label {zero_day_attack_label}) from training")
    
    meta_tasks = []
    unique_labels = torch.unique(data_y)
    
    # For training phase, exclude zero-day attack if specified
    if phase in ["training", "validation"] and zero_day_attack_label is not None:
        # Filter out zero-day attack from available labels
        available_labels = unique_labels[unique_labels != zero_day_attack_label]
        logger.info(f"Available labels for {phase}: {available_labels.tolist()}")
    else:
        available_labels = unique_labels
        logger.info(f"Available labels for {phase}: {available_labels.tolist()}")
    
    # Separate Normal (0) and Attack samples
    normal_mask = data_y == 0
    normal_indices = torch.where(normal_mask)[0]
    
    # For attack samples, exclude zero-day attack if specified
    if zero_day_attack_label is not None:
        attack_mask = (data_y != 0) & (data_y != zero_day_attack_label)
    else:
        attack_mask = data_y != 0
    attack_indices = torch.where(attack_mask)[0]
    
    for _ in range(n_tasks):
        # Sample classes for this task from available labels
        task_classes = torch.randperm(len(available_labels))[:n_way]
        selected_labels = available_labels[task_classes]
        
        # Create support set for each selected class
        support_x_list = []
        support_y_list = []
        
        for label in selected_labels:
            # Get samples for this class
            class_mask = data_y == label
            class_indices = torch.where(class_mask)[0]
            
            # Shuffle and select samples for support set
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]
            support_indices = shuffled_indices[:k_shot]
            
            support_x_list.append(data_x[support_indices])
            support_y_list.append(data_y[support_indices])
            
        # Combine support sets
        support_x = torch.cat(support_x_list, dim=0)
        support_y = torch.cat(support_y_list, dim=0)
        
        # SCIENTIFIC FIX: Use natural class distribution instead of artificial ratios
        # Sample query set with realistic distribution based on available data
        total_query_samples = n_query * n_way
        
        # Calculate natural distribution from available data
        total_available = len(normal_indices) + len(attack_indices)
        if total_available > 0:
            natural_normal_ratio = len(normal_indices) / total_available
            natural_attack_ratio = len(attack_indices) / total_available
        else:
            natural_normal_ratio = 0.5
            natural_attack_ratio = 0.5
        
        # Sample query set maintaining natural distribution
        target_normal_count = int(total_query_samples * natural_normal_ratio)
        target_attack_count = total_query_samples - target_normal_count
        
        # Sample normal samples for query set (from all available normal samples)
        if len(normal_indices) >= target_normal_count:
            normal_query_indices = normal_indices[torch.randperm(len(normal_indices))[:target_normal_count]]
        else:
            normal_query_indices = normal_indices
        
        # Sample attack samples for query set (from all available attack samples, excluding zero-day if specified)
        if len(attack_indices) >= target_attack_count:
            attack_query_indices = attack_indices[torch.randperm(len(attack_indices))[:target_attack_count]]
        else:
            attack_query_indices = attack_indices
        
        # Combine query samples
        if len(normal_query_indices) > 0 and len(attack_query_indices) > 0:
            query_indices = torch.cat([normal_query_indices, attack_query_indices])
        elif len(normal_query_indices) > 0:
            query_indices = normal_query_indices
        elif len(attack_query_indices) > 0:
            query_indices = attack_query_indices
        else:
            raise ValueError("Insufficient samples for query set creation")
        
        # Shuffle query indices
        query_indices = query_indices[torch.randperm(len(query_indices))]
        
        # Create query set
        query_x = data_x[query_indices]
        query_y = data_y[query_indices]
        
        # Verify query set distribution
        query_normal_count = (query_y == 0).sum().item()
        query_attack_count = (query_y != 0).sum().item()
        total_query = len(query_y)
        actual_normal_ratio = query_normal_count / total_query if total_query > 0 else 0
        
        logger.debug(f"Query set distribution: {query_normal_count}/{total_query} Normal ({actual_normal_ratio:.1%}), target: {normal_query_ratio:.1%}")
        
        # SCIENTIFIC FIX: Preserve original labels instead of arbitrary relabeling
        # This maintains semantic meaning and class relationships
        logger.debug(f"Preserving original labels for task {len(meta_tasks)}: {selected_labels.tolist()}")
        
        meta_tasks.append({
            'support_x': support_x,
            'support_y': support_y,  # ✅ Original labels preserved
            'query_x': query_x,
            'query_y': query_y,       # ✅ Original labels preserved
            'selected_labels': selected_labels,  # Track which classes are in this task
            'label_mapping': {label.item(): label.item() for label in selected_labels}  # Identity mapping
        })
    
    logger.info(f"Created {len(meta_tasks)} meta-learning tasks")
    return meta_tasks

def main():
    """Test the transductive few-shot model"""
    logger.info("Testing Transductive Few-Shot Model")
    
    # Create synthetic data for testing
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 25
    
    # Generate synthetic data
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Initialize model
    model = TransductiveFewShotModel(input_dim=n_features)
    
    # Validate data splits to prevent data leakage
    is_valid, overlap_info = model.validate_data_splits(X_train, y_train, X_val, y_val, X_test, y_test)
    if not is_valid:
        logger.error("Data leakage detected! Cannot proceed with evaluation.")
        return
    
    # Create meta-tasks
    meta_tasks = create_meta_tasks(X_train, y_train, n_tasks=50)
    
    # Meta-train the underlying transductive learner
    training_history = model.meta_learner.transductive_net.meta_train(meta_tasks, meta_epochs=20)
    
    # Simple base evaluation (no TTT here; adaptation is coordinator-side)
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_test).float().mean().item()
    logger.info("✅ Transductive few-shot model test completed!")
    logger.info(f"Final base accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()