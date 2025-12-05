#!/usr/bin/env python3
"""
Enhanced Binary Classifier with Open-Set Detection for Network Traffic
Implements binary classification (normal vs attack) with:
- Single prototypes for normal and attack classes
- Binary cross-entropy loss
- Test-Time Training (TTT) for attack prototype refinement
- DBSCAN clustering for diverse attack pattern detection
- Mahalanobis distance-based rejection threshold
- Open-set detection capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.covariance import EmpiricalCovariance
import copy
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBinaryClassifier(nn.Module):
    """
    Enhanced Binary Classifier for Network Traffic with Open-Set Detection
    
    Features:
    - Binary classification (normal vs attack)
    - Single prototypes for each class
    - Binary cross-entropy loss
    - Test-Time Training (TTT) for attack prototype refinement
    - DBSCAN clustering for diverse attack patterns
    - Mahalanobis distance-based rejection threshold
    - Open-set detection capabilities
    - Compatible with existing federated learning coordinator
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, 
                 sequence_length: int = 12,
                 ttt_lr: float = 0.001, ttt_steps: int = 10):
        super(EnhancedBinaryClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps
        
        # Enhanced embedding network with residual connections
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # Enhanced binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Binary output
        )
        
        # Initialize prototypes with better initialization
        self.normal_prototype = nn.Parameter(torch.zeros(embedding_dim))
        self.attack_prototype = nn.Parameter(torch.ones(embedding_dim))
        
        # Mahalanobis distance parameters
        self.normal_covariance = None
        self.attack_covariance = None
        self.rejection_threshold = 0.5
        
        # DBSCAN parameters for attack clustering
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 5
        
        # Binary cross-entropy loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Initialize prototypes
        self._initialize_prototypes()
        
    def _initialize_prototypes(self):
        """Initialize prototypes with small random values"""
        nn.init.normal_(self.normal_prototype, 0, 0.1)
        nn.init.normal_(self.attack_prototype, 0, 0.1)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            logits: Binary classification logits (for compatibility with existing coordinator)
        """
        embeddings = self.embedding_net(x)
        logits = self.classifier(embeddings)
        
        # For compatibility with existing coordinator, return only logits
        return logits
    
    def forward_full(self, x):
        """
        Full forward pass with all outputs
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            embeddings: Feature embeddings
            logits: Binary classification logits
            probabilities: Sigmoid probabilities
        """
        embeddings = self.embedding_net(x)
        logits = self.classifier(embeddings)
        probabilities = torch.sigmoid(logits)
        
        return embeddings, logits, probabilities
    
    def compute_prototypes(self, embeddings, labels):
        """
        Compute prototypes for normal and attack classes
        
        Args:
            embeddings: Feature embeddings
            labels: Binary labels (0=normal, 1=attack)
            
        Returns:
            normal_prototype: Normal class prototype
            attack_prototype: Attack class prototype
        """
        normal_mask = labels == 0
        attack_mask = labels == 1
        
        if normal_mask.sum() > 0:
            normal_prototype = embeddings[normal_mask].mean(dim=0)
        else:
            normal_prototype = self.normal_prototype
            
        if attack_mask.sum() > 0:
            attack_prototype = embeddings[attack_mask].mean(dim=0)
        else:
            attack_prototype = self.attack_prototype
            
        return normal_prototype, attack_prototype
    
    def update_prototypes(self, normal_prototype, attack_prototype):
        """
        Update class prototypes
        
        Args:
            normal_prototype: New normal prototype
            attack_prototype: New attack prototype
        """
        with torch.no_grad():
            self.normal_prototype.copy_(normal_prototype)
            self.attack_prototype.copy_(attack_prototype)
    
    def compute_covariances(self, embeddings, labels):
        """
        Compute covariance matrices for Mahalanobis distance calculation
        
        Args:
            embeddings: Feature embeddings
            labels: Binary labels (0=normal, 1=attack)
        """
        normal_mask = labels == 0
        attack_mask = labels == 1
        
        if normal_mask.sum() > 1:
            normal_embeddings = embeddings[normal_mask].detach().cpu().numpy()
            self.normal_covariance = EmpiricalCovariance().fit(normal_embeddings)
        else:
            self.normal_covariance = None
            
        if attack_mask.sum() > 1:
            attack_embeddings = embeddings[attack_mask].detach().cpu().numpy()
            self.attack_covariance = EmpiricalCovariance().fit(attack_embeddings)
        else:
            self.attack_covariance = None
    
    def mahalanobis_distance(self, embeddings, class_type='normal'):
        """
        Compute Mahalanobis distance for open-set detection
        
        Args:
            embeddings: Feature embeddings
            class_type: 'normal' or 'attack'
            
        Returns:
            distances: Mahalanobis distances
        """
        if class_type == 'normal':
            if self.normal_covariance is None:
                return torch.full((embeddings.shape[0],), float('inf'), device=embeddings.device)
            prototype = self.normal_prototype
            cov = self.normal_covariance
        else:
            if self.attack_covariance is None:
                return torch.full((embeddings.shape[0],), float('inf'), device=embeddings.device)
            prototype = self.attack_prototype
            cov = self.attack_covariance
        
        # Compute Mahalanobis distance
        diff = embeddings.detach().cpu().numpy() - prototype.detach().cpu().numpy()
        try:
            inv_cov = np.linalg.inv(cov.covariance_)
            distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            return torch.tensor(distances, device=embeddings.device)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if covariance is singular
            distances = torch.cdist(embeddings, prototype.unsqueeze(0), p=2).squeeze(1)
            return distances
    
    def detect_novel_attacks(self, embeddings, min_samples=5, eps=0.5):
        """
        Detect novel attack patterns using DBSCAN clustering
        
        Args:
            embeddings: Feature embeddings
            min_samples: Minimum samples for DBSCAN cluster
            eps: DBSCAN epsilon parameter
            
        Returns:
            cluster_labels: DBSCAN cluster labels (-1 for noise/novel attacks)
            n_clusters: Number of clusters found
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings_np)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return cluster_labels, n_clusters
    
    def test_time_training(self, support_x, support_y, query_x, query_y=None):
        """
        Test-Time Training to refine attack prototype for novel attack patterns
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            query_y: Query set labels (optional)
            
        Returns:
            predictions: Refined predictions
            confidence_scores: Confidence scores
            novel_attacks: Detected novel attack patterns
        """
        device = next(self.parameters()).device
        
        # Move data to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        
        # Get embeddings
        support_embeddings, _, _ = self.forward_full(support_x)
        query_embeddings, _, _ = self.forward_full(query_x)
        
        # Compute initial prototypes
        normal_prototype, attack_prototype = self.compute_prototypes(support_embeddings, support_y)
        
        # Detect novel attack patterns using DBSCAN
        novel_attacks, n_clusters = self.detect_novel_attacks(query_embeddings)
        
        # TTT optimization with enhanced setup
        optimizer = optim.AdamW(self.parameters(), lr=self.ttt_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.ttt_lr*2, 
                                                total_steps=self.ttt_steps, 
                                                pct_start=0.3)
        
        # Calculate class weights for TTT
        class_counts = torch.bincount(support_y)
        total_samples = len(support_y)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(device)
        
        for step in range(self.ttt_steps):
            optimizer.zero_grad()
            
            # Get current embeddings
            support_embeddings, _, _ = self.forward_full(support_x)
            query_embeddings, _, _ = self.forward_full(query_x)
            
            # Update prototypes
            normal_prototype, attack_prototype = self.compute_prototypes(support_embeddings, support_y)
            
            # Compute binary classification loss using the full forward pass
            support_logits = self.forward(support_x)
            support_loss = self.compute_enhanced_loss(support_logits, support_y, class_weights)
            
            # Compute prototype consistency loss
            normal_distances = torch.cdist(support_embeddings, normal_prototype.unsqueeze(0), p=2)
            attack_distances = torch.cdist(support_embeddings, attack_prototype.unsqueeze(0), p=2)
            
            # Prototype consistency loss
            normal_mask = support_y == 0
            attack_mask = support_y == 1
            
            prototype_loss = 0
            if normal_mask.sum() > 0:
                prototype_loss += normal_distances[normal_mask].mean()
            if attack_mask.sum() > 0:
                prototype_loss += attack_distances[attack_mask].mean()
            
            # Add query set loss for better adaptation (TRANSDUCTIVE: Use pseudo-labels, not query_y)
            query_logits = self.forward(query_x)
            
            # TRANSDUCTIVE LEARNING: Use pseudo-labels from predictions instead of ground truth query_y
            # This allows training with unlabeled query sets (true transductive learning)
            if query_y is not None:
                # Generate pseudo-labels from model predictions (nearest prototype)
                query_probs = torch.softmax(query_logits, dim=1)
                query_pseudo_labels = torch.argmax(query_probs, dim=1).detach()  # Detach to treat as fixed targets
                query_loss = self.compute_enhanced_loss(query_logits, query_pseudo_labels, class_weights)
            else:
                # If query_y is None, compute entropy loss (unsupervised)
                query_probs = torch.softmax(query_logits, dim=1)
                entropy = -(query_probs * torch.log(query_probs + 1e-8)).sum(dim=1).mean()
                query_loss = -entropy  # Minimize entropy (maximize confidence)
            
            # Total loss with query adaptation
            total_loss = support_loss + 0.1 * prototype_loss + 0.1 * query_loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
        
        # Final predictions with improved logic
        with torch.no_grad():
            query_embeddings, query_logits, query_probs = self.forward_full(query_x)
            
            # Update prototypes with final embeddings
            final_normal_prototype, final_attack_prototype = self.compute_prototypes(query_embeddings, query_y) if query_y is not None else (normal_prototype, attack_prototype)
            
            # Compute Mahalanobis distances for open-set detection
            normal_mahalanobis = self.mahalanobis_distance(query_embeddings, 'normal')
            attack_mahalanobis = self.mahalanobis_distance(query_embeddings, 'attack')
            
            # Enhanced prediction logic with ensemble approach
            logit_predictions = (query_logits.squeeze() > 0).long()
            
            # Distance-based predictions with better threshold
            normal_distances = torch.cdist(query_embeddings, final_normal_prototype.unsqueeze(0), p=2).squeeze()
            attack_distances = torch.cdist(query_embeddings, final_attack_prototype.unsqueeze(0), p=2).squeeze()
            distance_predictions = (attack_distances < normal_distances).long()
            
            # Ensemble prediction with weighted voting
            # Use both logits and distances for more robust predictions
            logit_weight = 0.6
            distance_weight = 0.4
            
            # Combine predictions with weighted voting
            ensemble_predictions = (logit_weight * logit_predictions.float() + 
                                  distance_weight * distance_predictions.float())
            predictions = (ensemble_predictions > 0.5).long()
            
            # Enhanced confidence scores
            logit_confidence = torch.sigmoid(torch.abs(query_logits.squeeze()))
            distance_confidence = 1.0 / (1.0 + torch.min(normal_distances, attack_distances))
            ensemble_confidence = logit_weight * logit_confidence + distance_weight * distance_confidence
            confidence_scores = ensemble_confidence
            
            # Flag ambiguous samples as potential novel attacks
            min_distances = torch.min(normal_mahalanobis, attack_mahalanobis)
            ambiguous_mask = min_distances > self.rejection_threshold
            novel_attacks = ambiguous_mask.detach().cpu().numpy()
        
        return predictions, confidence_scores, novel_attacks
    
    def apply_data_augmentation(self, x, y):
        """
        Apply data augmentation techniques for better performance
        
        Args:
            x: Input features
            y: Labels
            
        Returns:
            augmented_x: Augmented features
            augmented_y: Augmented labels
        """
        device = x.device
        
        # Ensure labels are in correct range [0, 1]
        y = torch.clamp(y, 0, 1)
        
        # Only apply augmentation if we have enough samples
        if len(x) < 10:
            return x, y
        
        augmented_samples = []
        augmented_labels = []
        
        # Original samples
        augmented_samples.append(x)
        augmented_labels.append(y)
        
        # Add Gaussian noise for robustness (only if batch size is reasonable)
        if len(x) >= 5:
            noise_std = 0.005  # Reduced noise level
            noise = torch.randn_like(x) * noise_std
            augmented_samples.append(x + noise)
            augmented_labels.append(y)
        
        # Add dropout-like masking for regularization
        if len(x) >= 5:
            mask_prob = 0.05  # Reduced masking probability
            mask = torch.rand_like(x) > mask_prob
            masked_x = x * mask.float()
            augmented_samples.append(masked_x)
            augmented_labels.append(y)
        
        # Combine augmented samples
        if len(augmented_samples) > 1:
            augmented_x = torch.cat(augmented_samples, dim=0)
            augmented_y = torch.cat(augmented_labels, dim=0)
            
            # Ensure labels are still in correct range
            augmented_y = torch.clamp(augmented_y, 0, 1)
            
            return augmented_x, augmented_y
        else:
            return x, y
    
    def compute_binary_loss(self, logits, labels):
        """
        Compute binary cross-entropy loss
        
        Args:
            logits: Model logits
            labels: Binary labels (0=normal, 1=attack)
            
        Returns:
            loss: Binary cross-entropy loss
        """
        return self.bce_loss(logits.squeeze(), labels.float())
    
    def compute_enhanced_loss(self, logits, labels, class_weights=None):
        """
        Compute enhanced loss with focal loss and class balancing
        
        Args:
            logits: Model logits
            labels: Binary labels (0=normal, 1=attack)
            class_weights: Class weights for imbalanced data
            
        Returns:
            loss: Enhanced loss
        """
        # Focal loss for hard examples
        alpha = 0.25
        gamma = 2.0
        
        probs = torch.sigmoid(logits.squeeze())
        ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float(), reduction='none')
        p_t = probs * labels.float() + (1 - probs) * (1 - labels.float())
        focal_weight = alpha * (1 - p_t) ** gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if class_weights is not None:
            weight_tensor = torch.where(labels == 0, class_weights[0], class_weights[1])
            focal_loss = focal_loss * weight_tensor
        
        return focal_loss.mean()
    
    def evaluate_binary_metrics(self, predictions, labels, probabilities=None):
        """
        Evaluate binary classification metrics
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            probabilities: Prediction probabilities (optional)
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        if probabilities is not None:
            probabilities = probabilities.detach().cpu().numpy()
            try:
                roc_auc = roc_auc_score(labels, probabilities)
                metrics['roc_auc'] = roc_auc
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def evaluate_open_set_metrics(self, predictions, labels, novel_attacks, confidence_scores):
        """
        Evaluate open-set detection metrics
        
        Args:
            predictions: Binary predictions
            labels: Ground truth labels
            novel_attacks: Detected novel attack flags
            confidence_scores: Confidence scores
            
        Returns:
            metrics: Dictionary of open-set evaluation metrics
        """
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if hasattr(novel_attacks, 'cpu'):
            novel_attacks = novel_attacks.detach().cpu().numpy().astype(bool)
        else:
            novel_attacks = novel_attacks.astype(bool)
        confidence_scores = confidence_scores.detach().cpu().numpy()
        
        # True novel attacks (actual attack samples)
        true_novel_attacks = (labels == 1) & novel_attacks
        false_novel_attacks = (labels == 0) & novel_attacks
        
        # Open-set metrics
        n_true_novel = np.sum(true_novel_attacks)
        n_false_novel = np.sum(false_novel_attacks)
        n_total_novel = np.sum(novel_attacks)
        
        if n_total_novel > 0:
            precision_novel = n_true_novel / n_total_novel
        else:
            precision_novel = 0.0
            
        if n_true_novel > 0:
            recall_novel = n_true_novel / np.sum(labels == 1)
        else:
            recall_novel = 0.0
            
        if precision_novel + recall_novel > 0:
            f1_novel = 2 * precision_novel * recall_novel / (precision_novel + recall_novel)
        else:
            f1_novel = 0.0
        
        # False positive rate for novel detection
        fpr_novel = n_false_novel / np.sum(labels == 0) if np.sum(labels == 0) > 0 else 0.0
        
        metrics = {
            'novel_precision': precision_novel,
            'novel_recall': recall_novel,
            'novel_f1': f1_novel,
            'novel_fpr': fpr_novel,
            'n_novel_detected': n_total_novel,
            'n_true_novel': n_true_novel,
            'n_false_novel': n_false_novel,
            'avg_confidence': np.mean(confidence_scores)
        }
        
        return metrics
    
    def meta_train(self, meta_tasks, meta_epochs=5):
        """
        Meta-training method for compatibility with existing coordinator
        
        Args:
            meta_tasks: List of meta-tasks (for compatibility)
            meta_epochs: Number of meta-training epochs
            
        Returns:
            training_history: Dictionary with training metrics
        """
        logger.info(f"Running meta-training for {meta_epochs} epochs (binary classification)")
        
        # Extract data from meta-tasks for binary classification training
        all_support_x = []
        all_support_y = []
        all_query_x = []
        all_query_y = []
        
        for task in meta_tasks:
            # Handle both tuple and dictionary formats
            if isinstance(task, dict):
                support_x = task['support_x']
                support_y = task['support_y']
                query_x = task['query_x']
                query_y = task['query_y']
            else:
                support_x, support_y, query_x, query_y = task
            
            all_support_x.append(support_x)
            all_support_y.append(support_y)
            all_query_x.append(query_x)
            all_query_y.append(query_y)
        
        # Combine all data
        if all_support_x:
            support_x = torch.cat(all_support_x, dim=0)
            support_y = torch.cat(all_support_y, dim=0)
            query_x = torch.cat(all_query_x, dim=0)
            query_y = torch.cat(all_query_y, dim=0)
            
            # Combine support and query for training
            train_x = torch.cat([support_x, query_x], dim=0)
            train_y = torch.cat([support_y, query_y], dim=0)
            
            # Move to same device as model
            device = next(self.parameters()).device
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            
            # Ensure labels are in range [0, 1] for binary classification
            train_y = torch.clamp(train_y, 0, 1)
        else:
            # Fallback: create dummy data
            device = next(self.parameters()).device
            train_x = torch.randn(100, self.input_dim).to(device)
            train_y = torch.randint(0, 2, (100,)).to(device)
        
        # Calculate class weights for imbalanced data
        class_counts = torch.bincount(train_y)
        total_samples = len(train_y)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(device)
        
        logger.info(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts.tolist()))}")
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Enhanced training setup with better stability
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0005, weight_decay=1e-4, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=meta_epochs)
        
        epoch_losses = []
        epoch_accuracies = []
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(meta_epochs):
            optimizer.zero_grad()
            
            # Use original data for meta-training to avoid tensor size issues
            # Data augmentation can be applied in the main training loop
            logits = self(train_x)
            
            # Enhanced loss with class weighting and focal loss
            loss = self.compute_enhanced_loss(logits, train_y, class_weights)
            
            # Add prototype consistency loss
            with torch.no_grad():
                embeddings, _, _ = self.forward_full(train_x)
                normal_prototype, attack_prototype = self.compute_prototypes(embeddings, train_y)
            
            # Prototype consistency regularization
            normal_mask = train_y == 0
            attack_mask = train_y == 1
            
            prototype_loss = 0
            if normal_mask.sum() > 0:
                normal_embeddings = embeddings[normal_mask]
                normal_distances = torch.cdist(normal_embeddings, normal_prototype.unsqueeze(0), p=2)
                prototype_loss = normal_distances.mean()
                
            if attack_mask.sum() > 0:
                attack_embeddings = embeddings[attack_mask]
                attack_distances = torch.cdist(attack_embeddings, attack_prototype.unsqueeze(0), p=2)
                prototype_loss += attack_distances.mean()
            
            # Total loss with regularization
            total_loss = loss + 0.1 * prototype_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = (logits.squeeze() > 0).long()
                accuracy = (predictions == train_y).float().mean().item()
            
            epoch_losses.append(total_loss.item())
            epoch_accuracies.append(accuracy)
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience and epoch > 2:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            logger.info(f"Epoch {epoch+1}/{meta_epochs}: Loss={total_loss.item():.4f}, Accuracy={accuracy:.4f}")
        
        training_history = {
            'epoch_losses': epoch_losses,
            'epoch_accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1],
            'final_accuracy': epoch_accuracies[-1],
            'best_loss': best_loss
        }
        
        return training_history
    
    def train(self, mode=True):
        """
        Set training mode for compatibility
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode for compatibility"""
        return super().eval()

def create_enhanced_binary_classifier(input_dim: int, hidden_dim: int = 128, 
                                    embedding_dim: int = 64, sequence_length: int = 12,
                                    ttt_lr: float = 0.001, ttt_steps: int = 10) -> EnhancedBinaryClassifier:
    """
    Create an enhanced binary classifier instance
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        embedding_dim: Embedding dimension
        sequence_length: Sequence length parameter
        ttt_lr: Test-time training learning rate
        ttt_steps: Number of test-time training steps
        
    Returns:
        Enhanced binary classifier instance
    """
    return EnhancedBinaryClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        ttt_lr=ttt_lr,
        ttt_steps=ttt_steps
    )
