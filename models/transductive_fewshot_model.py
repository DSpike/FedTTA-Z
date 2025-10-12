#!/usr/bin/env python3
"""
Transductive Few-Shot Model with Test-Time Training for Zero-Day Detection
Implements meta-learning approach for rapid adaptation to new attack patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import copy
import math
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThresholdAgent:
    """
    Reinforcement Learning agent for learning optimal TTT threshold
    Uses a simple neural network to map state to threshold value
    """
    
    def __init__(self, state_dim=2, hidden_dim=32, learning_rate=0.001):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Simple neural network for threshold prediction
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = []  # Store (state, action, reward) tuples
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Track performance for reward calculation
        self.adaptation_history = []
        self.threshold_history = []
        
    def get_threshold(self, state):
        """
        Get threshold based on current state using RL agent
        
        Args:
            state: torch.Tensor of shape [2] containing [mean_confidence, adaptation_success_rate]
            
        Returns:
            threshold: float between 0 and 1
        """
        with torch.no_grad():
            if random.random() < self.epsilon:
                # Exploration: random threshold
                threshold = random.uniform(0.1, 0.9)
            else:
                # Exploitation: use neural network
                threshold = self.network(state.unsqueeze(0)).item()
                # Ensure threshold is in reasonable range
                threshold = max(0.1, min(0.9, threshold))
        
        self.threshold_history.append(threshold)
        return threshold
    
    def update(self, state, threshold, adaptation_success_rate, accuracy_improvement, 
               false_positives=0, false_negatives=0, true_positives=0, true_negatives=0,
               precision=0.0, recall=0.0, f1_score=0.0, samples_selected=0, total_samples=0):
        """
        Update the agent based on adaptation results with enhanced metrics
        
        Args:
            state: Current state [mean_confidence, adaptation_success_rate]
            threshold: Threshold used
            adaptation_success_rate: Success rate of adaptation
            accuracy_improvement: Improvement in accuracy after TTT
            false_positives: Number of false positive predictions
            false_negatives: Number of false negative predictions
            true_positives: Number of true positive predictions
            true_negatives: Number of true negative predictions
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            samples_selected: Number of samples selected for TTT
            total_samples: Total number of samples available
        """
        # Calculate reward based on comprehensive metrics
        reward = self._calculate_reward(
            adaptation_success_rate, accuracy_improvement, 
            false_positives, false_negatives, true_positives, true_negatives,
            precision, recall, f1_score, samples_selected, total_samples, threshold
        )
        
        # Store experience
        self.memory.append((state, threshold, reward))
        
        # Update adaptation history
        self.adaptation_history.append(adaptation_success_rate)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Train the network if we have enough experiences
        if len(self.memory) >= 10:
            self._train_network()
    
    def _calculate_reward(self, adaptation_success_rate, accuracy_improvement, 
                         false_positives=0, false_negatives=0, true_positives=0, true_negatives=0,
                         precision=0.0, recall=0.0, f1_score=0.0, samples_selected=0, total_samples=0, threshold=0.5):
        """
        Calculate comprehensive reward based on multiple performance metrics
        
        Args:
            adaptation_success_rate: Rate of successful adaptations
            accuracy_improvement: Improvement in accuracy
            false_positives: Number of false positive predictions
            false_negatives: Number of false negative predictions
            true_positives: Number of true positive predictions
            true_negatives: Number of true negative predictions
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            samples_selected: Number of samples selected for TTT
            total_samples: Total number of samples available
            threshold: Threshold value used
            
        Returns:
            reward: float reward value
        """
        # Base reward from adaptation success
        base_reward = adaptation_success_rate * 10.0
        
        # Accuracy improvement bonus
        accuracy_bonus = accuracy_improvement * 20.0
        
        # Precision and recall rewards (weighted by importance)
        precision_reward = precision * 15.0
        recall_reward = recall * 15.0
        f1_reward = f1_score * 25.0
        
        # False positive penalty (more severe for security applications)
        fp_penalty = false_positives * 2.0
        
        # False negative penalty (critical for attack detection)
        fn_penalty = false_negatives * 3.0
        
        # True positive bonus (rewarding correct attack detection)
        tp_bonus = true_positives * 0.5
        
        # True negative bonus (rewarding correct normal detection)
        tn_bonus = true_negatives * 0.3
        
        # Sample selection efficiency reward/penalty
        if total_samples > 0:
            selection_ratio = samples_selected / total_samples
            # Optimal selection ratio is between 0.1 and 0.4
            if 0.1 <= selection_ratio <= 0.4:
                selection_efficiency = 5.0  # Bonus for good selection ratio
            elif selection_ratio < 0.05:  # Too few samples selected
                selection_efficiency = -10.0  # Penalty for under-selection
            elif selection_ratio > 0.6:  # Too many samples selected
                selection_efficiency = -5.0  # Penalty for over-selection
            else:
                selection_efficiency = 0.0  # Neutral for moderate ratios
        else:
            selection_efficiency = 0.0
        
        # Threshold stability reward (prefer thresholds in reasonable range)
        if 0.2 <= threshold <= 0.8:
            threshold_stability = 2.0
        else:
            threshold_stability = -1.0
        
        # Balance reward (encourage balanced precision and recall)
        if precision > 0 and recall > 0:
            balance_ratio = min(precision, recall) / max(precision, recall)
            balance_reward = balance_ratio * 10.0
        else:
            balance_reward = 0.0
        
        # Calculate total reward
        total_reward = (base_reward + accuracy_bonus + precision_reward + recall_reward + 
                       f1_reward + tp_bonus + tn_bonus + selection_efficiency + 
                       threshold_stability + balance_reward - fp_penalty - fn_penalty)
        
        # Normalize reward to prevent extreme values
        total_reward = max(-50.0, min(50.0, total_reward))
        
        return total_reward
    
    def get_reward_breakdown(self, adaptation_success_rate, accuracy_improvement, 
                            false_positives=0, false_negatives=0, true_positives=0, true_negatives=0,
                            precision=0.0, recall=0.0, f1_score=0.0, samples_selected=0, total_samples=0, threshold=0.5):
        """
        Get detailed breakdown of reward components for debugging
        
        Returns:
            dict: Detailed breakdown of all reward components
        """
        # Calculate all components
        base_reward = adaptation_success_rate * 10.0
        accuracy_bonus = accuracy_improvement * 20.0
        precision_reward = precision * 15.0
        recall_reward = recall * 15.0
        f1_reward = f1_score * 25.0
        fp_penalty = false_positives * 2.0
        fn_penalty = false_negatives * 3.0
        tp_bonus = true_positives * 0.5
        tn_bonus = true_negatives * 0.3
        
        # Sample selection efficiency
        if total_samples > 0:
            selection_ratio = samples_selected / total_samples
            if 0.1 <= selection_ratio <= 0.4:
                selection_efficiency = 5.0
            elif selection_ratio < 0.05:
                selection_efficiency = -10.0
            elif selection_ratio > 0.6:
                selection_efficiency = -5.0
            else:
                selection_efficiency = 0.0
        else:
            selection_efficiency = 0.0
        
        # Threshold stability
        if 0.2 <= threshold <= 0.8:
            threshold_stability = 2.0
        else:
            threshold_stability = -1.0
        
        # Balance reward
        if precision > 0 and recall > 0:
            balance_ratio = min(precision, recall) / max(precision, recall)
            balance_reward = balance_ratio * 10.0
        else:
            balance_reward = 0.0
        
        total_reward = (base_reward + accuracy_bonus + precision_reward + recall_reward + 
                       f1_reward + tp_bonus + tn_bonus + selection_efficiency + 
                       threshold_stability + balance_reward - fp_penalty - fn_penalty)
        
        return {
            'base_reward': base_reward,
            'accuracy_bonus': accuracy_bonus,
            'precision_reward': precision_reward,
            'recall_reward': recall_reward,
            'f1_reward': f1_reward,
            'tp_bonus': tp_bonus,
            'tn_bonus': tn_bonus,
            'fp_penalty': -fp_penalty,
            'fn_penalty': -fn_penalty,
            'selection_efficiency': selection_efficiency,
            'threshold_stability': threshold_stability,
            'balance_reward': balance_reward,
            'total_reward': total_reward,
            'selection_ratio': samples_selected / total_samples if total_samples > 0 else 0.0
        }
    
    def _train_network(self):
        """
        Train the neural network using stored experiences
        """
        if len(self.memory) < 10:
            return
        
        # Sample recent experiences
        recent_memories = self.memory[-10:]
        
        states = torch.stack([mem[0] for mem in recent_memories])
        thresholds = torch.tensor([mem[1] for mem in recent_memories]).unsqueeze(1)
        rewards = torch.tensor([mem[2] for mem in recent_memories]).unsqueeze(1)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Predict thresholds
        predicted_thresholds = self.network(states)
        
        # Calculate loss (MSE between predicted and actual thresholds, weighted by rewards)
        loss = F.mse_loss(predicted_thresholds, thresholds, reduction='none')
        weighted_loss = (loss * (1 + rewards)).mean()
        
        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Clear old memories to prevent memory overflow
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]
    
    def get_adaptation_success_rate(self):
        """Get current adaptation success rate"""
        if not self.adaptation_history:
            return 0.5  # Default value
        return np.mean(self.adaptation_history[-10:])  # Average of last 10 adaptations
    
    def reset(self):
        """Reset the agent for new training session"""
        self.memory = []
        self.adaptation_history = []
        self.threshold_history = []
        self.epsilon = 0.1

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
            feature_extractors: Multi-scale feature extractors
            feature_projection: Feature projection layer
            self_attention: Self-attention mechanism
            x: Input features
            
        Returns:
            Normalized embeddings with self-attention applied
        """
        # Extract features from multiple scales
        features = []
        for extractor in feature_extractors:
            features.append(extractor(x))
        
        # Concatenate multi-scale features
        combined_features = torch.cat(features, dim=1)
        
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
    def update_prototypes(support_embeddings, support_y, test_embeddings, test_predictions, support_weight=0.7, test_weight=0.3):
        """
        Update prototypes using both support and test set information
        
        Args:
            support_embeddings: Support set embeddings
            support_y: Support set labels
            test_embeddings: Test set embeddings
            test_predictions: Test set predictions
            support_weight: Weight for support set contribution (default: 0.7)
            test_weight: Weight for test set contribution (default: 0.3)
            
        Returns:
            updated_prototypes: Updated class prototypes
            unique_labels: Unique class labels
        """
        unique_labels = torch.unique(support_y)
        updated_prototypes = []
        
        for label in unique_labels:
            # Support set contribution
            support_mask = support_y == label
            if support_mask.sum() > 0:
                support_class_embeddings = support_embeddings[support_mask]
                support_prototype = support_class_embeddings.mean(dim=0)
            else:
                support_prototype = torch.zeros_like(support_embeddings[0])
            
            # Test set contribution with attention weighting
            if test_predictions is not None and len(test_predictions) > 0:
                test_weights = test_predictions[:, label.item()] if len(test_predictions.shape) > 1 else test_predictions
                if test_weights.sum() > 0:
                    test_prototype = torch.sum(test_embeddings * test_weights.unsqueeze(1), dim=0) / test_weights.sum()
                    # Combine support and test contributions with configurable weights
                    updated_prototype = support_weight * support_prototype + test_weight * test_prototype
                else:
                    updated_prototype = support_prototype
            else:
                updated_prototype = support_prototype
            
            updated_prototypes.append(updated_prototype)
        
        return torch.stack(updated_prototypes), unique_labels

class LossUtils:
    """
    Centralized utility class for loss computation
    Breaks down complex loss calculation into reusable components
    """
    
    @staticmethod
    def compute_support_loss(support_embeddings, support_y, classifier):
        """Compute classification loss on support set"""
        support_logits = classifier(support_embeddings)
        return F.cross_entropy(support_logits, support_y)
    
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
    
    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss
        
        Args:
            inputs: Model predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
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
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3):
        super(TransductiveLearner, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.support_weight = support_weight
        self.test_weight = test_weight
        
        # RL-based threshold agent for dynamic TTT sample selection
        self.threshold_agent = ThresholdAgent()
        self.ttt_threshold = 0.5  # Fallback threshold
        self.adaptation_success_history = []
        
        # Multi-scale feature extractors with increased dropout for TTT overfitting prevention
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5)
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        ])
        
        # Feature projection to embedding space
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classification network
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # Attention mechanism for global context
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=2)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Transductive learning parameters
        self.transductive_lr = 0.001
        self.transductive_steps = 25
    
    def get_dynamic_threshold(self, confidence_scores):
        """
        Get dynamic threshold using RL agent
        
        Args:
            confidence_scores: torch.Tensor of confidence scores
            
        Returns:
            threshold: float threshold value
        """
        # Calculate current state
        mean_confidence = torch.mean(confidence_scores).item()
        adaptation_success_rate = self.threshold_agent.get_adaptation_success_rate()
        
        # Create state vector
        state = torch.tensor([mean_confidence, adaptation_success_rate], dtype=torch.float32)
        
        # Get threshold from RL agent
        threshold = self.threshold_agent.get_threshold(state)
        
        # Update fallback threshold
        self.ttt_threshold = threshold
        
        return threshold
    
    def update_adaptation_success(self, success_rate, accuracy_improvement, 
                                 initial_predictions=None, adapted_predictions=None, 
                                 true_labels=None, samples_selected=0, total_samples=0):
        """
        Update the RL agent with adaptation results and comprehensive metrics
        
        Args:
            success_rate: float, success rate of the adaptation
            accuracy_improvement: float, improvement in accuracy
            initial_predictions: Initial predictions before TTT
            adapted_predictions: Predictions after TTT adaptation
            true_labels: True labels for the samples
            samples_selected: Number of samples selected for TTT
            total_samples: Total number of samples available
        """
        # Calculate current state
        mean_confidence = 0.5  # Placeholder - would need actual confidence scores
        adaptation_success_rate = self.threshold_agent.get_adaptation_success_rate()
        state = torch.tensor([mean_confidence, adaptation_success_rate], dtype=torch.float32)
        
        # Calculate comprehensive metrics if predictions are available
        if initial_predictions is not None and adapted_predictions is not None and true_labels is not None:
            # Convert to numpy for sklearn metrics
            if torch.is_tensor(adapted_predictions):
                adapted_preds = adapted_predictions.cpu().numpy()
            else:
                adapted_preds = adapted_predictions
                
            if torch.is_tensor(true_labels):
                true_labels_np = true_labels.cpu().numpy()
            else:
                true_labels_np = true_labels
            
            # Calculate confusion matrix components
            tp = ((adapted_preds == 1) & (true_labels_np == 1)).sum()
            tn = ((adapted_preds == 0) & (true_labels_np == 0)).sum()
            fp = ((adapted_preds == 1) & (true_labels_np == 0)).sum()
            fn = ((adapted_preds == 0) & (true_labels_np == 1)).sum()
            
            # Calculate precision, recall, and F1
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
                
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
                
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
        else:
            # Default values when predictions are not available
            tp = tn = fp = fn = 0
            precision = recall = f1_score = 0.0
        
        # Update the agent with comprehensive metrics
        self.threshold_agent.update(
            state, self.ttt_threshold, success_rate, accuracy_improvement,
            fp, fn, tp, tn, precision, recall, f1_score, 
            samples_selected, total_samples
        )
        
        # Store for tracking
        self.adaptation_success_history.append(success_rate)
        
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
    
    def _apply_self_attention(self, embeddings):
        """
        Apply self-attention with residual connection
        """
        embeddings_reshaped = embeddings.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        attended_embeddings, _ = self.self_attention(
            embeddings_reshaped, embeddings_reshaped, embeddings_reshaped
        )
        return attended_embeddings.squeeze(1) + embeddings  # Residual connection
    
    
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
                LoggingUtils.log_early_stopping(step, 8, best_loss, best_acc)
                break
            
            if step % 5 == 0:
                logger.info(f"Transductive step {step}: Loss = {total_loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        return test_predictions, prototypes, unique_labels
    
    def update_prototypes(self, support_embeddings, support_y, test_embeddings, test_predictions):
        """
        Unified prototype update method handling both support and test contributions
        Now uses centralized utility for consistency with configurable weights
        """
        return PrototypeUtils.update_prototypes(
            support_embeddings, support_y, test_embeddings, test_predictions,
            support_weight=self.support_weight, test_weight=self.test_weight
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
    

class MetaLearner(nn.Module):
    """
    Meta-Learning model for few-shot adaptation with transductive learning
    Learns to quickly adapt to new tasks with minimal examples
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3):
        super(MetaLearner, self).__init__()
        
        self.transductive_net = TransductiveLearner(input_dim, hidden_dim, embedding_dim, num_classes, support_weight, test_weight)
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
    
    def adapt_to_task(self, support_x, support_y, adaptation_steps: int = None):
        """
        Adapt the model to a specific task using support set with transductive learning
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps
            
        Returns:
            adapted_model: Model adapted to the task
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform adaptation steps using transductive learning
        for step in range(adaptation_steps):
            adapted_optimizer.zero_grad()
            
            # Get embeddings
            support_embeddings = adapted_model.transductive_net.extract_embeddings(support_x)
            
            # Compute prototypes
            prototypes, prototype_labels = adapted_model.transductive_net.update_prototypes(
                support_embeddings, support_y, support_embeddings, None
            )
            
            # Compute loss (prototype consistency)
            loss = 0
            for i, label in enumerate(prototype_labels):
                mask = support_y == label
                if mask.sum() > 0:
                    class_embeddings = support_embeddings[mask]
                    prototype = prototypes[i]
                    loss += F.mse_loss(class_embeddings.mean(dim=0), prototype)
            
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model

class TransductiveFewShotModel(nn.Module):
    """
    Transductive Few-Shot Model for Zero-Day Detection
    Combines meta-learning with test-time training for rapid adaptation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3):
        super(TransductiveFewShotModel, self).__init__()
        
        self.meta_learner = MetaLearner(input_dim, hidden_dim, embedding_dim, num_classes, support_weight, test_weight)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Test-time training parameters
        self.ttt_lr = 0.001
        self.ttt_steps = 21
        self.ttt_threshold = 0.1  # Confidence threshold for test-time training
        
        # Zero-day detection parameters
        self.anomaly_threshold = 0.5
        self.adaptation_threshold = 0.3
        
        # Dropout regularization for TTT overfitting prevention
        self.dropout_prob = 0.5
        
    def forward(self, x):
        return self.meta_learner(x)
    
    def get_embeddings(self, x):
        """
        Extract embeddings from the model
        Now uses centralized utility for consistency
        """
        return self.meta_learner.get_embeddings(x)
    
    def set_ttt_mode(self, training=True):
        """
        Set model mode for Test-Time Training
        
        Args:
            training: If True, set to training mode (dropout active)
                     If False, set to evaluation mode (dropout disabled)
        """
        if training:
            self.train()  # Enable dropout during TTT adaptation
            LoggingUtils.log_model_mode("training", 3)  # 3 dropout layers
        else:
            self.eval()   # Disable dropout during evaluation
            LoggingUtils.log_model_mode("evaluation", 3)  # 3 dropout layers
    
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
    
    def detect_zero_day(self, x, support_x, support_y, adaptation_steps: int = None):
        """
        Detect zero-day attacks using transductive few-shot learning
        
        Args:
            x: Test samples
            support_x: Support set (known attacks)
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps
            
        Returns:
            predictions: Binary predictions (0=normal, 1=attack)
            confidence: Confidence scores
            is_zero_day: Zero-day detection flags
        """
        if adaptation_steps is None:
            adaptation_steps = self.ttt_steps
        
        logger.info("Starting transductive zero-day detection")
        
        # Move tensors to the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        # Get embeddings
        test_embeddings = self.meta_learner.get_embeddings(x)
        support_embeddings = self.meta_learner.get_embeddings(support_x)
        
        # Compute prototypes from support set
        prototypes, prototype_labels = self.meta_learner.transductive_net.update_prototypes(
            support_embeddings, support_y, test_embeddings, None
        )
        
        # Initial classification using distance-based approach
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        initial_predictions = prototype_labels[torch.argmin(distances, dim=1)]
        
        # Compute confidence scores
        confidence = self.compute_confidence(test_embeddings, prototypes, prototype_labels)
        
        # Get dynamic threshold using RL agent
        dynamic_threshold = self.get_dynamic_threshold(confidence)
        
        # Identify low-confidence samples for test-time training
        low_confidence_mask = confidence < dynamic_threshold
        low_confidence_indices = torch.where(low_confidence_mask)[0]
        
        logger.info(f"Dynamic TTT threshold: {dynamic_threshold:.4f}, Selected {len(low_confidence_indices)} samples for adaptation")
        
        if len(low_confidence_indices) > 0:
            logger.info(f"Test-time training on {len(low_confidence_indices)} low-confidence samples")
            
            # Perform test-time training on low-confidence samples
            adapted_model = self.meta_learner.adapt_to_task(
                support_x, support_y, adaptation_steps
            )
            
            # Re-classify with adapted model
            # Get the original test samples for low-confidence cases
            low_confidence_samples = x[low_confidence_mask]
            adapted_embeddings = adapted_model.get_embeddings(low_confidence_samples)
            adapted_distances = torch.cdist(adapted_embeddings, prototypes, p=2)
            adapted_predictions = prototype_labels[torch.argmin(adapted_distances, dim=1)]
            
            # Update predictions for low-confidence samples
            final_predictions = initial_predictions.clone()
            final_predictions[low_confidence_mask] = adapted_predictions
            
            # Update confidence scores
            adapted_confidence = self.compute_confidence(adapted_embeddings, prototypes, prototype_labels)
            final_confidence = confidence.clone()
            final_confidence[low_confidence_mask] = adapted_confidence
            
            # Calculate adaptation success metrics for RL agent
            if len(low_confidence_indices) > 0:
                # Calculate accuracy improvement
                initial_accuracy = (initial_predictions[low_confidence_mask] == y[low_confidence_mask]).float().mean().item()
                adapted_accuracy = (adapted_predictions == y[low_confidence_mask]).float().mean().item()
                accuracy_improvement = adapted_accuracy - initial_accuracy
                
                # Calculate success rate (how many samples improved)
                improvement_mask = adapted_confidence > confidence[low_confidence_mask]
                success_rate = improvement_mask.float().mean().item()
                
                # Update RL agent with comprehensive metrics
                self.update_adaptation_success(
                    success_rate, accuracy_improvement,
                    initial_predictions[low_confidence_mask], 
                    adapted_predictions,
                    y[low_confidence_mask],
                    len(low_confidence_indices),
                    len(y)
                )
                
                # Log comprehensive metrics
                logger.info(f"TTT Adaptation Success - Rate: {success_rate:.3f}, Accuracy Improvement: {accuracy_improvement:.3f}")
                logger.info(f"Enhanced RL Metrics - Samples Selected: {len(low_confidence_indices)}/{len(y)} ({len(low_confidence_indices)/len(y)*100:.1f}%)")
        else:
            final_predictions = initial_predictions
            final_confidence = confidence
        
        # Zero-day detection: samples with low confidence and predicted as attack
        attack_mask = final_predictions == 1
        low_confidence_attacks = attack_mask & (final_confidence < self.adaptation_threshold)
        
        # Convert to binary labels (0=normal, 1=attack)
        binary_predictions = (final_predictions == 1).long()
        
        logger.info(f"Transductive zero-day detection completed")
        logger.info(f"Zero-day samples detected: {low_confidence_attacks.sum().item()}")
        
        return binary_predictions, final_confidence, low_confidence_attacks
    
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
                
                # Meta-update
                loss, predictions = self.meta_learner.meta_update(
                    support_x, support_y, query_x, query_y
                )
                
                # Compute accuracy
                accuracy = (predictions == query_y).float().mean().item()
                
                # Backward pass
                self.meta_learner.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_learner.meta_optimizer.step()
                
                epoch_losses.append(loss.item())
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
    
    def evaluate_zero_day_detection(self, test_x, test_y, support_x, support_y):
        """
        Evaluate zero-day detection performance
        
        Args:
            test_x: Test samples
            test_y: Test labels
            support_x: Support set
            support_y: Support set labels
            
        Returns:
            metrics: Evaluation metrics
        """
        logger.info("Evaluating transductive zero-day detection performance")
        
        # Detect zero-day attacks
        predictions, confidence, is_zero_day = self.detect_zero_day(
            test_x, support_x, support_y
        )
        
        # Convert to numpy for sklearn metrics
        predictions_np = predictions.detach().cpu().numpy()
        test_y_np = test_y.detach().cpu().numpy()
        confidence_np = confidence.detach().cpu().numpy()
        is_zero_day_np = is_zero_day.detach().cpu().numpy()
        
        # Compute metrics
        accuracy = accuracy_score(test_y_np, predictions_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y_np, predictions_np, average='binary'
        )
        
        try:
            roc_auc = roc_auc_score(test_y_np, confidence_np)
        except:
            roc_auc = 0.5
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix, matthews_corrcoef
        cm = confusion_matrix(test_y_np, predictions_np)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Compute Matthews Correlation Coefficient (MCCC)
        try:
            mccc = matthews_corrcoef(test_y_np, predictions_np)
        except:
            mccc = 0.0
        
        # Zero-day specific metrics
        zero_day_detection_rate = is_zero_day_np.mean()
        avg_confidence = confidence_np.mean()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'mccc': mccc,
            'zero_day_detection_rate': zero_day_detection_rate,
            'avg_confidence': avg_confidence,
            'num_zero_day_samples': is_zero_day_np.sum(),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'total_samples': len(test_y_np)
        }
        
        logger.info(f"Transductive zero-day detection results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  MCCC: {mccc:.4f}")
        logger.info(f"  Zero-day detection rate: {zero_day_detection_rate:.4f}")
        logger.info(f"  Average confidence: {avg_confidence:.4f}")
        logger.info(f"  Confusion Matrix - TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        
        return metrics

def create_meta_tasks(data_x, data_y, n_way: int = 2, k_shot: int = 5, n_query: int = 15, n_tasks: int = 100):
    """
    Create meta-learning tasks for few-shot learning
    
    Args:
        data_x: Input data
        data_y: Labels
        n_way: Number of classes per task
        k_shot: Number of support samples per class
        n_query: Number of query samples per class
        n_tasks: Number of tasks to create
        
    Returns:
        meta_tasks: List of meta-learning tasks
    """
    logger.info(f"Creating {n_tasks} meta-learning tasks ({n_way}-way, {k_shot}-shot)")
    
    meta_tasks = []
    unique_labels = torch.unique(data_y)
    
    for _ in range(n_tasks):
        # Sample classes for this task
        task_classes = torch.randperm(len(unique_labels))[:n_way]
        selected_labels = unique_labels[task_classes]
        
        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []
        
        for label in selected_labels:
            # Get samples for this class
            class_mask = data_y == label
            class_indices = torch.where(class_mask)[0]
            
            # Shuffle and select samples
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]
            
            # Support set
            support_indices = shuffled_indices[:k_shot]
            support_x_list.append(data_x[support_indices])
            support_y_list.append(data_y[support_indices])
            
            # Query set
            query_indices = shuffled_indices[k_shot:k_shot + n_query]
            query_x_list.append(data_x[query_indices])
            query_y_list.append(data_y[query_indices])
        
        # Combine all classes
        support_x = torch.cat(support_x_list, dim=0)
        support_y = torch.cat(support_y_list, dim=0)
        query_x = torch.cat(query_x_list, dim=0)
        query_y = torch.cat(query_y_list, dim=0)
        
        # Relabel to 0, 1, 2, ... for this task
        label_mapping = {label.item(): i for i, label in enumerate(selected_labels)}
        support_y_relabeled = torch.tensor([label_mapping[label.item()] for label in support_y])
        query_y_relabeled = torch.tensor([label_mapping[label.item()] for label in query_y])
        
        meta_tasks.append({
            'support_x': support_x,
            'support_y': support_y_relabeled,
            'query_x': query_x,
            'query_y': query_y_relabeled
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
    
    # Create meta-tasks
    meta_tasks = create_meta_tasks(X_train, y_train, n_tasks=50)
    
    # Meta-train the model
    training_history = model.meta_train(meta_tasks, meta_epochs=20)
    
    # Evaluate zero-day detection
    metrics = model.evaluate_zero_day_detection(X_test, y_test, X_train, y_train)
    
    logger.info("✅ Transductive few-shot model test completed!")
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
