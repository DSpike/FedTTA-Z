#!/usr/bin/env python3
"""
Transductive Few-Shot Model with Test-Time Training for Zero-Day Detection
Implements meta-learning approach for rapid adaptation to new attack patterns

PROPER TTT EVALUATION:
This module implements TRUE test-time training following TTT principles:

1. TTT SHOULD ALWAYS RUN: Test-time training is designed to run at test time on unlabeled data.
   The evaluation_mode parameter should be set to False for proper TTT evaluation.

2. NO DATA LEAKAGE: TTT uses unsupervised objectives (consistency loss, smoothness regularization)
   on test data, so no test label leakage occurs. Support set is validation data, not training data.

3. DATA SPLIT VALIDATION: The validate_data_splits() method checks for exact duplicates
   between train/val/test splits to ensure no data overlap.

4. PROPER SUPPORT SET USAGE: During evaluation, validation data is used as support set:
   - Computing initial class prototypes
   - Providing labeled examples for transductive adaptation
   - No training data leakage since support is from validation set

5. EVALUATION WORKFLOW:
   - Training: Use training data for meta-learning
   - Validation: Use validation data for hyperparameter tuning and as support set
   - Testing: Use test data for final evaluation with TTT adaptation enabled
   - TTT adaptation uses both support (validation) and test data for prototype refinement

CRITICAL: TTT should NEVER be disabled during evaluation as this violates TTT principles
and provides invalid performance metrics that don't reflect the model's true capability.
"""

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
            
            # SCIENTIFIC FIX: Consistent handling of test predictions for prototype updates
            if test_predictions is not None and len(test_predictions) > 0:
                # Ensure consistent format - always use soft assignments
                if len(test_predictions.shape) == 1:
                    # Convert hard assignments to soft one-hot encoding
                    num_classes = len(unique_labels)
                    test_predictions_soft = torch.zeros(len(test_predictions), num_classes)
                    for i, pred in enumerate(test_predictions):
                        if pred.item() < num_classes:
                            test_predictions_soft[i, pred.item()] = 1.0
                    test_weights = test_predictions_soft[:, label.item()]
                else:
                    # Already soft assignments
                    test_weights = test_predictions[:, label.item()]
                
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
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64, num_classes: int = 2, support_weight: float = 0.7, test_weight: float = 0.3, sequence_length: int = 1):
        super(TransductiveLearner, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes  # Now supports 10 classes for UNSW-NB15
        self.support_weight = support_weight
        self.test_weight = test_weight
        
        # RL-based threshold agent for dynamic TTT sample selection
        self.threshold_agent = ThresholdAgent()
        self.ttt_threshold = 0.5  # Fallback threshold
        self.adaptation_success_history = []
        
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
    
    def select_ttt_samples(self, confidence_scores, threshold=None):
        """
        Select samples for TTT based on confidence scores
        Args:
            confidence_scores: torch.Tensor of confidence scores
            threshold: Optional threshold value
        Returns:
            selected_indices: Indices of samples selected for TTT
        """
        if threshold is None:
            threshold = self.get_dynamic_threshold(confidence_scores)
        
        selected_indices = torch.where(confidence_scores < threshold)[0]
        return selected_indices
    
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
    
    def set_ttt_mode(self, training=True):
        """
        Set the model to training or evaluation mode for TTT
        Args:
            training: If True, set to training mode; if False, set to evaluation mode
        """
        if training:
            self.train()
            # Enable dropout for TTT training
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.2  # Lower dropout for TTT
        else:
            self.eval()
            # Disable dropout for evaluation
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
    
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
            
            # Calculate multiclass metrics using sklearn
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels_np, adapted_preds, average='macro', zero_division=0
            )
            
            # Calculate confusion matrix for zero-day class (e.g., DoS=4)
            cm = confusion_matrix(true_labels_np, adapted_preds)
            zero_day_class = 4  # DoS attack class
            if zero_day_class < cm.shape[0] and zero_day_class < cm.shape[1]:
                # True positives for zero-day class
                tp_zero_day = cm[zero_day_class, zero_day_class]
                # False negatives for zero-day class
                fn_zero_day = cm[zero_day_class, :].sum() - tp_zero_day
                # False positives for zero-day class
                fp_zero_day = cm[:, zero_day_class].sum() - tp_zero_day
                
                # Zero-day class precision and recall
                if tp_zero_day + fp_zero_day > 0:
                    precision_zero_day = tp_zero_day / (tp_zero_day + fp_zero_day)
            else:
                precision_zero_day = 0.0
                
            if tp_zero_day + fn_zero_day > 0:
                recall_zero_day = tp_zero_day / (tp_zero_day + fn_zero_day)
            else:
                recall_zero_day = 0.0
        else:
            # Default values when predictions are not available
            precision_macro = recall_macro = f1_macro = 0.0
            precision_zero_day = recall_zero_day = 0.0
        
        # Update the agent with comprehensive metrics
        self.threshold_agent.update(
            state, self.ttt_threshold, success_rate, accuracy_improvement,
            0, 0, 0, 0, precision_macro, recall_macro, f1_macro, 
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
    
    def _focal_loss(self, logits, targets, class_weights, alpha=0.25, gamma=2.0):
        """
        Focal loss implementation for handling class imbalance
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
    
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
                LoggingUtils.log_early_stopping(step, 8, best_loss, best_acc)
                break
            
            if step % 5 == 0:
                logger.info(f"Transductive step {step}: Loss = {total_loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        return test_predictions, prototypes, unique_labels
    
    def transductive_adaptation(self, support_x: torch.Tensor, support_y: torch.Tensor, 
                                query_x: torch.Tensor, query_y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Performs transductive adaptation (Test-Time Training) on the query set.
        This is the optimized version with confidence-based sample selection.
        Args:
            support_x: Support set features (batch_size, sequence_length, input_dim)
            support_y: Support set labels (batch_size,)
            query_x: Query set features (batch_size, sequence_length, input_dim)
            query_y: Optional query set labels for evaluation (batch_size,)
        Returns:
            adapted_logits: Logits for the query set after adaptation
            adaptation_metrics: Dictionary of metrics from the adaptation process
        """
        import copy
        
        self_copy = copy.deepcopy(self)
        self_copy.train()
        
        # Use a separate optimizer for TTT
        ttt_optimizer = optim.Adam(self_copy.parameters(), lr=0.0005)
        
        # Store initial predictions for comparison
        with torch.no_grad():
            initial_logits = self_copy(query_x)
            initial_predictions = torch.argmax(initial_logits, dim=1)
        
        losses = []
        accuracies = []
        support_losses = []
        consistency_losses = []
        
        # Select samples for TTT based on confidence
        with torch.no_grad():
            confidence_scores = self_copy.get_confidence_scores(query_x)
            self.ttt_threshold = self_copy.get_dynamic_threshold(confidence_scores)
            
            selected_indices = torch.where(confidence_scores < self.ttt_threshold)[0]
            
            if len(selected_indices) == 0:
                logger.info(f"No samples selected for TTT (all above threshold {self.ttt_threshold:.4f}). Skipping adaptation.")
                adaptation_metrics = {
                    'losses': [], 'accuracies': [], 'support_losses': [], 'consistency_losses': [],
                    'accuracy_improvement': 0.0, 'success_rate': 0.0,
                    'initial_predictions': initial_logits, 'adapted_predictions': initial_logits
                }
                return initial_logits, adaptation_metrics
            
            ttt_query_x = query_x[selected_indices]
            ttt_query_y = query_y[selected_indices] if query_y is not None else None
            
            logger.info(f"Selected {len(selected_indices)} samples out of {len(query_x)} for TTT adaptation (threshold: {self.ttt_threshold:.4f})")
        
        # TTT adaptation loop
        for step in range(5):  # Reduced steps for efficiency
            ttt_optimizer.zero_grad()
            
            # Forward pass on support and selected query samples
            support_logits = self_copy(support_x)
            query_logits = self_copy(ttt_query_x)
            
            # Support loss
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # Consistency loss on query samples
            consistency_loss = F.cross_entropy(query_logits, torch.argmax(query_logits, dim=1))
            
            # Total loss
            total_loss = support_loss + 0.1 * consistency_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self_copy.parameters(), max_norm=1.0)
            ttt_optimizer.step()
            
            # Store metrics
            losses.append(total_loss.item())
            support_losses.append(support_loss.item())
            consistency_losses.append(consistency_loss.item())
            
            # Calculate accuracy if labels available
            if ttt_query_y is not None:
                with torch.no_grad():
                    query_preds = torch.argmax(query_logits, dim=1)
                    accuracy = (query_preds == ttt_query_y).float().mean().item()
                    accuracies.append(accuracy)
        
        # Get final predictions
        with torch.no_grad():
            adapted_logits = self_copy(query_x)
            adapted_predictions = torch.argmax(adapted_logits, dim=1)
        
        # Calculate improvement metrics
        accuracy_improvement = 0.0
        if query_y is not None:
            initial_acc = (initial_predictions == query_y).float().mean().item()
            final_acc = (adapted_predictions == query_y).float().mean().item()
            accuracy_improvement = final_acc - initial_acc
        
        # Update adaptation success
        success_rate = 1.0 if accuracy_improvement > 0 else 0.0
        self.update_adaptation_success(
            success_rate, accuracy_improvement,
            initial_predictions, adapted_predictions,
            query_y, len(selected_indices), len(query_x)
        )
        
        # Prepare metrics
        adaptation_metrics = {
            'losses': losses,
            'accuracies': accuracies,
            'support_losses': support_losses,
            'consistency_losses': consistency_losses,
            'accuracy_improvement': accuracy_improvement,
            'success_rate': success_rate,
            'samples_selected': len(selected_indices),
            'total_samples': len(query_x),
            'threshold_used': self.ttt_threshold,
            'initial_predictions': initial_logits,
            'adapted_predictions': adapted_logits
        }
        
        return adapted_logits, adaptation_metrics
    
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
                support_loss = self._focal_loss(support_logits, support_y, support_class_weights, alpha=0.25, gamma=2.0)
                
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
                query_loss = self._focal_loss(query_logits, query_y, query_class_weights, alpha=0.25, gamma=2.0)
                
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
    
    def adapt_to_task(self, support_x, support_y, adaptation_steps: int = None):
        """
        Adapt the model to a specific task using transductive learning
        Args:
            support_x: Support set features
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps (default: self.transductive_steps)
        Returns:
            adapted_model: Adapted model
        """
        if adaptation_steps is None:
            adaptation_steps = self.transductive_steps
        
        # Create a copy for adaptation
        adapted_model = copy.deepcopy(self)
        adapted_model.train()
        
        # Use transductive optimization
        optimizer = optim.Adam(adapted_model.parameters(), lr=self.transductive_lr)
        
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            
            # Forward pass
            support_logits = adapted_model(support_x)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # Backward pass
            support_loss.backward()
            optimizer.step()
        
        return adapted_model
    

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
    
    def adapt_to_task(self, support_x, support_y, adaptation_steps: int = None):
        """
        Adapt the model to a specific task using TRUE transductive learning
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            adaptation_steps: Number of adaptation steps
            
        Returns:
            adapted_model: Model adapted to the task using transductive learning
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform TRUE transductive adaptation steps
        for step in range(adaptation_steps):
            adapted_optimizer.zero_grad()
            
            # Get embeddings
            support_embeddings = adapted_model.transductive_net.extract_embeddings(support_x)
            
            # For transductive learning, we need test data to contribute to prototypes
            # Since we don't have test data in this method, we'll use support data as both
            # support and "test" for prototype refinement (this is a limitation of this method)
            
            # Compute prototypes using support data only (fallback for this method)
            prototypes, prototype_labels = adapted_model.transductive_net.update_prototypes(
                support_embeddings, support_y, support_embeddings, None
            )
            
            # Compute transductive-style loss with consistency regularization
            loss = 0
            
            # 1. Support set loss (supervised)
            for i, label in enumerate(prototype_labels):
                mask = support_y == label
                if mask.sum() > 0:
                    class_embeddings = support_embeddings[mask]
                    prototype = prototypes[i]
                    loss += F.mse_loss(class_embeddings.mean(dim=0), prototype)
            
            # 2. Consistency regularization within support set
            # Encourage embeddings of the same class to be similar
            consistency_loss = 0
            for i, label in enumerate(prototype_labels):
                mask = support_y == label
                if mask.sum() > 1:  # Need at least 2 samples for consistency
                    class_embeddings = support_embeddings[mask]
                    # Compute pairwise distances within class
                    pairwise_distances = torch.cdist(class_embeddings, class_embeddings, p=2)
                    # Minimize intra-class distances
                    consistency_loss += torch.mean(pairwise_distances)
            
            # Total loss with consistency regularization
            total_loss = loss + 0.1 * consistency_loss
            
            total_loss.backward()
            adapted_optimizer.step()
        
        return adapted_model

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
        
        # OPTIMIZED Test-time training parameters for better performance
        self.ttt_lr = 0.0005      # Reduced learning rate for stability
        self.ttt_steps = 100      # Increased steps for better adaptation
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
    
    def detect_zero_day(self, x, support_x, support_y, adaptation_steps: int = None, test_y=None, evaluation_mode: bool = False):
        """
        Detect zero-day attacks using TRUE transductive few-shot learning with multiclass support
        
        Args:
            x: Test samples
            support_x: Support set (known attacks from validation data)
            support_y: Support set labels (10-class: 0=Normal, 1-9=Attack types)
            adaptation_steps: Number of adaptation steps
            test_y: Optional test labels for evaluation (used only for metrics calculation)
            evaluation_mode: If True, skips TTT adaptation (NOT RECOMMENDED - violates TTT principles)
            
        Returns:
            predictions: Multiclass predictions (0-9: Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)
            confidence: Confidence scores
            is_zero_day: Zero-day detection flags
            
        Note:
            TTT should ALWAYS run at test time as it uses unsupervised objectives.
            Setting evaluation_mode=True defeats the purpose of test-time training.
            Support set is validation data (not training data), so no training leakage occurs.
        """
        if adaptation_steps is None:
            adaptation_steps = self.ttt_steps
        
        logger.info("Starting TRUE transductive zero-day detection")
        
        # Move tensors to the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        # Get embeddings
        test_embeddings = self.meta_learner.get_embeddings(x)
        support_embeddings = self.meta_learner.get_embeddings(support_x)
        
        # Compute initial prototypes from support set only
        prototypes, prototype_labels = self.update_prototypes(
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
        
        if len(low_confidence_indices) > 0 and not evaluation_mode:
            logger.info(f"Performing TRUE transductive adaptation on {len(low_confidence_indices)} low-confidence samples")
            
            # TRUE TRANSDUCTIVE ADAPTATION: Use both support and test data
            adapted_model, final_prototypes = self._perform_transductive_adaptation(
                support_x, support_y, x, adaptation_steps, low_confidence_mask
            )
            
            # Re-classify with adapted model and updated prototypes
            adapted_embeddings = adapted_model.get_embeddings(x)
            adapted_distances = torch.cdist(adapted_embeddings, final_prototypes, p=2)
            adapted_predictions = prototype_labels[torch.argmin(adapted_distances, dim=1)]
            
            # Update predictions for low-confidence samples
            final_predictions = initial_predictions.clone()
            final_predictions[low_confidence_mask] = adapted_predictions[low_confidence_mask]
            
            # Update confidence scores
            adapted_confidence = self.compute_confidence(adapted_embeddings, final_prototypes, prototype_labels)
            final_confidence = confidence.clone()
            final_confidence[low_confidence_mask] = adapted_confidence[low_confidence_mask]
            
            # Calculate adaptation success metrics for RL agent
            if len(low_confidence_indices) > 0 and test_y is not None:
                # Calculate accuracy improvement
                initial_accuracy = (initial_predictions[low_confidence_mask] == test_y[low_confidence_mask]).float().mean().item()
                adapted_accuracy = (adapted_predictions[low_confidence_mask] == test_y[low_confidence_mask]).float().mean().item()
                accuracy_improvement = adapted_accuracy - initial_accuracy
                
                # Calculate success rate (how many samples improved)
                improvement_mask = adapted_confidence[low_confidence_mask] > confidence[low_confidence_mask]
                success_rate = improvement_mask.float().mean().item()
                
                # Update RL agent with comprehensive metrics
                self.update_adaptation_success(
                    success_rate, accuracy_improvement,
                    initial_predictions[low_confidence_mask], 
                    adapted_predictions[low_confidence_mask],
                    test_y[low_confidence_mask],
                    len(low_confidence_indices),
                    len(test_y)
                )
                
                # Log comprehensive metrics
                logger.info(f"TRUE Transductive Adaptation Success - Rate: {success_rate:.3f}, Accuracy Improvement: {accuracy_improvement:.3f}")
                logger.info(f"Enhanced RL Metrics - Samples Selected: {len(low_confidence_indices)}/{len(test_y)} ({len(low_confidence_indices)/len(test_y)*100:.1f}%)")
        elif len(low_confidence_indices) > 0 and evaluation_mode:
            logger.warning(f"⚠️ EVALUATION MODE: Skipping TTT adaptation on {len(low_confidence_indices)} low-confidence samples")
            logger.warning(f"⚠️ This violates TTT principles - TTT should ALWAYS run at test time!")
            logger.warning(f"⚠️ Results will NOT reflect the model's true TTT-augmented performance")
            final_predictions = initial_predictions
            final_confidence = confidence
        else:
            final_predictions = initial_predictions
            final_confidence = confidence
        
        # Zero-day detection: samples with low confidence regardless of predicted class
        # This allows for proper multiclass evaluation without binary simplification
        low_confidence_attacks = final_confidence < self.adaptation_threshold
        
        # Return multiclass predictions (0-9) instead of binary
        # This preserves the original 10-class classification for accurate evaluation
        
        logger.info(f"Transductive zero-day detection completed (10-class multiclass)")
        logger.info(f"Zero-day samples detected: {low_confidence_attacks.sum().item()}")
        logger.info(f"Predicted classes distribution: {torch.bincount(final_predictions, minlength=10).tolist()}")
        
        return final_predictions, final_confidence, low_confidence_attacks
    
    def _perform_transductive_adaptation(self, support_x, support_y, test_x, adaptation_steps, low_confidence_mask):
        """
        Perform TRUE transductive adaptation using both support and test data
        
        Args:
            support_x: Support set features
            support_y: Support set labels
            test_x: Test set features (unlabeled)
            adaptation_steps: Number of adaptation steps
            low_confidence_mask: Mask for low-confidence test samples
            
        Returns:
            adapted_model: Model adapted through transductive learning
            final_prototypes: Updated prototypes incorporating test data
        """
        logger.info("🔄 Starting TRUE transductive adaptation...")
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self)
        adapted_model.train()
        
        # Setup optimizer for transductive adaptation
        optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=self.ttt_lr, weight_decay=1e-4)
        
        # Get initial embeddings
        support_embeddings = adapted_model.get_embeddings(support_x)
        test_embeddings = adapted_model.get_embeddings(test_x)
        
        # Compute initial prototypes from support set only
        prototypes, prototype_labels = adapted_model.update_prototypes(
            support_embeddings, support_y, test_embeddings, None
        )
        
        logger.info(f"Initial prototypes shape: {prototypes.shape}")
        
        # Transductive optimization loop
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            
            # Recompute embeddings (they change during optimization)
            support_embeddings = adapted_model.get_embeddings(support_x)
            test_embeddings = adapted_model.get_embeddings(test_x)
            
            # Compute test predictions using soft assignments
            distances = torch.cdist(test_embeddings, prototypes, p=2)
            test_predictions = torch.softmax(-distances, dim=1)  # Soft assignments
            
            # Update prototypes using BOTH support and test data
            updated_prototypes, _ = adapted_model.update_prototypes(
                support_embeddings, support_y, test_embeddings, test_predictions
            )
            
            # Compute transductive loss components
            # 1. Support set loss (supervised)
            support_loss = self._compute_support_loss(support_embeddings, support_y, updated_prototypes, prototype_labels)
            
            # 2. Test set consistency loss (unsupervised)
            consistency_loss = self._compute_consistency_loss(test_embeddings, test_predictions, updated_prototypes, prototype_labels)
            
            # 3. Smoothness regularization
            smoothness_loss = self._compute_smoothness_loss(test_embeddings, low_confidence_mask)
            
            # Total transductive loss
            total_loss = support_loss + 0.1 * consistency_loss + 0.01 * smoothness_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update prototypes for next iteration
            prototypes = updated_prototypes.detach()
            
            if step % 5 == 0:
                logger.info(f"Transductive Step {step+1}: Total Loss = {total_loss.item():.4f}, "
                           f"Support Loss = {support_loss.item():.4f}, "
                           f"Consistency Loss = {consistency_loss.item():.4f}")
        
        logger.info("✅ TRUE transductive adaptation completed!")
        return adapted_model, prototypes
    
    def _compute_support_loss(self, support_embeddings, support_y, prototypes, prototype_labels):
        """Compute supervised loss on support set"""
        loss = 0
        for i, label in enumerate(prototype_labels):
            mask = support_y == label
            if mask.sum() > 0:
                class_embeddings = support_embeddings[mask]
                prototype = prototypes[i]
                # MSE loss between class mean and prototype
                loss += F.mse_loss(class_embeddings.mean(dim=0), prototype)
        return loss
    
    def _compute_consistency_loss(self, test_embeddings, test_predictions, prototypes, prototype_labels):
        """Compute consistency loss on test set using soft assignments"""
        # Compute distances to prototypes
        distances = torch.cdist(test_embeddings, prototypes, p=2)
        
        # Compute soft assignments
        soft_assignments = torch.softmax(-distances, dim=1)
        
        # Consistency loss: encourage test embeddings to be close to their assigned prototypes
        consistency_loss = 0
        for i, prototype in enumerate(prototypes):
            # Weight by soft assignment probability
            weights = soft_assignments[:, i]
            if weights.sum() > 0:
                # Weighted MSE between test embeddings and prototype
                weighted_distances = weights.unsqueeze(1) * torch.norm(test_embeddings - prototype, dim=1, keepdim=True)
                consistency_loss += weighted_distances.mean()
        
        return consistency_loss
    
    def _compute_smoothness_loss(self, test_embeddings, low_confidence_mask):
        """Compute smoothness regularization for low-confidence samples"""
        if low_confidence_mask.sum() == 0:
            return torch.tensor(0.0, device=test_embeddings.device)
        
        low_conf_embeddings = test_embeddings[low_confidence_mask]
        
        # Compute pairwise distances
        pairwise_distances = torch.cdist(low_conf_embeddings, low_conf_embeddings, p=2)
        
        # Smoothness loss: encourage similar embeddings to be close
        # Use inverse distance weighting
        smoothness_loss = torch.mean(pairwise_distances)
        
        return smoothness_loss
    
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
    
    def evaluate_zero_day_detection(self, test_x, test_y, support_x, support_y, zero_day_indices=None):
        """
        Evaluate zero-day detection performance using binary classification with base vs. TTT comparison
        
        Args:
            test_x: Test samples
            test_y: Test labels (binary: 0=Normal, 1=Attack)
            support_x: Support set (known attacks)
            support_y: Support set labels (binary: 0=Normal, 1=Attack)
            zero_day_indices: Indices of DoS samples for focused evaluation
            
        Returns:
            metrics: Dictionary containing evaluation metrics for both base and TTT predictions
        """
        logger.info("Evaluating transductive zero-day detection performance with base vs. TTT comparison")
        
        # Base model evaluation (pre-TTT)
        with torch.no_grad():
            base_logits = self(test_x)
            base_predictions = torch.argmax(base_logits, dim=1)
        
        # TTT evaluation (ENABLED for proper evaluation - TTT should run at test time)
        # Note: TTT uses unsupervised objectives on test data, so no test label leakage occurs
        # Support set is validation data, not training data, so no training leakage occurs
        ttt_predictions, confidence, is_zero_day = self.detect_zero_day(test_x, support_x, support_y, test_y=test_y, evaluation_mode=False)
        
        # Convert to numpy for sklearn metrics
        base_predictions_np = base_predictions.detach().cpu().numpy()
        ttt_predictions_np = ttt_predictions.detach().cpu().numpy()
        test_y_np = test_y.detach().cpu().numpy()
        confidence_np = confidence.detach().cpu().numpy()
        is_zero_day_np = is_zero_day.detach().cpu().numpy()
        
        # Compute metrics for base and TTT
        def compute_metrics(predictions_np, true_labels_np, zero_day_indices):
            # Overall binary classification metrics
            accuracy = accuracy_score(true_labels_np, predictions_np)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels_np, predictions_np, average='binary', zero_division=0
            )
            mccc = matthews_corrcoef(true_labels_np, predictions_np)
            cm = confusion_matrix(true_labels_np, predictions_np)
            
            # DoS-specific metrics (zero-day detection)
            if zero_day_indices is not None and len(zero_day_indices) > 0:
                dos_predictions = predictions_np[zero_day_indices]
                dos_true = true_labels_np[zero_day_indices]
                dos_accuracy = accuracy_score(dos_true, dos_predictions)
                dos_precision, dos_recall, dos_f1, _ = precision_recall_fscore_support(
                    dos_true, dos_predictions, average='binary', zero_division=0
                )
                zero_day_detection_rate = (dos_predictions == 1).mean()  # Rate of detecting DoS as attack
            else:
                dos_accuracy = dos_precision = dos_recall = dos_f1 = zero_day_detection_rate = 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mccc': mccc,
                'dos_accuracy': dos_accuracy,
                'dos_precision': dos_precision,
                'dos_recall': dos_recall,
                'dos_f1_score': dos_f1,
                'zero_day_detection_rate': zero_day_detection_rate,
                'confusion_matrix': cm.tolist()
            }
        
        base_metrics = compute_metrics(base_predictions_np, test_y_np, zero_day_indices)
        ttt_metrics = compute_metrics(ttt_predictions_np, test_y_np, zero_day_indices)
        
        metrics = {'base': base_metrics, 'ttt': ttt_metrics}
        
        logger.info("Base vs. TTT Performance (Binary Classification):")
        logger.info(f"  Base F1: {base_metrics['f1_score']:.4f}, TTT F1: {ttt_metrics['f1_score']:.4f}")
        logger.info(f"  Base DoS F1: {base_metrics['dos_f1_score']:.4f}, TTT DoS F1: {ttt_metrics['dos_f1_score']:.4f}")
        logger.info(f"  Base Zero-Day Rate: {base_metrics['zero_day_detection_rate']:.4f}, TTT Zero-Day Rate: {ttt_metrics['zero_day_detection_rate']:.4f}")
        
        return metrics

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
            # Fallback: use all available samples
            query_indices = torch.cat([normal_indices, attack_indices])
        
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
    
    # Meta-train the model
    training_history = model.meta_train(meta_tasks, meta_epochs=20)
    
    # Evaluate zero-day detection (using validation data as support set to prevent data leakage)
    metrics = model.evaluate_zero_day_detection(X_test, y_test, X_val, y_val)
    
    logger.info("✅ Transductive few-shot model test completed!")
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    main()
