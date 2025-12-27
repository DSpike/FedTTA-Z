"""
Fair Binary Evaluation for Zero-Day Detection
==============================================

This module implements a fair comparison between base model and TTT-enhanced model
for zero-day attack detection using binary classification (Normal vs Attack).

Key Principle: Use the SAME trained binary model for both base and TTT evaluations.
Only the TTT adaptation is different, ensuring we measure the effect of adaptation ONLY.

Author: PhD Research
Date: 2025-01-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from models.transductive_fewshot_model import TransductiveLearner, create_meta_tasks

logger = logging.getLogger(__name__)


class FairBinaryEvaluator:
    """
    Fair binary evaluation comparing base model vs TTT-enhanced model.

    Ensures fair comparison by:
    1. Training a single binary model (Normal vs Attack)
    2. Using the SAME model for both base and TTT evaluations
    3. Only applying TTT adaptation to create the adapted version
    4. Measuring improvement from adaptation ONLY
    """

    def __init__(self, config, device='cuda'):
        """
        Initialize fair binary evaluator.

        Args:
            config: System configuration
            device: Device to run evaluation on
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.binary_model = None

        logger.info("=" * 80)
        logger.info("🔬 FAIR BINARY EVALUATION - Initialized")
        logger.info("=" * 80)
        logger.info("Evaluation Principle: Same trained model for Base and TTT")
        logger.info("Difference: TTT adaptation applied to create adapted version")
        logger.info("=" * 80)

    def train_binary_model(
        self,
        X_train: torch.Tensor,
        y_train_binary: torch.Tensor,
        y_train_multiclass: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Train a binary classification model (Normal vs Attack).

        This model will be used for BOTH base and TTT evaluations.

        Args:
            X_train: Training features [N, seq_len, input_dim]
            y_train_binary: Binary labels (0=Normal, 1=Attack)
            y_train_multiclass: Optional multiclass labels for meta-learning

        Returns:
            Trained binary model
        """
        logger.info("=" * 80)
        logger.info("🎯 TRAINING BINARY MODEL (Normal vs Attack)")
        logger.info("=" * 80)

        # Move data to device
        X_train = X_train.to(self.device)
        y_train_binary = y_train_binary.to(self.device)
        if y_train_multiclass is not None:
            y_train_multiclass = y_train_multiclass.to(self.device)

        # Log dataset statistics
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Training labels shape: {y_train_binary.shape}")

        # Class distribution
        unique, counts = torch.unique(y_train_binary, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = "Normal" if label == 0 else "Attack"
            logger.info(f"  {label_name} (class {label}): {count} samples ({100*count/len(y_train_binary):.2f}%)")

        # Create binary model
        self.binary_model = TransductiveLearner(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,  # Use same hidden_dim as main model
            embedding_dim=self.config.embedding_dim,
            num_classes=2,  # Binary: Normal vs Attack
            support_weight=self.config.support_weight,
            test_weight=self.config.test_weight,
            sequence_length=self.config.sequence_length
        ).to(self.device)

        logger.info(f"Binary model architecture:")
        logger.info(f"  Input dim: {self.config.input_dim}")
        logger.info(f"  Hidden dim: {self.config.hidden_dim}")
        logger.info(f"  Embedding dim: {self.config.embedding_dim}")
        logger.info(f"  Num classes: 2 (Binary)")

        # Create meta-tasks for binary training
        logger.info(f"\n📋 Creating binary meta-tasks for training...")
        logger.info(f"Meta-learning config - k_shot: {self.config.k_shot}, n_query: {self.config.n_query}")

        # For binary training, we don't have a "zero-day" since we're training on all attacks
        # We use n_way=2 (Normal vs Attack) for binary classification
        meta_tasks = create_meta_tasks(
            X_train,
            y_train_binary,
            n_way=2,  # Binary: Normal vs Attack
            k_shot=self.config.k_shot,
            n_query=self.config.n_query,
            n_tasks=self.config.num_meta_tasks,
            phase="training",
            normal_query_ratio=0.5,  # 50/50 split for binary
            zero_day_attack_label=None,  # No zero-day exclusion during training
            enforce_equal_support_composition=True,
            include_all_attack_types_in_support=False,
            data_y_multiclass=y_train_multiclass,
        )

        logger.info(f"✅ Created {len(meta_tasks)} binary meta-tasks")

        # Train model using meta-learning
        logger.info(f"\n🎯 Training binary model ({self.config.meta_epochs} epochs)...")
        training_history = self.binary_model.meta_train(
            meta_tasks,
            meta_epochs=self.config.meta_epochs,
            config=self.config,
            global_params=None
        )

        # Log training results
        final_loss = training_history.get('epoch_losses', [0.0])[-1]
        final_acc = training_history.get('epoch_accuracies', [0.0])[-1]

        logger.info("\n" + "=" * 80)
        logger.info("✅ BINARY MODEL TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Final training loss: {final_loss:.4f}")
        logger.info(f"Final validation accuracy: {final_acc:.4f}")
        logger.info("=" * 80)

        return self.binary_model

    def evaluate_base_model(
        self,
        X_test: torch.Tensor,
        y_test_binary: torch.Tensor,
        zero_day_mask: torch.Tensor
    ) -> Dict:
        """
        Evaluate base model WITHOUT TTT adaptation.

        This is the baseline performance using the trained model as-is.

        Args:
            X_test: Test features
            y_test_binary: Binary test labels
            zero_day_mask: Boolean mask indicating zero-day samples

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("📊 EVALUATING BASE MODEL (No TTT Adaptation)")
        logger.info("=" * 80)

        if self.binary_model is None:
            raise ValueError("Binary model not trained. Call train_binary_model() first.")

        # Move data to device
        X_test = X_test.to(self.device)
        y_test_binary = y_test_binary.to(self.device)
        zero_day_mask = zero_day_mask.to(self.device)

        # Set model to evaluation mode
        self.binary_model.eval()

        with torch.no_grad():
            # Get predictions from base model (no adaptation)
            logits = self.binary_model(X_test)

            # Ensure logits are 2-dimensional for binary classification
            if logits.shape[-1] != 2:
                logger.warning(f"⚠️ Model output has {logits.shape[-1]} classes, expected 2")
                logger.warning(f"   Taking first 2 dimensions for binary classification")
                logits = logits[:, :2]

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Attack probabilities (class 1)
            attack_probs = probabilities[:, 1]

        # Convert to numpy for metrics calculation
        y_true = y_test_binary.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        attack_probs_np = attack_probs.cpu().numpy()
        zero_day_mask_np = zero_day_mask.cpu().numpy()

        # Validate binary predictions (should only have 0 and 1)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) > 2:
            logger.warning(f"⚠️ Predictions have {len(unique_preds)} classes: {unique_preds}")
            logger.warning(f"   Converting to binary (0 vs 1)")
            # Force to binary if somehow we have more classes
            y_pred = (y_pred > 0).astype(int)

        # Ensure labels are binary
        unique_labels = np.unique(y_true)
        if len(unique_labels) > 2:
            logger.warning(f"⚠️ Labels have {len(unique_labels)} classes: {unique_labels}")
            logger.warning(f"   Converting to binary (0 vs 1)")
            y_true = (y_true > 0).astype(int)

        # Calculate overall metrics (with binary average for binary classification)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

        # ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(y_true, attack_probs_np)
            pr_auc = average_precision_score(y_true, attack_probs_np)
        except:
            roc_auc = 0.5
            pr_auc = 0.5

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calculate FAR (False Alarm Rate)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Zero-day specific metrics
        if zero_day_mask_np.sum() > 0:
            zero_day_true = y_true[zero_day_mask_np]
            zero_day_pred = y_pred[zero_day_mask_np]

            # Zero-day detection rate (recall for zero-day attacks)
            zero_day_attacks = zero_day_true == 1
            if zero_day_attacks.sum() > 0:
                zero_day_detected = zero_day_pred[zero_day_attacks] == 1
                zero_day_detection_rate = zero_day_detected.sum() / zero_day_attacks.sum()
            else:
                zero_day_detection_rate = 0.0

            # Zero-day accuracy
            zero_day_accuracy = accuracy_score(zero_day_true, zero_day_pred)
            zero_day_f1 = f1_score(zero_day_true, zero_day_pred, average='binary', zero_division=0)
        else:
            zero_day_detection_rate = 0.0
            zero_day_accuracy = 0.0
            zero_day_f1 = 0.0

        # Log results
        logger.info(f"\n📊 Base Model Results:")
        logger.info(f"Overall Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  FAR: {far:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn}, FP: {fp}")
        logger.info(f"  FN: {fn}, TP: {tp}")
        logger.info(f"\nZero-Day Performance:")
        logger.info(f"  Zero-day samples: {zero_day_mask_np.sum()}")
        logger.info(f"  Zero-day Detection Rate: {zero_day_detection_rate:.4f}")
        logger.info(f"  Zero-day Accuracy: {zero_day_accuracy:.4f}")
        logger.info(f"  Zero-day F1-Score: {zero_day_f1:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'far': far,
            'confusion_matrix': cm,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'zero_day_detection_rate': zero_day_detection_rate,
            'zero_day_accuracy': zero_day_accuracy,
            'zero_day_f1': zero_day_f1,
            'zero_day_samples': int(zero_day_mask_np.sum()),
            'predictions': y_pred,
            'attack_probabilities': attack_probs_np
        }

    def apply_ttt_adaptation(
        self,
        X_test: torch.Tensor,
        support_ratio: float = 0.3
    ) -> nn.Module:
        """
        Apply Test-Time Training (TTT) adaptation to the binary model.

        This creates an adapted version of the base model using unsupervised
        test-time training with entropy minimization.

        Args:
            X_test: Test features for adaptation
            support_ratio: Ratio of test data to use for adaptation support

        Returns:
            Adapted binary model
        """
        logger.info("=" * 80)
        logger.info("🔄 APPLYING TTT ADAPTATION TO BINARY MODEL")
        logger.info("=" * 80)

        if self.binary_model is None:
            raise ValueError("Binary model not trained. Call train_binary_model() first.")

        # Create a copy of the model for adaptation (preserve base model)
        # Use state_dict to avoid deepcopy issues with PyTorch models
        adapted_model = TransductiveLearner(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim,
            num_classes=2,  # Binary
            support_weight=self.config.support_weight,
            test_weight=self.config.test_weight,
            sequence_length=self.config.sequence_length
        ).to(self.device)

        # Copy weights from base model using state_dict (safer than deepcopy)
        adapted_model.load_state_dict(self.binary_model.state_dict())
        adapted_model.train()  # Set to training mode for BatchNorm adaptation

        # Move data to device
        X_test = X_test.to(self.device)

        # Sample support set from test data
        support_size = int(len(X_test) * support_ratio)
        support_size = max(support_size, 100)  # At least 100 samples
        support_size = min(support_size, len(X_test) // 2)  # At most half

        indices = torch.randperm(len(X_test))[:support_size]
        X_support = X_test[indices]

        logger.info(f"TTT Adaptation Setup:")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Support samples: {len(X_support)} ({100*support_ratio:.1f}%)")
        logger.info(f"  TTT learning rate: {self.config.ttt_lr}")
        logger.info(f"  TTT steps: {self.config.ttt_base_steps}")
        logger.info(f"  Entropy weight: {self.config.entropy_weight}")
        logger.info(f"  L2 regularization: {self.config.ttt_l2_reg_weight}")

        # Setup optimizer for TTT (only adapt BatchNorm and classifier)
        # CRITICAL: Freeze feature extractor, only adapt normalization and classifier
        params_to_adapt = []
        for name, param in adapted_model.named_parameters():
            if 'bn' in name.lower() or 'batchnorm' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
                params_to_adapt.append(param)
            else:
                param.requires_grad = False

        optimizer = torch.optim.Adam(params_to_adapt, lr=self.config.ttt_lr)

        logger.info(f"  Adapting {len(params_to_adapt)} parameter groups (BatchNorm + Classifier)")

        # TTT adaptation loop
        adaptation_losses = []

        for step in range(self.config.ttt_base_steps):
            optimizer.zero_grad()

            # Forward pass
            logits = adapted_model(X_support)
            probs = torch.softmax(logits, dim=1)

            # Entropy minimization loss (unsupervised)
            entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

            # L2 regularization (stay close to base model)
            l2_loss = 0.0
            for param_adapted, param_base in zip(adapted_model.parameters(), self.binary_model.parameters()):
                if param_adapted.requires_grad:
                    l2_loss += ((param_adapted - param_base) ** 2).sum()

            # Total loss
            loss = (
                self.config.entropy_weight * entropy_loss +
                self.config.ttt_l2_reg_weight * l2_loss
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            adaptation_losses.append(loss.item())

            if (step + 1) % 20 == 0:
                logger.info(f"  Step {step+1}/{self.config.ttt_base_steps}: Loss={loss.item():.4f}, Entropy={entropy_loss.item():.4f}")

        # Set to evaluation mode
        adapted_model.eval()

        final_loss = adaptation_losses[-1] if adaptation_losses else 0.0
        logger.info(f"\n✅ TTT Adaptation Completed")
        logger.info(f"  Final loss: {final_loss:.4f}")
        logger.info(f"  Average loss: {np.mean(adaptation_losses):.4f}")
        logger.info("=" * 80)

        return adapted_model

    def evaluate_ttt_model(
        self,
        adapted_model: nn.Module,
        X_test: torch.Tensor,
        y_test_binary: torch.Tensor,
        zero_day_mask: torch.Tensor
    ) -> Dict:
        """
        Evaluate TTT-adapted model.

        This measures performance after TTT adaptation.

        Args:
            adapted_model: TTT-adapted binary model
            X_test: Test features
            y_test_binary: Binary test labels
            zero_day_mask: Boolean mask indicating zero-day samples

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("📊 EVALUATING TTT-ADAPTED MODEL")
        logger.info("=" * 80)

        # Move data to device
        X_test = X_test.to(self.device)
        y_test_binary = y_test_binary.to(self.device)
        zero_day_mask = zero_day_mask.to(self.device)

        # Set model to evaluation mode
        adapted_model.eval()

        with torch.no_grad():
            # Get predictions from adapted model
            logits = adapted_model(X_test)

            # Ensure logits are 2-dimensional for binary classification
            if logits.shape[-1] != 2:
                logger.warning(f"⚠️ Adapted model output has {logits.shape[-1]} classes, expected 2")
                logger.warning(f"   Taking first 2 dimensions for binary classification")
                logits = logits[:, :2]

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Attack probabilities (class 1)
            attack_probs = probabilities[:, 1]

        # Convert to numpy for metrics calculation
        y_true = y_test_binary.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        attack_probs_np = attack_probs.cpu().numpy()
        zero_day_mask_np = zero_day_mask.cpu().numpy()

        # Validate binary predictions (should only have 0 and 1)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) > 2:
            logger.warning(f"⚠️ Predictions have {len(unique_preds)} classes: {unique_preds}")
            logger.warning(f"   Converting to binary (0 vs 1)")
            # Force to binary if somehow we have more classes
            y_pred = (y_pred > 0).astype(int)

        # Ensure labels are binary
        unique_labels = np.unique(y_true)
        if len(unique_labels) > 2:
            logger.warning(f"⚠️ Labels have {len(unique_labels)} classes: {unique_labels}")
            logger.warning(f"   Converting to binary (0 vs 1)")
            y_true = (y_true > 0).astype(int)

        # Calculate overall metrics (with binary average for binary classification)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

        # ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(y_true, attack_probs_np)
            pr_auc = average_precision_score(y_true, attack_probs_np)
        except:
            roc_auc = 0.5
            pr_auc = 0.5

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calculate FAR (False Alarm Rate)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Zero-day specific metrics
        if zero_day_mask_np.sum() > 0:
            zero_day_true = y_true[zero_day_mask_np]
            zero_day_pred = y_pred[zero_day_mask_np]

            # Zero-day detection rate (recall for zero-day attacks)
            zero_day_attacks = zero_day_true == 1
            if zero_day_attacks.sum() > 0:
                zero_day_detected = zero_day_pred[zero_day_attacks] == 1
                zero_day_detection_rate = zero_day_detected.sum() / zero_day_attacks.sum()
            else:
                zero_day_detection_rate = 0.0

            # Zero-day accuracy
            zero_day_accuracy = accuracy_score(zero_day_true, zero_day_pred)
            zero_day_f1 = f1_score(zero_day_true, zero_day_pred, average='binary', zero_division=0)
        else:
            zero_day_detection_rate = 0.0
            zero_day_accuracy = 0.0
            zero_day_f1 = 0.0

        # Log results
        logger.info(f"\n📊 TTT Model Results:")
        logger.info(f"Overall Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  FAR: {far:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn}, FP: {fp}")
        logger.info(f"  FN: {fn}, TP: {tp}")
        logger.info(f"\nZero-Day Performance:")
        logger.info(f"  Zero-day samples: {zero_day_mask_np.sum()}")
        logger.info(f"  Zero-day Detection Rate: {zero_day_detection_rate:.4f}")
        logger.info(f"  Zero-day Accuracy: {zero_day_accuracy:.4f}")
        logger.info(f"  Zero-day F1-Score: {zero_day_f1:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'far': far,
            'confusion_matrix': cm,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'zero_day_detection_rate': zero_day_detection_rate,
            'zero_day_accuracy': zero_day_accuracy,
            'zero_day_f1': zero_day_f1,
            'zero_day_samples': int(zero_day_mask_np.sum()),
            'predictions': y_pred,
            'attack_probabilities': attack_probs_np
        }

    def compare_results(
        self,
        base_results: Dict,
        ttt_results: Dict
    ) -> Dict:
        """
        Compare base and TTT results, compute improvements.

        Args:
            base_results: Results from base model evaluation
            ttt_results: Results from TTT model evaluation

        Returns:
            Dictionary with comparison metrics
        """
        logger.info("=" * 80)
        logger.info("📊 BASE VS TTT COMPARISON")
        logger.info("=" * 80)

        # Compute improvements
        metrics_to_compare = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'pr_auc', 'zero_day_detection_rate',
            'zero_day_accuracy', 'zero_day_f1'
        ]

        comparison = {}

        logger.info(f"\n{'Metric':<30} {'Base':<12} {'TTT':<12} {'Improvement':<12}")
        logger.info("-" * 80)

        for metric in metrics_to_compare:
            base_val = base_results.get(metric, 0.0)
            ttt_val = ttt_results.get(metric, 0.0)
            improvement = ttt_val - base_val
            improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0.0

            comparison[f'{metric}_base'] = base_val
            comparison[f'{metric}_ttt'] = ttt_val
            comparison[f'{metric}_improvement'] = improvement
            comparison[f'{metric}_improvement_pct'] = improvement_pct

            # Format metric name for display
            display_name = metric.replace('_', ' ').title()

            # Color coding for improvement
            if improvement > 0:
                symbol = "✅"
            elif improvement < 0:
                symbol = "❌"
            else:
                symbol = "⚪"

            logger.info(
                f"{display_name:<30} {base_val:>11.4f} {ttt_val:>11.4f} "
                f"{symbol} {improvement:>+.4f} ({improvement_pct:>+.2f}%)"
            )

        # FAR comparison (lower is better)
        far_base = base_results.get('far', 0.0)
        far_ttt = ttt_results.get('far', 0.0)
        far_reduction = far_base - far_ttt
        far_reduction_pct = (far_reduction / far_base * 100) if far_base > 0 else 0.0

        comparison['far_base'] = far_base
        comparison['far_ttt'] = far_ttt
        comparison['far_reduction'] = far_reduction
        comparison['far_reduction_pct'] = far_reduction_pct

        far_symbol = "✅" if far_reduction > 0 else ("❌" if far_reduction < 0 else "⚪")
        logger.info(
            f"{'FAR (Lower is Better)':<30} {far_base:>11.4f} {far_ttt:>11.4f} "
            f"{far_symbol} {far_reduction:>+.4f} ({far_reduction_pct:>+.2f}%)"
        )

        logger.info("=" * 80)
        logger.info("✅ FAIR COMPARISON COMPLETED")
        logger.info("=" * 80)

        return comparison

    def run_full_evaluation(
        self,
        X_train: torch.Tensor,
        y_train_binary: torch.Tensor,
        X_test: torch.Tensor,
        y_test_binary: torch.Tensor,
        zero_day_mask: torch.Tensor,
        y_train_multiclass: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Run complete fair binary evaluation pipeline.

        Steps:
        1. Train binary model
        2. Evaluate base model (no adaptation)
        3. Apply TTT adaptation
        4. Evaluate TTT model
        5. Compare results

        Args:
            X_train: Training features
            y_train_binary: Binary training labels
            X_test: Test features
            y_test_binary: Binary test labels
            zero_day_mask: Boolean mask for zero-day samples
            y_train_multiclass: Optional multiclass labels

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING FULL FAIR BINARY EVALUATION PIPELINE")
        logger.info("=" * 80)

        # Step 1: Train binary model
        self.train_binary_model(X_train, y_train_binary, y_train_multiclass)

        # Step 2: Evaluate base model
        base_results = self.evaluate_base_model(X_test, y_test_binary, zero_day_mask)

        # Step 3: Apply TTT adaptation
        adapted_model = self.apply_ttt_adaptation(X_test, support_ratio=0.3)

        # Step 4: Evaluate TTT model
        ttt_results = self.evaluate_ttt_model(adapted_model, X_test, y_test_binary, zero_day_mask)

        # Step 5: Compare results
        comparison = self.compare_results(base_results, ttt_results)

        logger.info("=" * 80)
        logger.info("✅ FULL FAIR BINARY EVALUATION COMPLETED")
        logger.info("=" * 80)

        return {
            'base_results': base_results,
            'ttt_results': ttt_results,
            'comparison': comparison
        }
