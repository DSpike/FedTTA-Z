"""
Temperature Scaling for TTT Model Calibration

Reduces False Alarm Rate (FAR) by calibrating overconfident predictions
after TTT adaptation using Guo et al. 2017 method.

Key Idea:
- TTT entropy minimization makes model overconfident (median prob = 0.976)
- Temperature scaling T>1 softens probabilities without retraining
- Find optimal T that minimizes calibration error while maintaining ZDR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for Post-hoc Calibration

    Reference: Guo et al. "On Calibration of Modern Neural Networks" ICML 2017

    Temperature T is learned to minimize NLL on validation set.
    - T=1: No change
    - T>1: Softer probabilities (less confident) ← We need this for TTT
    - T<1: Sharper probabilities (more confident)
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        # Temperature is a learnable parameter
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature

        Args:
            logits: Raw model outputs (before softmax)

        Returns:
            calibrated_probs: Temperature-scaled probabilities
        """
        return torch.softmax(logits / self.temperature, dim=1)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = True
    ) -> float:
        """
        Learn optimal temperature on validation set

        Args:
            logits: Validation set logits (N, num_classes)
            labels: Ground truth labels (N,)
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            verbose: Print optimization progress

        Returns:
            optimal_temperature: Learned temperature value
        """
        # Freeze temperature for gradient descent
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        # Optimize temperature
        optimizer.step(eval_loss)
        optimal_temp = self.temperature.item()

        if verbose:
            logger.info(f"✅ Optimal temperature found: T={optimal_temp:.3f}")

            # Compute calibration metrics
            with torch.no_grad():
                before_probs = torch.softmax(logits, dim=1)
                after_probs = self.forward(logits)

                before_nll = F.cross_entropy(logits, labels).item()
                after_nll = F.cross_entropy(logits / self.temperature, labels).item()

                logger.info(f"   NLL before: {before_nll:.4f}")
                logger.info(f"   NLL after:  {after_nll:.4f} (improvement: {before_nll - after_nll:.4f})")

        return optimal_temp


def find_optimal_temperature_grid_search(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature_range: np.ndarray = np.arange(1.0, 4.0, 0.1),
    target_metric: str = 'far',
    target_far: float = 0.10,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Find optimal temperature via grid search to minimize FAR while maintaining ZDR

    Args:
        logits: Model logits (N, num_classes)
        labels: Ground truth binary labels (N,) - 0=normal, 1=attack
        temperature_range: Array of temperatures to try
        target_metric: Metric to optimize ('far', 'nll', 'ece')
        target_far: Target FAR constraint (e.g., 0.10 for 10% FAR)
        verbose: Print search progress

    Returns:
        optimal_temp: Best temperature value
        results: Dict with metrics for each temperature
    """
    device = logits.device
    results = []

    # Convert to binary if needed
    if labels.max() > 1:
        labels_binary = (labels > 0).long()
    else:
        labels_binary = labels

    logger.info(f"🔍 Grid searching optimal temperature (target FAR < {target_far*100:.1f}%)...")

    best_temp = 1.0
    best_score = float('inf')

    for temp in temperature_range:
        # Apply temperature scaling
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=1)

        # Get attack probabilities (class 1 for binary)
        if probs.shape[1] == 2:
            attack_probs = probs[:, 1]
        else:
            attack_probs = 1.0 - probs[:, 0]

        # Predictions with default threshold (0.5)
        predictions = (attack_probs >= 0.5).long()

        # Calculate metrics
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

        cm = confusion_matrix(labels_binary.cpu(), predictions.cpu())
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            zdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * zdr / (precision + zdr) if (precision + zdr) > 0 else 0.0
        else:
            far = 0.0
            zdr = 0.0
            accuracy = 0.0
            precision = 0.0
            f1 = 0.0

        # Negative log-likelihood
        nll = F.cross_entropy(scaled_logits, labels_binary).item()

        # Expected Calibration Error (simplified)
        conf, pred_labels = probs.max(dim=1)
        correct = (pred_labels == labels_binary).float()
        ece = torch.abs(conf - correct).mean().item()

        # Store results
        results.append({
            'temperature': temp,
            'far': far,
            'zdr': zdr,
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'nll': nll,
            'ece': ece
        })

        # Score based on target metric
        if target_metric == 'far':
            # Minimize FAR while maintaining ZDR > 0.90
            if zdr >= 0.90:
                score = far  # Lower is better
            else:
                score = float('inf')  # Penalize low ZDR
        elif target_metric == 'nll':
            score = nll
        elif target_metric == 'ece':
            score = ece
        else:
            score = far

        # Update best temperature
        if score < best_score:
            best_score = score
            best_temp = temp

        if verbose and temp in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            logger.info(f"   T={temp:.1f}: FAR={far*100:.2f}%, ZDR={zdr*100:.2f}%, "
                       f"Acc={accuracy*100:.2f}%, F1={f1*100:.2f}%, NLL={nll:.4f}")

    # Get best result
    best_result = [r for r in results if r['temperature'] == best_temp][0]

    logger.info(f"\n✅ Optimal temperature: T={best_temp:.2f}")
    logger.info(f"   FAR: {best_result['far']*100:.2f}% (target: <{target_far*100:.1f}%)")
    logger.info(f"   ZDR: {best_result['zdr']*100:.2f}%")
    logger.info(f"   Accuracy: {best_result['accuracy']*100:.2f}%")
    logger.info(f"   F1-Score: {best_result['f1']*100:.2f}%")
    logger.info(f"   NLL: {best_result['nll']:.4f}")

    return best_temp, {'results': results, 'best': best_result}


def find_optimal_threshold_given_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    target_far: float = 0.10,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Find optimal decision threshold to achieve target FAR

    Args:
        logits: Model logits (N, num_classes)
        labels: Ground truth binary labels (N,)
        temperature: Temperature for scaling
        target_far: Target FAR (e.g., 0.10 for 10%)
        verbose: Print results

    Returns:
        optimal_threshold: Threshold achieving target FAR
        metrics: Performance metrics at this threshold
    """
    # Apply temperature
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1)

    # Get attack probabilities
    if probs.shape[1] == 2:
        attack_probs = probs[:, 1].cpu().numpy()
    else:
        attack_probs = (1.0 - probs[:, 0]).cpu().numpy()

    # Convert labels to binary
    if labels.max() > 1:
        labels_binary = (labels > 0).long().cpu().numpy()
    else:
        labels_binary = labels.cpu().numpy()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels_binary, attack_probs)

    # Find threshold closest to target FAR
    idx = np.argmin(np.abs(fpr - target_far))
    optimal_threshold = thresholds[idx]
    achieved_far = fpr[idx]
    achieved_zdr = tpr[idx]

    # Compute metrics at this threshold
    predictions = (attack_probs >= optimal_threshold).astype(int)

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

    cm = confusion_matrix(labels_binary, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        accuracy = precision = recall = f1 = 0.0

    metrics = {
        'threshold': optimal_threshold,
        'far': achieved_far,
        'zdr': achieved_zdr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    if verbose:
        logger.info(f"\n✅ Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"   FAR: {achieved_far*100:.2f}% (target: {target_far*100:.1f}%)")
        logger.info(f"   ZDR: {achieved_zdr*100:.2f}%")
        logger.info(f"   Accuracy: {accuracy*100:.2f}%")
        logger.info(f"   Precision: {precision*100:.2f}%")
        logger.info(f"   F1-Score: {f1*100:.2f}%")

    return optimal_threshold, metrics


def calibrate_ttt_model(
    model: nn.Module,
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    method: str = 'grid_search',
    target_far: float = 0.10,
    verbose: bool = True
) -> Dict:
    """
    Main calibration function - calibrates TTT model to reduce FAR

    Args:
        model: TTT adapted model
        val_logits: Validation set logits
        val_labels: Validation set labels
        method: Calibration method ('grid_search', 'gradient', 'threshold_only')
        target_far: Target FAR constraint
        verbose: Print progress

    Returns:
        calibration_params: Dict with optimal temperature and threshold
    """
    logger.info("=" * 80)
    logger.info("🔧 TTT MODEL CALIBRATION - Reducing FAR while maintaining ZDR")
    logger.info("=" * 80)

    if method == 'grid_search':
        # Grid search for optimal temperature
        optimal_temp, temp_results = find_optimal_temperature_grid_search(
            val_logits, val_labels,
            temperature_range=np.arange(1.0, 4.0, 0.2),
            target_far=target_far,
            verbose=verbose
        )

        # Find optimal threshold given this temperature
        optimal_threshold, threshold_metrics = find_optimal_threshold_given_temperature(
            val_logits, val_labels,
            temperature=optimal_temp,
            target_far=target_far,
            verbose=verbose
        )

    elif method == 'gradient':
        # Gradient-based temperature optimization
        temp_scaler = TemperatureScaling(initial_temperature=1.5)
        optimal_temp = temp_scaler.fit(val_logits, val_labels, verbose=verbose)

        # Find threshold
        optimal_threshold, threshold_metrics = find_optimal_threshold_given_temperature(
            val_logits, val_labels,
            temperature=optimal_temp,
            target_far=target_far,
            verbose=verbose
        )

    elif method == 'threshold_only':
        # Just optimize threshold, no temperature scaling
        optimal_temp = 1.0
        optimal_threshold, threshold_metrics = find_optimal_threshold_given_temperature(
            val_logits, val_labels,
            temperature=1.0,
            target_far=target_far,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibration_params = {
        'temperature': optimal_temp,
        'threshold': optimal_threshold,
        'far': threshold_metrics['far'],
        'zdr': threshold_metrics['zdr'],
        'accuracy': threshold_metrics['accuracy'],
        'f1': threshold_metrics['f1'],
        'method': method
    }

    logger.info("\n" + "=" * 80)
    logger.info("📊 CALIBRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Method: {method}")
    logger.info(f"Temperature: {optimal_temp:.3f}")
    logger.info(f"Threshold: {optimal_threshold:.4f}")
    logger.info(f"FAR: {threshold_metrics['far']*100:.2f}% (target: <{target_far*100:.1f}%)")
    logger.info(f"ZDR: {threshold_metrics['zdr']*100:.2f}%")
    logger.info(f"Accuracy: {threshold_metrics['accuracy']*100:.2f}%")
    logger.info(f"F1-Score: {threshold_metrics['f1']*100:.2f}%")
    logger.info("=" * 80)

    return calibration_params


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Simulate overconfident TTT logits
    torch.manual_seed(42)
    n_samples = 1000
    n_classes = 2

    # Overconfident logits (high magnitude)
    logits = torch.randn(n_samples, n_classes) * 5.0  # High magnitude = overconfident
    labels = torch.randint(0, n_classes, (n_samples,))

    # Calibrate
    params = calibrate_ttt_model(
        model=None,  # Not needed for this function
        val_logits=logits,
        val_labels=labels,
        method='grid_search',
        target_far=0.10,
        verbose=True
    )

    print(f"\nCalibration parameters: {params}")
