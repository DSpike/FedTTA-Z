"""
Base + TTT Ensemble Predictor

Combines predictions from Base Model (low FAR) and TTT Model (high ZDR)
to achieve better FAR-ZDR trade-off.

Strategy Options:
1. Weighted Probability: alpha * base_prob + (1-alpha) * ttt_prob
2. Confidence-Weighted: Use base when confident about normal, TTT otherwise
3. Voting: Require agreement between models

Goal: Reduce FAR from 42.53% to <20% while maintaining ZDR >88%
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class BaseTTTEnsemble:
    """
    Ensemble predictor combining Base and TTT models
    """

    def __init__(
        self,
        ensemble_method: str = 'confidence_weighted',
        base_weight: float = 0.4,
        base_confidence_threshold: float = 0.85,
        ttt_confidence_threshold: float = 0.70,
        decision_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize ensemble predictor

        Args:
            ensemble_method: 'weighted_prob', 'confidence_weighted', or 'voting'
            base_weight: Weight for base model in weighted_prob method (0-1)
            base_confidence_threshold: Min confidence for trusting base "normal" predictions
            ttt_confidence_threshold: Min confidence for trusting TTT "attack" predictions
            decision_threshold: Threshold for final binary decision
            verbose: Print statistics
        """
        self.ensemble_method = ensemble_method
        self.base_weight = base_weight
        self.base_confidence_threshold = base_confidence_threshold
        self.ttt_confidence_threshold = ttt_confidence_threshold
        self.decision_threshold = decision_threshold
        self.verbose = verbose

        if verbose:
            logger.info(f"🎯 Initialized BaseTTTEnsemble:")
            logger.info(f"   Method: {ensemble_method}")
            if ensemble_method == 'weighted_prob':
                logger.info(f"   Base weight: {base_weight:.2f}, TTT weight: {1-base_weight:.2f}")
            elif ensemble_method == 'confidence_weighted':
                logger.info(f"   Base conf threshold: {base_confidence_threshold:.2f}")
                logger.info(f"   TTT conf threshold: {ttt_confidence_threshold:.2f}")
            logger.info(f"   Decision threshold: {decision_threshold:.2f}")

    def predict(
        self,
        base_probs: torch.Tensor,
        ttt_probs: torch.Tensor,
        attack_class_idx: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute ensemble predictions

        Args:
            base_probs: Base model probabilities (N, num_classes) - torch.Tensor or numpy.ndarray
            ttt_probs: TTT model probabilities (N, num_classes) - torch.Tensor or numpy.ndarray
            attack_class_idx: Index of attack class (1 for binary)

        Returns:
            predictions: Binary predictions (N,) - numpy.ndarray
            probs: Ensemble probabilities (N, num_classes) - numpy.ndarray
            stats: Statistics about ensemble decisions
        """
        # Convert to torch tensors if numpy arrays
        if isinstance(base_probs, np.ndarray):
            base_probs = torch.from_numpy(base_probs).float()
        if isinstance(ttt_probs, np.ndarray):
            ttt_probs = torch.from_numpy(ttt_probs).float()

        device = base_probs.device
        n_samples = base_probs.shape[0]

        if self.ensemble_method == 'weighted_prob':
            return self._weighted_probability(base_probs, ttt_probs, attack_class_idx)

        elif self.ensemble_method == 'confidence_weighted':
            return self._confidence_weighted(base_probs, ttt_probs, attack_class_idx)

        elif self.ensemble_method == 'voting':
            return self._voting(base_probs, ttt_probs, attack_class_idx)

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _weighted_probability(
        self,
        base_probs: torch.Tensor,
        ttt_probs: torch.Tensor,
        attack_class_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Weighted probability ensemble: alpha * base + (1-alpha) * ttt
        """
        # Weighted combination
        ensemble_probs = self.base_weight * base_probs + (1 - self.base_weight) * ttt_probs

        # Get attack probabilities
        if ensemble_probs.shape[1] == 2:
            attack_probs = ensemble_probs[:, attack_class_idx]
        else:
            attack_probs = 1.0 - ensemble_probs[:, 0]

        # Binary predictions
        predictions = (attack_probs >= self.decision_threshold).long()

        stats = {
            'method': 'weighted_prob',
            'base_weight': self.base_weight,
            'ttt_weight': 1 - self.base_weight,
            'decision_threshold': self.decision_threshold,
            'total_samples': len(predictions)
        }

        if self.verbose:
            logger.info(f"📊 Weighted Probability Ensemble:")
            logger.info(f"   Base weight: {self.base_weight:.2f}, TTT weight: {1-self.base_weight:.2f}")
            logger.info(f"   Predicted attacks: {predictions.sum().item()}/{len(predictions)} ({predictions.float().mean()*100:.1f}%)")

        return predictions, ensemble_probs, stats

    def _confidence_weighted(
        self,
        base_probs: torch.Tensor,
        ttt_probs: torch.Tensor,
        attack_class_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Confidence-weighted ensemble:
        - Use base when VERY confident about normal (reduces FAR)
        - Use TTT when confident about attack (maintains ZDR)
        - Use weighted average for uncertain region
        """
        device = base_probs.device
        n_samples = base_probs.shape[0]

        # Get predictions and confidences
        base_conf, base_pred = base_probs.max(dim=1)
        ttt_conf, ttt_pred = ttt_probs.max(dim=1)

        # Zone 1: Base model VERY confident it's normal
        # Use base to filter out obvious normal samples (reduces FAR)
        base_confident_normal = (base_conf >= self.base_confidence_threshold) & (base_pred == 0)

        # Zone 2: TTT model confident it's attack
        # Use TTT to detect attacks including zero-day (maintains ZDR)
        ttt_confident_attack = (ttt_conf >= self.ttt_confidence_threshold) & (ttt_pred == attack_class_idx)

        # Zone 3: Uncertain region
        # Use weighted ensemble
        uncertain_region = ~(base_confident_normal | ttt_confident_attack)

        # Initialize predictions
        ensemble_pred = torch.zeros(n_samples, dtype=torch.long, device=device)
        ensemble_probs = base_probs.clone()

        # Zone 1: Trust base model (normal)
        ensemble_pred[base_confident_normal] = 0
        ensemble_probs[base_confident_normal] = base_probs[base_confident_normal]

        # Zone 2: Trust TTT model (attack)
        ensemble_pred[ttt_confident_attack] = attack_class_idx
        ensemble_probs[ttt_confident_attack] = ttt_probs[ttt_confident_attack]

        # Zone 3: Weighted average (favor TTT slightly for better recall)
        if uncertain_region.sum() > 0:
            uncertain_probs = 0.4 * base_probs[uncertain_region] + 0.6 * ttt_probs[uncertain_region]
            ensemble_probs[uncertain_region] = uncertain_probs

            # Get attack probability for decision
            if uncertain_probs.shape[1] == 2:
                attack_probs_uncertain = uncertain_probs[:, attack_class_idx]
            else:
                attack_probs_uncertain = 1.0 - uncertain_probs[:, 0]

            ensemble_pred[uncertain_region] = (attack_probs_uncertain >= self.decision_threshold).long()

        stats = {
            'method': 'confidence_weighted',
            'base_conf_threshold': self.base_confidence_threshold,
            'ttt_conf_threshold': self.ttt_confidence_threshold,
            'total_samples': n_samples,
            'zone1_base_normal': base_confident_normal.sum().item(),
            'zone2_ttt_attack': ttt_confident_attack.sum().item(),
            'zone3_uncertain': uncertain_region.sum().item(),
            'zone1_pct': 100.0 * base_confident_normal.sum().item() / n_samples,
            'zone2_pct': 100.0 * ttt_confident_attack.sum().item() / n_samples,
            'zone3_pct': 100.0 * uncertain_region.sum().item() / n_samples
        }

        if self.verbose:
            logger.info(f"📊 Confidence-Weighted Ensemble:")
            logger.info(f"   Zone 1 (Base confident normal): {stats['zone1_base_normal']} ({stats['zone1_pct']:.1f}%)")
            logger.info(f"   Zone 2 (TTT confident attack):  {stats['zone2_ttt_attack']} ({stats['zone2_pct']:.1f}%)")
            logger.info(f"   Zone 3 (Uncertain - weighted):  {stats['zone3_uncertain']} ({stats['zone3_pct']:.1f}%)")
            logger.info(f"   Predicted attacks: {ensemble_pred.sum().item()}/{n_samples} ({ensemble_pred.float().mean()*100:.1f}%)")

        return ensemble_pred, ensemble_probs, stats

    def _voting(
        self,
        base_probs: torch.Tensor,
        ttt_probs: torch.Tensor,
        attack_class_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Voting ensemble:
        - Predict attack if EITHER model predicts attack (high recall)
        - Predict normal only if BOTH predict normal (conservative)
        """
        # Get predictions
        base_pred = base_probs.argmax(dim=1)
        ttt_pred = ttt_probs.argmax(dim=1)

        # Attack if EITHER predicts attack
        ensemble_pred = ((base_pred == attack_class_idx) | (ttt_pred == attack_class_idx)).long()

        # Ensemble probabilities: max of both models
        ensemble_probs = torch.max(base_probs, ttt_probs)

        # Statistics
        both_attack = (base_pred == attack_class_idx) & (ttt_pred == attack_class_idx)
        both_normal = (base_pred == 0) & (ttt_pred == 0)
        disagreement = ~(both_attack | both_normal)

        stats = {
            'method': 'voting',
            'total_samples': len(ensemble_pred),
            'both_attack': both_attack.sum().item(),
            'both_normal': both_normal.sum().item(),
            'disagreement': disagreement.sum().item(),
            'agreement_rate': (both_attack.sum() + both_normal.sum()).float().item() / len(ensemble_pred)
        }

        if self.verbose:
            logger.info(f"📊 Voting Ensemble:")
            logger.info(f"   Both predict attack: {stats['both_attack']} ({100*stats['both_attack']/len(ensemble_pred):.1f}%)")
            logger.info(f"   Both predict normal: {stats['both_normal']} ({100*stats['both_normal']/len(ensemble_pred):.1f}%)")
            logger.info(f"   Disagreement: {stats['disagreement']} ({100*stats['disagreement']/len(ensemble_pred):.1f}%)")
            logger.info(f"   Agreement rate: {stats['agreement_rate']*100:.1f}%")
            logger.info(f"   Predicted attacks: {ensemble_pred.sum().item()}/{len(ensemble_pred)} ({ensemble_pred.float().mean()*100:.1f}%)")

        return ensemble_pred, ensemble_probs, stats


def evaluate_ensemble(
    base_probs: torch.Tensor,
    ttt_probs: torch.Tensor,
    labels: torch.Tensor,
    ensemble_method: str = 'confidence_weighted',
    **ensemble_kwargs
) -> Dict:
    """
    Evaluate ensemble performance

    Args:
        base_probs: Base model probabilities (N, num_classes)
        ttt_probs: TTT model probabilities (N, num_classes)
        labels: Ground truth labels (N,)
        ensemble_method: Ensemble method to use
        **ensemble_kwargs: Additional parameters for ensemble

    Returns:
        results: Dictionary with metrics
    """
    # Create ensemble
    ensemble = BaseTTTEnsemble(
        ensemble_method=ensemble_method,
        verbose=True,
        **ensemble_kwargs
    )

    # Get predictions
    predictions, probs, stats = ensemble.predict(base_probs, ttt_probs)

    # Convert to numpy
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Convert labels to binary if needed
    if labels_np.max() > 1:
        labels_binary = (labels_np > 0).astype(int)
    else:
        labels_binary = labels_np

    # Compute metrics
    cm = confusion_matrix(labels_binary, predictions_np)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        zdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = zdr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results = {
            'ensemble_method': ensemble_method,
            'far': far,
            'zdr': zdr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'ensemble_stats': stats
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"ENSEMBLE EVALUATION RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Method: {ensemble_method}")
        logger.info(f"\nMetrics:")
        logger.info(f"  FAR: {far*100:.2f}%")
        logger.info(f"  ZDR: {zdr*100:.2f}%")
        logger.info(f"  Accuracy: {accuracy*100:.2f}%")
        logger.info(f"  Precision: {precision*100:.2f}%")
        logger.info(f"  F1-Score: {f1*100:.2f}%")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {tn}, FP: {fp}")
        logger.info(f"  FN: {fn}, TP: {tp}")
        logger.info(f"{'='*70}\n")

        return results
    else:
        logger.error("Invalid confusion matrix shape")
        return {}


if __name__ == '__main__':
    # Test ensemble on simulated data
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Simulate base and TTT predictions
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 5000
    n_attacks = 1500

    # Base model: Conservative (low FAR ~25%, moderate recall ~80%)
    base_probs_normal = torch.tensor(np.random.beta(5, 2, size=n_samples-n_attacks)).unsqueeze(1)
    base_probs_attack = torch.tensor(np.random.beta(3, 2, size=n_attacks)).unsqueeze(1)
    base_probs = torch.cat([
        torch.cat([base_probs_normal, 1-base_probs_normal], dim=1),
        torch.cat([1-base_probs_attack, base_probs_attack], dim=1)
    ])

    # TTT model: Aggressive (high FAR ~42%, high recall ~94%)
    ttt_probs_normal = torch.tensor(np.random.beta(2, 3, size=n_samples-n_attacks)).unsqueeze(1)
    ttt_probs_attack = torch.tensor(np.random.beta(6, 1, size=n_attacks)).unsqueeze(1)
    ttt_probs = torch.cat([
        torch.cat([ttt_probs_normal, 1-ttt_probs_normal], dim=1),
        torch.cat([1-ttt_probs_attack, ttt_probs_attack], dim=1)
    ])

    labels = torch.cat([torch.zeros(n_samples-n_attacks), torch.ones(n_attacks)]).long()

    print("\n" + "="*70)
    print("TESTING BASE + TTT ENSEMBLE")
    print("="*70)

    # Test each ensemble method
    methods = ['weighted_prob', 'confidence_weighted', 'voting']

    for method in methods:
        print(f"\n\nTesting {method} ensemble...")
        results = evaluate_ensemble(
            base_probs, ttt_probs, labels,
            ensemble_method=method,
            base_weight=0.4,
            base_confidence_threshold=0.85,
            ttt_confidence_threshold=0.70
        )
