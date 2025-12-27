"""
Ensemble Predictor: Combine Base + TTT Models to Reduce FAR while Maintaining ZDR

Strategy: Confidence-weighted ensemble
- Use base model when it's CONFIDENT about "normal" (high precision on normals)
- Use TTT model for everything else, especially attacks (high recall on attacks)

This combines:
- Base model's conservative nature (FAR 23.20%)
- TTT model's aggressive attack detection (ZDR 93.65%)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def confidence_weighted_ensemble(
    base_probs: torch.Tensor,
    ttt_probs: torch.Tensor,
    base_confidence_threshold: float = 0.90,
    ttt_confidence_threshold: float = 0.70,
    attack_class_idx: int = 1,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict]:
    """
    Confidence-weighted ensemble prediction

    Logic:
    1. If base model is VERY confident it's normal (conf > 0.90, pred = normal)
       → Trust base model (reduces false positives)
    2. If TTT model is confident it's attack (conf > 0.70, pred = attack)
       → Trust TTT model (maintains high recall)
    3. Otherwise, use weighted average favoring TTT (better at edge cases)

    Args:
        base_probs: Base model probabilities (N, num_classes)
        ttt_probs: TTT model probabilities (N, num_classes)
        base_confidence_threshold: Min confidence for trusting base "normal" predictions
        ttt_confidence_threshold: Min confidence for trusting TTT "attack" predictions
        attack_class_idx: Index of attack class (1 for binary)
        verbose: Log statistics

    Returns:
        predictions: Ensemble predictions (N,)
        stats: Dictionary with ensemble statistics
    """
    device = base_probs.device
    n_samples = base_probs.shape[0]

    # Get predictions and confidences
    base_conf, base_pred = base_probs.max(dim=1)
    ttt_conf, ttt_pred = ttt_probs.max(dim=1)

    # Strategy: Confidence-based hybrid
    # Zone 1: Base model confident normal (high precision filter)
    base_confident_normal = (base_conf >= base_confidence_threshold) & (base_pred == 0)

    # Zone 2: TTT model confident attack (high recall detector)
    ttt_confident_attack = (ttt_conf >= ttt_confidence_threshold) & (ttt_pred == attack_class_idx)

    # Zone 3: Uncertain region → use weighted ensemble favoring TTT
    uncertain_region = ~(base_confident_normal | ttt_confident_attack)

    # Initialize predictions
    ensemble_pred = torch.zeros(n_samples, dtype=torch.long, device=device)

    # Zone 1: Trust base model (reduce FP)
    ensemble_pred[base_confident_normal] = 0

    # Zone 2: Trust TTT model (maintain recall)
    ensemble_pred[ttt_confident_attack] = attack_class_idx

    # Zone 3: Weighted average (60% TTT, 40% base - TTT better at edge cases)
    if uncertain_region.sum() > 0:
        ensemble_probs_uncertain = 0.6 * ttt_probs[uncertain_region] + 0.4 * base_probs[uncertain_region]
        ensemble_pred[uncertain_region] = ensemble_probs_uncertain.argmax(dim=1)

    # Compute ensemble probabilities for metrics
    ensemble_probs = base_probs.clone()
    ensemble_probs[base_confident_normal] = base_probs[base_confident_normal]
    ensemble_probs[ttt_confident_attack] = ttt_probs[ttt_confident_attack]
    if uncertain_region.sum() > 0:
        ensemble_probs[uncertain_region] = 0.6 * ttt_probs[uncertain_region] + 0.4 * base_probs[uncertain_region]

    # Statistics
    stats = {
        'total_samples': n_samples,
        'base_confident_normal': base_confident_normal.sum().item(),
        'ttt_confident_attack': ttt_confident_attack.sum().item(),
        'uncertain_region': uncertain_region.sum().item(),
        'base_conf_threshold': base_confidence_threshold,
        'ttt_conf_threshold': ttt_confidence_threshold,
        'zone_percentages': {
            'base_zone': 100.0 * base_confident_normal.sum().item() / n_samples,
            'ttt_zone': 100.0 * ttt_confident_attack.sum().item() / n_samples,
            'uncertain_zone': 100.0 * uncertain_region.sum().item() / n_samples
        }
    }

    if verbose:
        logger.info("🎯 Confidence-Weighted Ensemble Statistics:")
        logger.info(f"   Zone 1 (Base confident normal): {stats['base_confident_normal']} ({stats['zone_percentages']['base_zone']:.1f}%)")
        logger.info(f"   Zone 2 (TTT confident attack):  {stats['ttt_confident_attack']} ({stats['zone_percentages']['ttt_zone']:.1f}%)")
        logger.info(f"   Zone 3 (Uncertain - weighted):  {stats['uncertain_region']} ({stats['zone_percentages']['uncertain_zone']:.1f}%)")

    return ensemble_pred, ensemble_probs, stats


def weighted_probability_ensemble(
    base_probs: torch.Tensor,
    ttt_probs: torch.Tensor,
    alpha: float = 0.4,
    threshold: float = 0.5,
    attack_class_idx: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple weighted probability ensemble

    ensemble_prob = alpha * base_prob + (1-alpha) * ttt_prob

    Args:
        base_probs: Base model probabilities (N, num_classes)
        ttt_probs: TTT model probabilities (N, num_classes)
        alpha: Weight for base model (1-alpha for TTT)
        threshold: Decision threshold
        attack_class_idx: Index of attack class

    Returns:
        predictions: Ensemble predictions (N,)
        probs: Ensemble probabilities (N, num_classes)
    """
    ensemble_probs = alpha * base_probs + (1.0 - alpha) * ttt_probs

    # For binary, use attack probability
    if ensemble_probs.shape[1] == 2:
        attack_probs = ensemble_probs[:, attack_class_idx]
        predictions = (attack_probs >= threshold).long()
    else:
        predictions = ensemble_probs.argmax(dim=1)

    return predictions, ensemble_probs


def find_optimal_ensemble_params(
    base_probs: torch.Tensor,
    ttt_probs: torch.Tensor,
    labels: torch.Tensor,
    target_far: float = 0.10,
    target_zdr: float = 0.90,
    method: str = 'confidence_weighted',
    verbose: bool = True
) -> Dict:
    """
    Find optimal ensemble parameters via grid search

    Args:
        base_probs: Base model probabilities (N, num_classes)
        ttt_probs: TTT model probabilities (N, num_classes)
        labels: Ground truth labels (N,)
        target_far: Target FAR constraint
        target_zdr: Target ZDR constraint
        method: Ensemble method ('confidence_weighted', 'weighted_prob')
        verbose: Print progress

    Returns:
        best_params: Optimal parameters and metrics
    """
    logger.info(f"🔍 Searching optimal {method} ensemble parameters...")
    logger.info(f"   Target: FAR < {target_far*100:.1f}%, ZDR > {target_zdr*100:.1f}%")

    # Convert labels to binary if needed
    if labels.max() > 1:
        labels_binary = (labels > 0).long()
    else:
        labels_binary = labels

    labels_np = labels_binary.cpu().numpy()

    best_params = None
    best_score = float('inf')  # Lower is better
    results = []

    if method == 'confidence_weighted':
        # Grid search over confidence thresholds
        base_thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97]
        ttt_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]

        for base_thresh in base_thresholds:
            for ttt_thresh in ttt_thresholds:
                # Get ensemble predictions
                pred, probs, stats = confidence_weighted_ensemble(
                    base_probs, ttt_probs,
                    base_confidence_threshold=base_thresh,
                    ttt_confidence_threshold=ttt_thresh,
                    verbose=False
                )

                pred_np = pred.cpu().numpy()

                # Calculate metrics
                cm = confusion_matrix(labels_np, pred_np)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    zdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = zdr
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    continue

                # Score: minimize FAR while maintaining ZDR
                if zdr >= target_zdr:
                    score = far  # Lower FAR is better
                else:
                    score = far + 10.0 * (target_zdr - zdr)  # Penalty for low ZDR

                results.append({
                    'base_thresh': base_thresh,
                    'ttt_thresh': ttt_thresh,
                    'far': far,
                    'zdr': zdr,
                    'accuracy': accuracy,
                    'precision': precision,
                    'f1': f1,
                    'score': score
                })

                if score < best_score:
                    best_score = score
                    best_params = {
                        'method': 'confidence_weighted',
                        'base_confidence_threshold': base_thresh,
                        'ttt_confidence_threshold': ttt_thresh,
                        'far': far,
                        'zdr': zdr,
                        'accuracy': accuracy,
                        'precision': precision,
                        'f1': f1,
                        'zone_stats': stats
                    }

    elif method == 'weighted_prob':
        # Grid search over alpha (base model weight)
        alphas = np.arange(0.0, 1.01, 0.1)
        thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]

        for alpha in alphas:
            for thresh in thresholds:
                pred, probs = weighted_probability_ensemble(
                    base_probs, ttt_probs,
                    alpha=alpha,
                    threshold=thresh
                )

                pred_np = pred.cpu().numpy()

                cm = confusion_matrix(labels_np, pred_np)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    zdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = zdr
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    continue

                if zdr >= target_zdr:
                    score = far
                else:
                    score = far + 10.0 * (target_zdr - zdr)

                results.append({
                    'alpha': alpha,
                    'threshold': thresh,
                    'far': far,
                    'zdr': zdr,
                    'accuracy': accuracy,
                    'precision': precision,
                    'f1': f1,
                    'score': score
                })

                if score < best_score:
                    best_score = score
                    best_params = {
                        'method': 'weighted_prob',
                        'alpha': alpha,
                        'threshold': thresh,
                        'far': far,
                        'zdr': zdr,
                        'accuracy': accuracy,
                        'precision': precision,
                        'f1': f1
                    }

    if verbose and best_params:
        logger.info(f"\n✅ Optimal {method} ensemble found:")
        if method == 'confidence_weighted':
            logger.info(f"   Base conf threshold: {best_params['base_confidence_threshold']:.2f}")
            logger.info(f"   TTT conf threshold: {best_params['ttt_confidence_threshold']:.2f}")
        elif method == 'weighted_prob':
            logger.info(f"   Alpha (base weight): {best_params['alpha']:.2f}")
            logger.info(f"   Decision threshold: {best_params['threshold']:.2f}")

        logger.info(f"\n   FAR: {best_params['far']*100:.2f}% (target: <{target_far*100:.1f}%)")
        logger.info(f"   ZDR: {best_params['zdr']*100:.2f}% (target: >{target_zdr*100:.1f}%)")
        logger.info(f"   Accuracy: {best_params['accuracy']*100:.2f}%")
        logger.info(f"   Precision: {best_params['precision']*100:.2f}%")
        logger.info(f"   F1-Score: {best_params['f1']*100:.2f}%")

        # Check if target met
        target_met = best_params['far'] <= target_far and best_params['zdr'] >= target_zdr
        if target_met:
            logger.info(f"\n✅ TARGET MET: FAR < {target_far*100:.1f}% AND ZDR > {target_zdr*100:.1f}%")
        else:
            if best_params['far'] > target_far:
                logger.warning(f"\n⚠️  FAR still high: {best_params['far']*100:.2f}% (target: <{target_far*100:.1f}%)")
            if best_params['zdr'] < target_zdr:
                logger.warning(f"\n⚠️  ZDR too low: {best_params['zdr']*100:.2f}% (target: >{target_zdr*100:.1f}%)")

    best_params['results'] = results
    return best_params


if __name__ == '__main__':
    # Test ensemble on simulated data
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Simulate base and TTT predictions
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 5000
    n_attacks = 1500

    # Base model: Conservative (low FAR, moderate recall)
    base_probs_normal = torch.tensor(np.random.beta(5, 2, size=n_samples-n_attacks)).unsqueeze(1)
    base_probs_attack = torch.tensor(np.random.beta(3, 3, size=n_attacks)).unsqueeze(1)
    base_probs = torch.cat([
        torch.cat([base_probs_normal, 1-base_probs_normal], dim=1),
        torch.cat([1-base_probs_attack, base_probs_attack], dim=1)
    ])

    # TTT model: Aggressive (high FAR, high recall)
    ttt_probs_normal = torch.tensor(np.random.beta(2, 5, size=n_samples-n_attacks)).unsqueeze(1)
    ttt_probs_attack = torch.tensor(np.random.beta(6, 1.5, size=n_attacks)).unsqueeze(1)
    ttt_probs = torch.cat([
        torch.cat([ttt_probs_normal, 1-ttt_probs_normal], dim=1),
        torch.cat([1-ttt_probs_attack, ttt_probs_attack], dim=1)
    ])

    labels = torch.cat([torch.zeros(n_samples-n_attacks), torch.ones(n_attacks)]).long()

    print("Testing Confidence-Weighted Ensemble:")
    params = find_optimal_ensemble_params(
        base_probs, ttt_probs, labels,
        target_far=0.10,
        target_zdr=0.90,
        method='confidence_weighted',
        verbose=True
    )
