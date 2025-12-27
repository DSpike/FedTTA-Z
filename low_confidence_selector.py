"""
Low-Confidence Sample Selection for TTT Adaptation

This module implements various strategies to identify low-confidence samples
that are likely to be zero-day attacks, enabling focused TTT adaptation.

Reference: Based on analysis in LOW_CONFIDENCE_ONLY_TTT_EXPLANATION.md
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LowConfidenceSampleSelector:
    """
    Selects low-confidence samples from test data for focused TTT adaptation.

    The key insight: Zero-day samples typically have LOW confidence because
    the model hasn't seen them before, making them ideal candidates for adaptation.
    """

    def __init__(
        self,
        method: str = 'entropy',
        threshold_percentile: float = 0.7,
        min_samples: int = 50,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the low-confidence sample selector.

        Args:
            method: Selection method ('entropy', 'probability', 'distance', 'combined')
            threshold_percentile: Percentile for threshold (0.7 = top 30% most uncertain)
            min_samples: Minimum number of samples to select
            max_samples: Maximum number of samples to select (None = no limit)
        """
        self.method = method
        self.threshold_percentile = threshold_percentile
        self.min_samples = min_samples
        self.max_samples = max_samples

        logger.info(f"🎯 Low-Confidence Selector initialized:")
        logger.info(f"   Method: {method}")
        logger.info(f"   Threshold percentile: {threshold_percentile:.2f} (top {(1-threshold_percentile)*100:.0f}% most uncertain)")
        logger.info(f"   Min samples: {min_samples}, Max samples: {max_samples}")

    def select_low_confidence_samples(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Select low-confidence samples from test data.

        Args:
            model: Base model (before adaptation)
            X_test: Test samples (N, seq_len, features)
            y_test: Test labels (optional, only for statistics)
            prototypes: Class prototypes (optional, for distance-based selection)

        Returns:
            selected_samples: Low-confidence samples
            selected_mask: Boolean mask for selected samples
            stats: Statistics about selection
        """
        model.eval()
        device = next(model.parameters()).device
        X_test = X_test.to(device)

        with torch.no_grad():
            if self.method == 'entropy':
                selected_samples, selected_mask, stats = self._select_by_entropy(
                    model, X_test, y_test
                )
            elif self.method == 'probability':
                selected_samples, selected_mask, stats = self._select_by_probability(
                    model, X_test, y_test
                )
            elif self.method == 'distance':
                selected_samples, selected_mask, stats = self._select_by_distance(
                    model, X_test, y_test, prototypes
                )
            elif self.method == 'combined':
                selected_samples, selected_mask, stats = self._select_by_combined(
                    model, X_test, y_test, prototypes
                )
            else:
                raise ValueError(f"Unknown selection method: {self.method}")

        return selected_samples, selected_mask, stats

    def _select_by_entropy(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Select samples with HIGH entropy (uncertain predictions).

        High entropy = model is uncertain = likely zero-day
        """
        # Get model predictions
        outputs = model(X_test)
        probs = F.softmax(outputs, dim=1)

        # Compute entropy: H = -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # High entropy = uncertain = likely zero-day
        threshold = torch.quantile(entropy, self.threshold_percentile)
        low_confidence_mask = entropy > threshold

        # Apply min/max constraints
        low_confidence_mask = self._apply_sample_constraints(
            low_confidence_mask, entropy, descending=True
        )

        selected_samples = X_test[low_confidence_mask]

        # Compute statistics
        stats = self._compute_selection_stats(
            low_confidence_mask,
            y_test,
            confidence_scores=entropy,
            metric_name='entropy',
            threshold=threshold
        )

        return selected_samples, low_confidence_mask, stats

    def _select_by_probability(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Select samples with LOW max probability (uncertain predictions).

        Low max probability = model is uncertain = likely zero-day
        """
        # Get model predictions
        outputs = model(X_test)
        probs = F.softmax(outputs, dim=1)

        # Get maximum probability (confidence)
        max_probs, _ = probs.max(dim=1)

        # Low max probability = uncertain = likely zero-day
        threshold = torch.quantile(max_probs, 1 - self.threshold_percentile)
        low_confidence_mask = max_probs < threshold

        # Apply min/max constraints
        low_confidence_mask = self._apply_sample_constraints(
            low_confidence_mask, max_probs, descending=False
        )

        selected_samples = X_test[low_confidence_mask]

        # Compute statistics
        stats = self._compute_selection_stats(
            low_confidence_mask,
            y_test,
            confidence_scores=max_probs,
            metric_name='max_probability',
            threshold=threshold
        )

        return selected_samples, low_confidence_mask, stats

    def _select_by_distance(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Select samples FAR from prototypes (uncertain predictions).

        Far from prototypes = model is uncertain = likely zero-day
        """
        # Get embeddings
        if hasattr(model, 'forward_with_prototypes'):
            # Prototype-based model: get embeddings
            if prototypes is None:
                logger.warning("⚠️ Distance-based selection requires prototypes, falling back to entropy")
                return self._select_by_entropy(model, X_test, y_test)

            embeddings = model(X_test)  # Get embeddings

            # Compute minimum distance to any prototype
            # Distance: (batch, prototypes) -> min over prototypes
            distances = torch.cdist(embeddings, prototypes, p=2)  # L2 distance
            min_distances, _ = distances.min(dim=1)

            # Far from prototypes = uncertain = likely zero-day
            threshold = torch.quantile(min_distances, self.threshold_percentile)
            low_confidence_mask = min_distances > threshold

            # Apply min/max constraints
            low_confidence_mask = self._apply_sample_constraints(
                low_confidence_mask, min_distances, descending=True
            )

            selected_samples = X_test[low_confidence_mask]

            # Compute statistics
            stats = self._compute_selection_stats(
                low_confidence_mask,
                y_test,
                confidence_scores=min_distances,
                metric_name='min_distance_to_prototype',
                threshold=threshold
            )

            return selected_samples, low_confidence_mask, stats
        else:
            logger.warning("⚠️ Model doesn't support distance-based selection, falling back to entropy")
            return self._select_by_entropy(model, X_test, y_test)

    def _select_by_combined(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Combine multiple selection methods for robust selection.

        Combines entropy, probability, and distance (if available).
        """
        # Get all confidence scores
        outputs = model(X_test)
        probs = F.softmax(outputs, dim=1)

        # 1. Entropy score (normalized)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

        # 2. Probability score (normalized, inverted so low prob = high score)
        max_probs, _ = probs.max(dim=1)
        prob_norm = 1 - ((max_probs - max_probs.min()) / (max_probs.max() - max_probs.min() + 1e-8))

        # 3. Distance score (if available)
        if prototypes is not None and hasattr(model, 'forward_with_prototypes'):
            embeddings = model(X_test)
            distances = torch.cdist(embeddings, prototypes, p=2)
            min_distances, _ = distances.min(dim=1)
            dist_norm = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min() + 1e-8)

            # Combined score: average of all three
            combined_score = (entropy_norm + prob_norm + dist_norm) / 3
        else:
            # Combined score: average of entropy and probability
            combined_score = (entropy_norm + prob_norm) / 2

        # Select samples with high combined uncertainty score
        threshold = torch.quantile(combined_score, self.threshold_percentile)
        low_confidence_mask = combined_score > threshold

        # Apply min/max constraints
        low_confidence_mask = self._apply_sample_constraints(
            low_confidence_mask, combined_score, descending=True
        )

        selected_samples = X_test[low_confidence_mask]

        # Compute statistics
        stats = self._compute_selection_stats(
            low_confidence_mask,
            y_test,
            confidence_scores=combined_score,
            metric_name='combined_uncertainty',
            threshold=threshold
        )

        return selected_samples, low_confidence_mask, stats

    def _apply_sample_constraints(
        self,
        mask: torch.Tensor,
        scores: torch.Tensor,
        descending: bool = True
    ) -> torch.Tensor:
        """
        Apply min/max sample constraints to selection mask.

        Args:
            mask: Initial selection mask
            scores: Confidence scores
            descending: If True, select highest scores; if False, select lowest

        Returns:
            Constrained mask
        """
        n_selected = mask.sum().item()

        # Apply minimum constraint
        if n_selected < self.min_samples:
            # Need more samples: select top-k by score
            k = min(self.min_samples, len(scores))
            _, top_indices = torch.topk(scores, k, largest=descending)
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[top_indices] = True
            n_selected = k

        # Apply maximum constraint
        if self.max_samples is not None and n_selected > self.max_samples:
            # Too many samples: select top-k by score
            selected_indices = mask.nonzero(as_tuple=True)[0]
            selected_scores = scores[selected_indices]
            _, top_k_in_selected = torch.topk(
                selected_scores, self.max_samples, largest=descending
            )

            # Create new mask with only top-k
            new_mask = torch.zeros_like(mask, dtype=torch.bool)
            new_mask[selected_indices[top_k_in_selected]] = True
            mask = new_mask

        return mask

    def _compute_selection_stats(
        self,
        mask: torch.Tensor,
        y_test: Optional[torch.Tensor],
        confidence_scores: torch.Tensor,
        metric_name: str,
        threshold: torch.Tensor
    ) -> Dict:
        """
        Compute statistics about sample selection.

        Returns:
            stats: Dictionary with selection statistics
        """
        n_selected = mask.sum().item()
        n_total = len(mask)
        selection_rate = n_selected / n_total if n_total > 0 else 0.0

        stats = {
            'n_selected': n_selected,
            'n_total': n_total,
            'selection_rate': selection_rate,
            'method': self.method,
            'metric_name': metric_name,
            'threshold': threshold.item(),
            'mean_score_selected': confidence_scores[mask].mean().item() if n_selected > 0 else 0.0,
            'mean_score_all': confidence_scores.mean().item(),
        }

        # If labels are provided, compute zero-day correlation
        if y_test is not None:
            y_test = y_test.cpu()
            mask_cpu = mask.cpu()

            # Identify zero-day samples (assuming label 14 or other specific label)
            # For general case, we'll check if labels are available
            stats['has_labels'] = True

            # Count zero-day in selected vs total
            # Note: This is just for analysis, NOT used for selection (unsupervised)
            selected_labels = y_test[mask_cpu]

            stats['selected_label_distribution'] = {
                label.item(): (selected_labels == label).sum().item()
                for label in torch.unique(selected_labels)
            }

        else:
            stats['has_labels'] = False

        # Log statistics
        logger.info(f"📊 Low-Confidence Selection Statistics ({self.method}):")
        logger.info(f"   Selected: {n_selected}/{n_total} samples ({selection_rate*100:.1f}%)")
        logger.info(f"   {metric_name} threshold: {threshold.item():.4f}")
        logger.info(f"   Mean {metric_name} (selected): {stats['mean_score_selected']:.4f}")
        logger.info(f"   Mean {metric_name} (all): {stats['mean_score_all']:.4f}")

        return stats


def select_low_confidence_samples_simple(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: Optional[torch.Tensor] = None,
    method: str = 'entropy',
    percentile: float = 0.7,
    min_samples: int = 50,
    max_samples: Optional[int] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Simple function interface for low-confidence sample selection.

    Args:
        model: Base model (before adaptation)
        X_test: Test samples
        y_test: Test labels (optional, for statistics only)
        method: Selection method ('entropy', 'probability', 'distance', 'combined')
        percentile: Percentile threshold (0.7 = top 30% most uncertain)
        min_samples: Minimum samples to select
        max_samples: Maximum samples to select

    Returns:
        selected_samples: Low-confidence samples for TTT adaptation
        stats: Selection statistics
    """
    selector = LowConfidenceSampleSelector(
        method=method,
        threshold_percentile=percentile,
        min_samples=min_samples,
        max_samples=max_samples
    )

    selected_samples, selected_mask, stats = selector.select_low_confidence_samples(
        model, X_test, y_test
    )

    return selected_samples, stats
