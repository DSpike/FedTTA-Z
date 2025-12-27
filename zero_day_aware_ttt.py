"""
Zero-Day Aware Test-Time Training
==================================

Enhanced TTT that focuses adaptation on likely zero-day samples (low-confidence)
instead of generic entropy minimization on all samples.

Key Insight: Low-confidence predictions are likely zero-day attacks!

Author: PhD Research
Date: 2025-12-17
"""

import torch
import torch.nn as nn
import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_zero_day_aware_ttt(
    model: nn.Module,
    X_test: torch.Tensor,
    config,
    device='cuda'
) -> nn.Module:
    """
    Apply zero-day aware TTT adaptation.

    Unlike generic entropy minimization, this:
    1. Identifies likely zero-day samples (low confidence)
    2. Weights their contribution higher in the loss
    3. Focuses adaptation on what matters: unseen attacks

    Args:
        model: Trained binary model to adapt
        X_test: Test data (unlabeled)
        config: System configuration
        device: Device to run on

    Returns:
        Adapted model
    """
    logger.info("=" * 80)
    logger.info("🔬 ZERO-DAY AWARE TTT ADAPTATION")
    logger.info("=" * 80)

    model = model.to(device)
    X_test = X_test.to(device)
    model.train()  # Enable BatchNorm adaptation

    # Step 1: Identify likely zero-day samples
    with torch.no_grad():
        logits = model(X_test)
        if logits.shape[-1] > 2:
            logits = logits[:, :2]
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]  # Max probability

    # Low confidence = likely zero-day
    low_confidence_threshold = 0.6  # Configurable
    zero_day_candidates = confidence < low_confidence_threshold

    n_zero_day_candidates = zero_day_candidates.sum().item()
    logger.info(f"📊 Identified {n_zero_day_candidates}/{len(X_test)} likely zero-day samples")
    logger.info(f"   (confidence < {low_confidence_threshold})")

    if n_zero_day_candidates == 0:
        logger.warning("⚠️ No zero-day candidates found! Using all samples.")
        zero_day_weights = torch.ones(len(X_test), device=device)
    else:
        # Compute weights: inversely proportional to confidence
        # Low confidence → High weight
        zero_day_weights = 1.0 / (confidence + 0.1)

        # Normalize weights
        zero_day_weights = zero_day_weights / zero_day_weights.sum()

        # Report weight distribution
        logger.info(f"   Weight stats:")
        logger.info(f"     Min: {zero_day_weights.min().item():.6f}")
        logger.info(f"     Max: {zero_day_weights.max().item():.6f}")
        logger.info(f"     Mean: {zero_day_weights.mean().item():.6f}")

    # Step 2: Sample support set (focus on zero-day candidates)
    support_ratio = 0.3
    support_size = int(len(X_test) * support_ratio)
    support_size = max(support_size, 100)
    support_size = min(support_size, len(X_test) // 2)

    # Weighted sampling: higher probability for low-confidence samples
    support_indices = torch.multinomial(
        zero_day_weights,
        support_size,
        replacement=False
    )

    X_support = X_test[support_indices]
    support_weights = zero_day_weights[support_indices]

    logger.info(f"📊 Support set:")
    logger.info(f"   Size: {len(X_support)}")
    logger.info(f"   Sampling: Weighted by confidence (favors zero-day)")

    # Step 3: Setup optimizer (only BatchNorm + Classifier)
    params_to_adapt = []
    for name, param in model.named_parameters():
        if 'bn' in name.lower() or 'batchnorm' in name.lower() or 'classifier' in name.lower():
            param.requires_grad = True
            params_to_adapt.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(params_to_adapt, lr=config.ttt_lr)

    logger.info(f"   Optimizing {len(params_to_adapt)} parameter groups")
    logger.info(f"   Learning rate: {config.ttt_lr}")
    logger.info(f"   Steps: {config.ttt_base_steps}")

    # Step 4: TTT adaptation loop with zero-day weighting
    adaptation_losses = []
    entropy_losses = []
    l2_losses = []

    # Store base model parameters for L2 regularization
    base_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            base_params[name] = param.data.clone()

    for step in range(config.ttt_base_steps):
        optimizer.zero_grad()

        # Forward pass
        logits = model(X_support)
        if logits.shape[-1] > 2:
            logits = logits[:, :2]
        probs = torch.softmax(logits, dim=1)

        # Weighted entropy loss (focus on zero-day)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        weighted_entropy = (entropy * support_weights).sum()  # Already normalized

        # L2 regularization (stay close to base model)
        l2_loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in base_params:
                l2_loss += ((param - base_params[name]) ** 2).sum()

        # Total loss
        total_loss = (
            config.entropy_weight * weighted_entropy +
            config.ttt_l2_reg_weight * l2_loss
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Track losses
        adaptation_losses.append(total_loss.item())
        entropy_losses.append(weighted_entropy.item())
        l2_losses.append(l2_loss.item())

        if (step + 1) % 20 == 0:
            logger.info(
                f"  Step {step+1}/{config.ttt_base_steps}: "
                f"Loss={total_loss.item():.4f}, "
                f"Entropy={weighted_entropy.item():.4f}, "
                f"L2={l2_loss.item():.4f}"
            )

    # Set to evaluation mode
    model.eval()

    logger.info(f"\n✅ Zero-Day Aware TTT Completed")
    logger.info(f"   Final loss: {adaptation_losses[-1]:.4f}")
    logger.info(f"   Avg entropy: {np.mean(entropy_losses):.4f}")
    logger.info(f"   Avg L2: {np.mean(l2_losses):.4f}")
    logger.info("=" * 80)

    return model


def apply_contrastive_ttt(
    model: nn.Module,
    X_test: torch.Tensor,
    config,
    device='cuda'
) -> nn.Module:
    """
    Apply contrastive TTT adaptation.

    Uses contrastive learning to separate normal and attack embeddings.
    Assumes low-confidence samples are zero-day attacks.

    Args:
        model: Trained binary model to adapt
        X_test: Test data (unlabeled)
        config: System configuration
        device: Device to run on

    Returns:
        Adapted model
    """
    logger.info("=" * 80)
    logger.info("🔬 CONTRASTIVE TTT ADAPTATION")
    logger.info("=" * 80)

    model = model.to(device)
    X_test = X_test.to(device)
    model.train()

    # Step 1: Get initial pseudo-labels
    with torch.no_grad():
        logits = model(X_test)
        if logits.shape[-1] > 2:
            logits = logits[:, :2]
        probs = torch.softmax(logits, dim=1)
        pseudo_labels = probs.argmax(dim=1)
        confidence = probs.max(dim=1)[0]

    # Step 2: Select high-confidence samples for anchors
    # Low-confidence samples might be zero-day
    high_conf_mask = confidence > 0.8

    logger.info(f"📊 High-confidence samples: {high_conf_mask.sum().item()}/{len(X_test)}")

    if high_conf_mask.sum() < 10:
        logger.warning("⚠️ Too few high-confidence samples for contrastive learning")
        logger.warning("   Falling back to zero-day aware TTT")
        return apply_zero_day_aware_ttt(model, X_test, config, device)

    # Step 3: Sample support set
    support_size = int(len(X_test) * 0.3)
    support_size = max(support_size, 100)
    support_size = min(support_size, len(X_test) // 2)

    support_indices = torch.randperm(len(X_test))[:support_size]
    X_support = X_test[support_indices]
    support_pseudo_labels = pseudo_labels[support_indices]

    # Step 4: Setup optimizer
    params_to_adapt = []
    for name, param in model.named_parameters():
        if 'bn' in name.lower() or 'batchnorm' in name.lower() or 'classifier' in name.lower():
            param.requires_grad = True
            params_to_adapt.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(params_to_adapt, lr=config.ttt_lr)

    logger.info(f"   Support size: {len(X_support)}")
    logger.info(f"   Optimizing {len(params_to_adapt)} parameter groups")

    # Step 5: Contrastive adaptation
    adaptation_losses = []

    # Store base model parameters
    base_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            base_params[name] = param.data.clone()

    for step in range(config.ttt_base_steps):
        optimizer.zero_grad()

        # Forward pass
        logits = model(X_support)
        if logits.shape[-1] > 2:
            logits = logits[:, :2]

        # Extract embeddings (features before final layer)
        if hasattr(model, 'extract_features'):
            embeddings = model.extract_features(X_support)
        else:
            # Fallback: use logits as features
            embeddings = logits

        # Contrastive loss: pull same class together, push different apart
        normal_mask = support_pseudo_labels == 0
        attack_mask = support_pseudo_labels == 1

        if normal_mask.sum() > 0 and attack_mask.sum() > 0:
            normal_center = embeddings[normal_mask].mean(dim=0)
            attack_center = embeddings[attack_mask].mean(dim=0)

            # Push centers apart
            center_dist = torch.cdist(normal_center.unsqueeze(0), attack_center.unsqueeze(0))
            separation_loss = -center_dist.mean()  # Negative to maximize distance

            # Pull samples to their centers
            normal_dist = torch.cdist(embeddings[normal_mask], normal_center.unsqueeze(0)).mean()
            attack_dist = torch.cdist(embeddings[attack_mask], attack_center.unsqueeze(0)).mean()
            compactness_loss = normal_dist + attack_dist

            contrastive_loss = separation_loss + compactness_loss
        else:
            # Fallback to entropy if one class is missing
            probs = torch.softmax(logits, dim=1)
            contrastive_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

        # L2 regularization
        l2_loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in base_params:
                l2_loss += ((param - base_params[name]) ** 2).sum()

        # Total loss
        total_loss = contrastive_loss + config.ttt_l2_reg_weight * l2_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        adaptation_losses.append(total_loss.item())

        if (step + 1) % 20 == 0:
            logger.info(f"  Step {step+1}/{config.ttt_base_steps}: Loss={total_loss.item():.4f}")

    model.eval()

    logger.info(f"\n✅ Contrastive TTT Completed")
    logger.info(f"   Final loss: {adaptation_losses[-1]:.4f}")
    logger.info("=" * 80)

    return model
