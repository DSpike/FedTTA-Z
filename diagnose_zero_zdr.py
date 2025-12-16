#!/usr/bin/env python3
"""
Diagnostic script to investigate why ZDR is zero
"""

import torch
import numpy as np
import logging
from config import SystemConfig
from main import BlockchainFederatedIncentiveSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_zero_zdr():
    """Diagnose why zero-day detection rate is zero"""
    
    logger.info("🔍 Diagnosing Zero-Day Detection Rate (ZDR) issue...")
    
    # Load config
    config = SystemConfig()
    
    # Initialize system
    system = BlockchainFederatedIncentiveSystem(config)
    
    # Check if preprocessed data exists
    if not hasattr(system, 'preprocessed_data'):
        logger.error("❌ System not initialized. Run preprocessing first.")
        return
    
    # Get test data
    X_test = system.preprocessed_data.get('X_test')
    y_test = system.preprocessed_data.get('y_test')
    y_test_multiclass = system.preprocessed_data.get('y_test_multiclass_seq')
    
    if X_test is None or y_test is None:
        logger.error("❌ Test data not found!")
        return
    
    logger.info(f"✅ Test data found: {len(X_test)} samples")
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(system.device)
    y_test_tensor = torch.LongTensor(y_test).to(system.device)
    
    # Check zero-day mask
    zero_day_attack_label = config.zero_day_attack_label
    logger.info(f"🔍 Zero-day attack: '{config.zero_day_attack}', label: {zero_day_attack_label}")
    
    if y_test_multiclass is not None:
        if isinstance(y_test_multiclass, np.ndarray):
            y_test_multiclass = torch.from_numpy(y_test_multiclass).to(system.device)
        zero_day_mask = (y_test_multiclass == zero_day_attack_label)
        num_zero_day = zero_day_mask.sum().item()
        logger.info(f"🔍 Zero-day samples found: {num_zero_day}/{len(y_test_tensor)} ({100*num_zero_day/len(y_test_tensor):.1f}%)")
        
        if num_zero_day == 0:
            logger.error("❌ PROBLEM: No zero-day samples found in test data!")
            logger.info(f"   Available multiclass labels: {torch.unique(y_test_multiclass).tolist()}")
            logger.info(f"   Looking for label: {zero_day_attack_label}")
            return
        
        # Check what the actual labels are for zero-day samples
        zero_day_actual_labels = y_test_multiclass[zero_day_mask]
        logger.info(f"🔍 Zero-day actual labels (multiclass): {torch.unique(zero_day_actual_labels).tolist()}")
        
        # Check binary labels for zero-day samples
        zero_day_binary_labels = y_test_tensor[zero_day_mask]
        logger.info(f"🔍 Zero-day binary labels: {torch.bincount(zero_day_binary_labels.long()).tolist()}")
        
    else:
        logger.warning("⚠️ No multiclass labels available for zero-day identification!")
        return
    
    # Check if coordinator model exists
    if not hasattr(system, 'coordinator') or system.coordinator is None:
        logger.error("❌ Coordinator not initialized. Run federated learning first.")
        return
    
    global_model = system.coordinator.model
    if global_model is None:
        logger.error("❌ Global model not found!")
        return
    
    logger.info("✅ Global model found")
    
    # Create support set from validation data
    X_val_tensor = torch.FloatTensor(system.preprocessed_data['X_val']).to(system.device)
    y_val_tensor = torch.LongTensor(system.preprocessed_data['y_val']).to(system.device)
    y_val_binary = (y_val_tensor != 0).long()
    
    support_size = min(200, len(X_val_tensor))
    support_indices = torch.randperm(len(X_val_tensor))[:support_size]
    support_x = X_val_tensor[support_indices]
    support_y = y_val_binary[support_indices]
    
    # Ensure both classes are present
    unique_support_labels = torch.unique(support_y)
    if len(unique_support_labels) < 2:
        logger.warning("⚠️ Support set only has one class! Fixing...")
        normal_indices = torch.where(y_val_binary == 0)[0]
        attack_indices = torch.where(y_val_binary == 1)[0]
        if len(normal_indices) > 0 and len(attack_indices) > 0:
            min_per_class = min(50, len(normal_indices), len(attack_indices))
            normal_sample = normal_indices[torch.randperm(len(normal_indices))[:min_per_class]]
            attack_sample = attack_indices[torch.randperm(len(attack_indices))[:min_per_class]]
            support_indices = torch.cat([normal_sample, attack_sample])
            support_x = X_val_tensor[support_indices]
            support_y = y_val_binary[support_indices]
    
    logger.info(f"✅ Support set: {len(support_x)} samples, labels: {torch.bincount(support_y, minlength=2).tolist()}")
    
    # Compute prototypes
    global_model.eval()
    with torch.no_grad():
        prototypes, unique_labels = global_model.compute_prototypes(support_x, support_y)
        logger.info(f"✅ Prototypes computed: {len(prototypes)} prototypes for labels {unique_labels.tolist()}")
        
        # Get predictions for zero-day samples
        zero_day_x = X_test_tensor[zero_day_mask]
        zero_day_logits = global_model.forward_with_prototypes(zero_day_x, prototypes)
        zero_day_predictions_indices = torch.argmax(zero_day_logits, dim=1)
        zero_day_predictions = unique_labels[zero_day_predictions_indices]
        
        logger.info(f"🔍 Zero-day predictions (mapped to labels): {torch.bincount(zero_day_predictions.long(), minlength=max(unique_labels.max().item()+1, 2)).tolist()}")
        logger.info(f"🔍 Zero-day predictions (indices into prototypes): {torch.bincount(zero_day_predictions_indices, minlength=len(unique_labels)).tolist()}")
        
        # Calculate ZDR
        zero_day_detection_rate = (zero_day_predictions != 0).float().mean().item()
        logger.info(f"🔍 Zero-Day Detection Rate: {zero_day_detection_rate:.4f} ({100*zero_day_detection_rate:.1f}%)")
        
        if zero_day_detection_rate == 0.0:
            logger.error("❌ PROBLEM: ZDR is zero! All zero-day samples predicted as Normal (0)")
            logger.info(f"   Zero-day predictions breakdown: {torch.bincount(zero_day_predictions.long(), minlength=2).tolist()}")
            logger.info(f"   Expected: All should be predicted as Attack (1)")
            
            # Check probabilities
            zero_day_probs = torch.softmax(zero_day_logits, dim=1)
            logger.info(f"   Zero-day probabilities (mean): {zero_day_probs.mean(dim=0).cpu().tolist()}")
            logger.info(f"   Zero-day probabilities (std): {zero_day_probs.std(dim=0).cpu().tolist()}")
            
            # Check which prototype is closer
            zero_day_embeddings = global_model.extract_embeddings(zero_day_x)
            distances = torch.cdist(zero_day_embeddings.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
            closest_prototype = torch.argmin(distances, dim=1)
            logger.info(f"   Closest prototype (0=first, 1=second): {torch.bincount(closest_prototype, minlength=len(unique_labels)).tolist()}")
            logger.info(f"   Distances to prototypes (mean): {distances.mean(dim=0).cpu().tolist()}")
            
            # Check if prototypes are in correct order
            logger.info(f"   Prototype order: {unique_labels.tolist()} (index 0 → label {unique_labels[0].item()}, index 1 → label {unique_labels[1].item() if len(unique_labels) > 1 else 'N/A'})")
            
        else:
            logger.info(f"✅ ZDR is non-zero: {zero_day_detection_rate:.4f}")
        
        # Check overall predictions
        all_logits = global_model.forward_with_prototypes(X_test_tensor, prototypes)
        all_predictions_indices = torch.argmax(all_logits, dim=1)
        all_predictions = unique_labels[all_predictions_indices]
        overall_attack_rate = (all_predictions != 0).float().mean().item()
        logger.info(f"🔍 Overall attack prediction rate: {overall_attack_rate:.4f} ({100*overall_attack_rate:.1f}%)")

if __name__ == "__main__":
    diagnose_zero_zdr()









