"""
Diagnostic script to investigate why base and TTT models have identical zero-day performance
"""
import torch
import numpy as np
from main import BlockchainFederatedIncentiveSystem
from config import SystemConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_zero_day_performance():
    """Compare base and adapted model predictions on zero-day samples"""
    
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC: Zero-Day Performance Comparison")
    logger.info("=" * 80)
    
    # Initialize system
    config = SystemConfig()
    system = BlockchainFederatedIncentiveSystem(config)
    
    # Run preprocessing
    logger.info("🔍 Preprocessing data...")
    system.preprocess_data()
    
    # Run federated training
    logger.info("🔍 Running federated training...")
    system.run_federated_training()
    
    # Get test data
    X_test = system.preprocessed_data['X_test']
    y_test = system.preprocessed_data['y_test']
    y_test_multiclass = system.preprocessed_data.get('y_test_multiclass')
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(system.device)
    y_test_tensor = torch.LongTensor(y_test).to(system.device)
    
    # Create zero-day mask
    zero_day_attack_label = config.zero_day_attack_label
    if y_test_multiclass is not None:
        if not torch.is_tensor(y_test_multiclass):
            y_test_multiclass = torch.tensor(y_test_multiclass)
        y_test_multiclass = y_test_multiclass.to(system.device)
        zero_day_mask = (y_test_multiclass == zero_day_attack_label)
    else:
        zero_day_mask = torch.zeros(len(y_test_tensor), dtype=torch.bool)
    
    num_zero_day = zero_day_mask.sum().item()
    logger.info(f"📊 Found {num_zero_day} zero-day samples out of {len(y_test_tensor)} total")
    
    if num_zero_day == 0:
        logger.error("❌ No zero-day samples found! Cannot diagnose.")
        return
    
    # Get base model predictions
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Base Model Predictions")
    logger.info("=" * 80)
    
    base_model = system.coordinator.model
    base_model.eval()
    
    with torch.no_grad():
        base_logits = base_model(X_test_tensor)
        base_probabilities = torch.softmax(base_logits, dim=1)
        base_predictions = torch.argmax(base_logits, dim=1)
    
    # Extract zero-day predictions
    zero_day_base_predictions = base_predictions[zero_day_mask]
    zero_day_base_probs = base_probabilities[zero_day_mask]
    
    logger.info(f"📊 Base Model - Zero-day predictions:")
    logger.info(f"   Predictions distribution: {torch.bincount(zero_day_base_predictions, minlength=2).tolist()}")
    logger.info(f"   Attack probabilities - min: {zero_day_base_probs[:, 1].min():.4f}, max: {zero_day_base_probs[:, 1].max():.4f}, mean: {zero_day_base_probs[:, 1].mean():.4f}")
    
    # Get adapted model (after TTT)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: TTT Adapted Model Predictions")
    logger.info("=" * 80)
    
    logger.info("🔄 Performing TTT adaptation...")
    adapted_model = system.coordinator.adapt_to_test_data(
        query_x=X_test_tensor,
        query_y=None,
        config=config,
        method='tent_pseudo'
    )
    
    adapted_model.eval()
    adapted_model.set_ttt_mode(training=False)
    
    with torch.no_grad():
        adapted_logits = adapted_model(X_test_tensor)
        # Apply temperature scaling
        temperature = getattr(config, 'ttt_temperature', 1.5)
        if temperature != 1.0:
            adapted_logits = adapted_logits / temperature
        adapted_probabilities = torch.softmax(adapted_logits, dim=1)
        adapted_predictions = torch.argmax(adapted_logits, dim=1)
    
    # Extract zero-day predictions
    zero_day_adapted_predictions = adapted_predictions[zero_day_mask]
    zero_day_adapted_probs = adapted_probabilities[zero_day_mask]
    
    logger.info(f"📊 TTT Model - Zero-day predictions:")
    logger.info(f"   Predictions distribution: {torch.bincount(zero_day_adapted_predictions, minlength=2).tolist()}")
    logger.info(f"   Attack probabilities - min: {zero_day_adapted_probs[:, 1].min():.4f}, max: {zero_day_adapted_probs[:, 1].max():.4f}, mean: {zero_day_adapted_probs[:, 1].mean():.4f}")
    
    # Compare predictions
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Comparison")
    logger.info("=" * 80)
    
    # Check if predictions are identical
    predictions_match = (zero_day_base_predictions == zero_day_adapted_predictions).float().mean().item()
    logger.info(f"📊 Prediction Match Rate: {predictions_match:.1%} ({int(predictions_match * num_zero_day)}/{num_zero_day} identical)")
    
    if predictions_match == 1.0:
        logger.error("❌ CRITICAL: All zero-day predictions are IDENTICAL!")
        logger.error("   This means TTT did NOT change predictions for zero-day samples.")
    elif predictions_match > 0.95:
        logger.warning(f"⚠️  WARNING: {predictions_match:.1%} of zero-day predictions are identical!")
        logger.warning("   TTT is only changing a few predictions.")
    else:
        logger.info(f"✅ TTT is changing {1-predictions_match:.1%} of zero-day predictions")
    
    # Compare probabilities
    prob_diff = (zero_day_adapted_probs[:, 1] - zero_day_base_probs[:, 1]).abs()
    logger.info(f"📊 Probability Changes:")
    logger.info(f"   Mean absolute difference: {prob_diff.mean():.4f}")
    logger.info(f"   Max absolute difference: {prob_diff.max():.4f}")
    logger.info(f"   Min absolute difference: {prob_diff.min():.4f}")
    
    # Check actual labels
    zero_day_labels = y_test_tensor[zero_day_mask]
    logger.info(f"\n📊 Zero-day sample labels:")
    logger.info(f"   Label distribution: {torch.bincount(zero_day_labels, minlength=2).tolist()}")
    
    # Calculate metrics for comparison
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Metric Calculation")
    logger.info("=" * 80)
    
    # Base model metrics
    zero_day_base_binary = (zero_day_base_predictions != 0).long()
    zero_day_labels_binary = (zero_day_labels != 0).long()
    
    base_correct = (zero_day_base_binary == zero_day_labels_binary).float().mean().item()
    base_detection_rate = (zero_day_base_predictions != 0).float().mean().item()
    
    logger.info(f"📊 Base Model Zero-Day Metrics:")
    logger.info(f"   Accuracy: {base_correct:.4f}")
    logger.info(f"   Detection Rate (predicted as attack): {base_detection_rate:.4f}")
    
    # Adapted model metrics
    zero_day_adapted_binary = (zero_day_adapted_predictions != 0).long()
    
    adapted_correct = (zero_day_adapted_binary == zero_day_labels_binary).float().mean().item()
    adapted_detection_rate = (zero_day_adapted_predictions != 0).float().mean().item()
    
    logger.info(f"📊 TTT Model Zero-Day Metrics:")
    logger.info(f"   Accuracy: {adapted_correct:.4f}")
    logger.info(f"   Detection Rate (predicted as attack): {adapted_detection_rate:.4f}")
    
    if abs(base_correct - adapted_correct) < 1e-6:
        logger.error("\n❌ ROOT CAUSE IDENTIFIED:")
        logger.error("   Base and TTT models have IDENTICAL accuracy on zero-day samples!")
        logger.error("   This explains why performance is the same.")
        
        if predictions_match == 1.0:
            logger.error("\n   REASON: All predictions are identical (TTT didn't change anything)")
        else:
            logger.error(f"\n   REASON: Even though {1-predictions_match:.1%} of predictions changed,")
            logger.error("   the changes didn't improve accuracy (same number of correct predictions)")
            
            # Find which predictions changed
            changed_mask = (zero_day_base_predictions != zero_day_adapted_predictions)
            changed_base = zero_day_base_predictions[changed_mask]
            changed_adapted = zero_day_adapted_predictions[changed_mask]
            changed_labels = zero_day_labels[changed_mask]
            
            logger.info(f"\n   Changed predictions: {changed_mask.sum().item()}")
            logger.info(f"   Base predictions: {changed_base.tolist()}")
            logger.info(f"   Adapted predictions: {changed_adapted.tolist()}")
            logger.info(f"   True labels: {changed_labels.tolist()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    diagnose_zero_day_performance()










