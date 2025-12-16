"""
Script to display validation set sample distribution
Shows class distribution, attack type distribution, and sample counts
"""
import torch
import numpy as np
import logging
from config import SystemConfig
from main import BlockchainFederatedIncentiveSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_validation_distribution():
    """Display detailed validation set distribution"""
    logger.info("="*80)
    logger.info("VALIDATION SET SAMPLE DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    # Initialize system
    config = SystemConfig()
    system = BlockchainFederatedIncentiveSystem(config)
    
    if not system.initialize_system():
        logger.error("❌ System initialization failed")
        return
    
    # Preprocess data
    logger.info("\n📦 Preprocessing data...")
    if not system.preprocess_data():
        logger.error("❌ Data preprocessing failed")
        return
    
    # Get validation data
    preprocessed_data = system.preprocessed_data
    
    if 'X_val' not in preprocessed_data or 'y_val' not in preprocessed_data:
        logger.error("❌ Validation data not found in preprocessed_data")
        return
    
    X_val = preprocessed_data['X_val']
    y_val = preprocessed_data['y_val']
    y_val_multiclass = preprocessed_data.get('y_val_multiclass', None)
    
    # Convert to numpy if tensor
    if torch.is_tensor(X_val):
        X_val_np = X_val.cpu().numpy()
    else:
        X_val_np = np.array(X_val)
    
    if torch.is_tensor(y_val):
        y_val_np = y_val.cpu().numpy()
    else:
        y_val_np = np.array(y_val)
    
    if y_val_multiclass is not None:
        if torch.is_tensor(y_val_multiclass):
            y_val_multiclass_np = y_val_multiclass.cpu().numpy()
        else:
            y_val_multiclass_np = np.array(y_val_multiclass)
    else:
        y_val_multiclass_np = None
    
    # Get attack type mapping
    attack_types = preprocessed_data.get('attack_types', {})
    attack_type_reverse = {v: k for k, v in attack_types.items()}
    
    logger.info("\n" + "="*80)
    logger.info("📊 VALIDATION SET OVERVIEW")
    logger.info("="*80)
    logger.info(f"Total validation samples: {len(X_val_np):,}")
    logger.info(f"Feature dimension: {X_val_np.shape[1] if len(X_val_np.shape) > 1 else 'N/A'}")
    
    # Binary label distribution
    logger.info("\n" + "-"*80)
    logger.info("📈 BINARY LABEL DISTRIBUTION (Normal vs Attack)")
    logger.info("-"*80)
    unique_binary, counts_binary = np.unique(y_val_np, return_counts=True)
    total_binary = len(y_val_np)
    
    for label, count in zip(unique_binary, counts_binary):
        label_name = "Normal" if label == 0 else "Attack"
        percentage = (count / total_binary) * 100
        logger.info(f"  {label_name} (label {label}): {count:,} samples ({percentage:.2f}%)")
    
    # Multiclass label distribution
    if y_val_multiclass_np is not None:
        logger.info("\n" + "-"*80)
        logger.info("📈 MULTICLASS LABEL DISTRIBUTION (Attack Types)")
        logger.info("-"*80)
        unique_multiclass, counts_multiclass = np.unique(y_val_multiclass_np, return_counts=True)
        total_multiclass = len(y_val_multiclass_np)
        
        # Sort by label value
        sorted_indices = np.argsort(unique_multiclass)
        unique_multiclass = unique_multiclass[sorted_indices]
        counts_multiclass = counts_multiclass[sorted_indices]
        
        logger.info(f"  Total unique classes: {len(unique_multiclass)}")
        logger.info("")
        
        for label, count in zip(unique_multiclass, counts_multiclass):
            label_name = attack_type_reverse.get(label, f"Unknown (label {label})")
            percentage = (count / total_multiclass) * 100
            logger.info(f"  {label_name:20s} (label {label:2d}): {count:7,} samples ({percentage:6.2f}%)")
        
        # Check for zero-day attack
        zero_day_attack = config.zero_day_attack
        zero_day_label = attack_types.get(zero_day_attack, None)
        
        if zero_day_label is not None:
            zero_day_count = np.sum(y_val_multiclass_np == zero_day_label)
            logger.info("\n" + "-"*80)
            logger.info(f"🔍 ZERO-DAY ATTACK CHECK: {zero_day_attack} (label {zero_day_label})")
            logger.info("-"*80)
            if zero_day_count == 0:
                logger.info(f"  ✅ Zero-day attack '{zero_day_attack}' correctly EXCLUDED from validation set")
                logger.info(f"     Count: {zero_day_count} samples")
            else:
                logger.warning(f"  ⚠️  Zero-day attack '{zero_day_attack}' found in validation set!")
                logger.warning(f"     Count: {zero_day_count} samples (should be 0)")
    
    # Sequence-level distribution (if sequences exist)
    if len(X_val_np.shape) == 3:  # (num_sequences, sequence_length, features)
        logger.info("\n" + "-"*80)
        logger.info("📈 SEQUENCE-LEVEL DISTRIBUTION")
        logger.info("-"*80)
        logger.info(f"  Number of sequences: {X_val_np.shape[0]:,}")
        logger.info(f"  Sequence length: {X_val_np.shape[1]}")
        logger.info(f"  Features per timestep: {X_val_np.shape[2]}")
        logger.info(f"  Total timesteps: {X_val_np.shape[0] * X_val_np.shape[1]:,}")
    elif len(X_val_np.shape) == 2:  # (num_samples, features)
        logger.info("\n" + "-"*80)
        logger.info("📈 SAMPLE-LEVEL DISTRIBUTION")
        logger.info("-"*80)
        logger.info(f"  Number of samples: {X_val_np.shape[0]:,}")
        logger.info(f"  Features per sample: {X_val_np.shape[1]}")
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Binary class balance
    if len(unique_binary) == 2:
        normal_count = counts_binary[unique_binary == 0][0] if 0 in unique_binary else 0
        attack_count = counts_binary[unique_binary == 1][0] if 1 in unique_binary else 0
        total_count = normal_count + attack_count
        
        if total_count > 0:
            normal_pct = (normal_count / total_count) * 100
            attack_pct = (attack_count / total_count) * 100
            balance_ratio = min(normal_count, attack_count) / max(normal_count, attack_count) if max(normal_count, attack_count) > 0 else 0
            
            logger.info(f"  Binary Class Balance Ratio: {balance_ratio:.3f} (1.0 = perfectly balanced)")
            logger.info(f"  Normal samples: {normal_pct:.2f}%")
            logger.info(f"  Attack samples: {attack_pct:.2f}%")
    
    # Multiclass diversity
    if y_val_multiclass_np is not None:
        num_classes = len(unique_multiclass)
        logger.info(f"  Number of attack types: {num_classes}")
        
        # Calculate diversity (how evenly distributed)
        if len(counts_multiclass) > 1:
            max_count = np.max(counts_multiclass)
            min_count = np.min(counts_multiclass)
            diversity = min_count / max_count if max_count > 0 else 0
            logger.info(f"  Class diversity: {diversity:.3f} (1.0 = perfectly uniform distribution)")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Validation set distribution analysis complete!")
    logger.info("="*80)

if __name__ == "__main__":
    show_validation_distribution()










