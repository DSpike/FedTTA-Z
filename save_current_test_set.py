#!/usr/bin/env python3
"""
Helper script to save the current test set from a regular run.
This allows you to reuse the test set for future runs without re-running optimization.
"""

import pickle
from pathlib import Path
import logging
from config import SystemConfig
from main import BlockchainFederatedIncentiveSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_current_test_set():
    """Save the test set from the current configuration"""
    
    logger.info("🚀 Initializing system with current configuration...")
    
    # Initialize system
    config = SystemConfig()
    system = BlockchainFederatedIncentiveSystem(config)
    
    if not system.initialize_system():
        logger.error("❌ System initialization failed")
        return False
    
    logger.info("📦 Preprocessing data...")
    
    # Preprocess data (this will create the test set)
    if not system.preprocess_data():
        logger.error("❌ Data preprocessing failed")
        return False
    
    # Create saved_test_sets directory
    test_set_dir = Path("saved_test_sets")
    test_set_dir.mkdir(exist_ok=True)
    
    # Prepare test set data
    test_set_data = {
        'X_test': system.preprocessed_data.get('X_test'),
        'y_test': system.preprocessed_data.get('y_test'),
        'y_test_multiclass': system.preprocessed_data.get('y_test_multiclass'),
        'test_attack_cat': system.preprocessed_data.get('test_attack_cat'),
        'X_test_original': system.preprocessed_data.get('X_test_original'),
        'y_test_original': system.preprocessed_data.get('y_test_original'),
        'test_attack_cat_original': system.preprocessed_data.get('test_attack_cat_original'),
        'zero_day_indices': system.preprocessed_data.get('zero_day_indices'),
        'zero_day_attack': system.preprocessed_data.get('zero_day_attack'),
        'trial_number': 'current_run',  # Mark as current run
    }
    
    # Save test set
    test_set_path = test_set_dir / "test_set_best_trial.pkl"
    with open(test_set_path, 'wb') as f:
        pickle.dump(test_set_data, f)
    
    logger.info(f"✅ Test set saved to: {test_set_path}")
    logger.info(f"   Test samples: {len(test_set_data['X_test'])}")
    logger.info(f"   Zero-day attack: {test_set_data.get('zero_day_attack', 'unknown')}")
    
    if test_set_data.get('y_test_multiclass') is not None:
        import torch
        import numpy as np
        y_multiclass = test_set_data['y_test_multiclass']
        if torch.is_tensor(y_multiclass):
            y_multiclass_np = y_multiclass.cpu().numpy()
        else:
            y_multiclass_np = np.array(y_multiclass)
        
        zero_day_label = config.zero_day_attack_label
        zero_day_count = (y_multiclass_np == zero_day_label).sum()
        total_count = len(y_multiclass_np)
        zero_day_percentage = 100 * zero_day_count / total_count if total_count > 0 else 0
        
        logger.info(f"   Zero-day samples: {zero_day_count}/{total_count} ({zero_day_percentage:.1f}%)")
    
    logger.info("\n🎯 Next steps:")
    logger.info("   1. Run main.py normally - it will automatically use this saved test set")
    logger.info("   2. This ensures reproducible evaluation conditions")
    
    return True

if __name__ == "__main__":
    save_current_test_set()










