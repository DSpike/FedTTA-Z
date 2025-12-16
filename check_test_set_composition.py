#!/usr/bin/env python3
"""
Check Test Set Composition - Verify Zero-Day Samples
"""
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from config import SystemConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_test_set_composition():
    """Check the current test set composition"""
    config = SystemConfig()
    
    logger.info("=" * 80)
    logger.info("TEST SET COMPOSITION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Zero-day attack configured: '{config.zero_day_attack}' (label: {config.zero_day_attack_label})")
    logger.info("")
    
    # Method 1: Check saved test set files
    saved_test_sets_dir = Path("saved_test_sets")
    if saved_test_sets_dir.exists():
        logger.info("📦 Checking saved test sets...")
        for test_file in saved_test_sets_dir.glob("test_set_*.pkl"):
            try:
                with open(test_file, 'rb') as f:
                    saved_data = pickle.load(f)
                
                logger.info(f"\n📄 File: {test_file.name}")
                if 'zero_day_attack' in saved_data:
                    logger.info(f"   Zero-day attack: {saved_data['zero_day_attack']}")
                if 'X_test' in saved_data:
                    x_test = saved_data['X_test']
                    y_test = saved_data.get('y_test', None)
                    y_test_multiclass = saved_data.get('y_test_multiclass', None)
                    test_attack_cat = saved_data.get('test_attack_cat', None)
                    
                    logger.info(f"   Total samples: {len(x_test)}")
                    
                    if y_test is not None:
                        if torch.is_tensor(y_test):
                            y_test_np = y_test.cpu().numpy()
                        else:
                            y_test_np = np.array(y_test)
                        
                        unique_labels, counts = np.unique(y_test_np, return_counts=True)
                        logger.info(f"   Binary labels: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
                    
                    if y_test_multiclass is not None:
                        if torch.is_tensor(y_test_multiclass):
                            y_multiclass_np = y_test_multiclass.cpu().numpy()
                        else:
                            y_multiclass_np = np.array(y_test_multiclass)
                        
                        unique_multiclass, counts_multiclass = np.unique(y_multiclass_np, return_counts=True)
                        logger.info(f"   Multiclass labels: {dict(zip(unique_multiclass.tolist(), counts_multiclass.tolist()))}")
                        
                        # Check for zero-day
                        zero_day_label = config.zero_day_attack_label
                        zero_day_count = (y_multiclass_np == zero_day_label).sum()
                        logger.info(f"   Zero-day samples (label {zero_day_label}): {zero_day_count} ({zero_day_count/len(y_multiclass_np)*100:.1f}%)")
                    
                    if test_attack_cat is not None:
                        if isinstance(test_attack_cat, list):
                            attack_cat_array = np.array(test_attack_cat)
                        else:
                            attack_cat_array = np.array(test_attack_cat)
                        
                        unique_attacks = np.unique(attack_cat_array)
                        logger.info(f"   Attack categories: {unique_attacks.tolist()}")
                        
                        zero_day_count_cat = (attack_cat_array == config.zero_day_attack).sum()
                        benign_count = (attack_cat_array == 'BENIGN').sum()
                        other_attacks = (attack_cat_array != 'BENIGN') & (attack_cat_array != config.zero_day_attack)
                        other_attacks_count = other_attacks.sum()
                        
                        logger.info(f"\n   📊 COMPOSITION:")
                        logger.info(f"      BENIGN/Normal: {benign_count} ({benign_count/len(attack_cat_array)*100:.1f}%)")
                        logger.info(f"      Zero-day ({config.zero_day_attack}): {zero_day_count_cat} ({zero_day_count_cat/len(attack_cat_array)*100:.1f}%)")
                        logger.info(f"      Other attacks: {other_attacks_count} ({other_attacks_count/len(attack_cat_array)*100:.1f}%)")
                        
                        # Show distribution of all attack types
                        attack_counts = {}
                        for attack in unique_attacks:
                            count = (attack_cat_array == attack).sum()
                            attack_counts[attack] = count
                        
                        logger.info(f"\n   📈 Detailed distribution:")
                        for attack, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
                            pct = count / len(attack_cat_array) * 100
                            logger.info(f"      '{attack}': {count} ({pct:.1f}%)")
                        
            except Exception as e:
                logger.error(f"   ❌ Error loading {test_file}: {e}")
    
    # Method 2: Try to load from main system if it's been run
    logger.info("\n" + "=" * 80)
    logger.info("Checking if main.py has been run (preprocessed_data)...")
    logger.info("=" * 80)
    
    try:
        from main import BlockchainFederatedIncentiveSystem
        
        system = BlockchainFederatedIncentiveSystem(config=config)
        if system.initialize_system():
            logger.info("✅ System initialized")
            
            if system.preprocess_data():
                logger.info("✅ Data preprocessed")
                
                if hasattr(system, 'preprocessed_data') and system.preprocessed_data:
                    preprocessed = system.preprocessed_data
                    
                    if 'X_test' in preprocessed:
                        X_test = preprocessed['X_test']
                        y_test = preprocessed.get('y_test', None)
                        y_test_multiclass = preprocessed.get('y_test_multiclass', None)
                        test_attack_cat = preprocessed.get('test_attack_cat', None)
                        
                        logger.info(f"\n📊 CURRENT TEST SET COMPOSITION:")
                        logger.info(f"   Total samples: {len(X_test)}")
                        
                        if y_test is not None:
                            if torch.is_tensor(y_test):
                                y_test_np = y_test.cpu().numpy()
                            else:
                                y_test_np = np.array(y_test)
                            
                            unique_labels, counts = np.unique(y_test_np, return_counts=True)
                            logger.info(f"   Binary labels: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
                        
                        if y_test_multiclass is not None:
                            if torch.is_tensor(y_test_multiclass):
                                y_multiclass_np = y_test_multiclass.cpu().numpy()
                            else:
                                y_multiclass_np = np.array(y_test_multiclass)
                            
                            unique_multiclass, counts_multiclass = np.unique(y_multiclass_np, return_counts=True)
                            logger.info(f"   Multiclass labels: {dict(zip(unique_multiclass.tolist(), counts_multiclass.tolist()))}")
                            
                            # Check for zero-day
                            zero_day_label = config.zero_day_attack_label
                            zero_day_count = (y_multiclass_np == zero_day_label).sum()
                            logger.info(f"   Zero-day samples (label {zero_day_label}): {zero_day_count} ({zero_day_count/len(y_multiclass_np)*100:.1f}%)")
                        
                        if test_attack_cat is not None:
                            if isinstance(test_attack_cat, list):
                                attack_cat_array = np.array(test_attack_cat)
                            elif isinstance(test_attack_cat, np.ndarray):
                                attack_cat_array = test_attack_cat
                            elif torch.is_tensor(test_attack_cat):
                                attack_cat_array = test_attack_cat.cpu().numpy()
                            else:
                                attack_cat_array = np.array(test_attack_cat)
                            
                            unique_attacks = np.unique(attack_cat_array)
                            logger.info(f"   Attack categories found: {len(unique_attacks)} types")
                            logger.info(f"   Categories: {unique_attacks.tolist()}")
                            
                            zero_day_count_cat = (attack_cat_array == config.zero_day_attack).sum()
                            benign_count = (attack_cat_array == 'BENIGN').sum()
                            other_attacks = (attack_cat_array != 'BENIGN') & (attack_cat_array != config.zero_day_attack)
                            other_attacks_count = other_attacks.sum()
                            
                            logger.info(f"\n   📊 COMPOSITION:")
                            logger.info(f"      BENIGN/Normal: {benign_count} ({benign_count/len(attack_cat_array)*100:.1f}%)")
                            logger.info(f"      Zero-day ({config.zero_day_attack}): {zero_day_count_cat} ({zero_day_count_cat/len(attack_cat_array)*100:.1f}%)")
                            logger.info(f"      Other attacks: {other_attacks_count} ({other_attacks_count/len(attack_cat_array)*100:.1f}%)")
                            
                            # Show distribution of all attack types
                            attack_counts = {}
                            for attack in unique_attacks:
                                count = (attack_cat_array == attack).sum()
                                attack_counts[attack] = count
                            
                            logger.info(f"\n   📈 Detailed distribution:")
                            for attack, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
                                pct = count / len(attack_cat_array) * 100
                                marker = "🎯" if attack == config.zero_day_attack else ""
                                logger.info(f"      {marker} '{attack}': {count} ({pct:.1f}%)")
                            
                            # Final verdict
                            logger.info("\n" + "=" * 80)
                            if zero_day_count_cat > 0:
                                logger.info(f"✅ SUCCESS: Test set contains {zero_day_count_cat} zero-day samples ({zero_day_count_cat/len(attack_cat_array)*100:.1f}%)")
                            else:
                                logger.warning(f"⚠️  WARNING: Test set contains NO zero-day samples!")
                                logger.warning(f"   Looking for: '{config.zero_day_attack}'")
                                logger.warning(f"   Available categories: {unique_attacks.tolist()}")
                            logger.info("=" * 80)
                        else:
                            logger.warning("⚠️  test_attack_cat not found in preprocessed_data")
                    else:
                        logger.warning("⚠️  X_test not found in preprocessed_data")
                else:
                    logger.warning("⚠️  preprocessed_data not available")
            else:
                logger.warning("⚠️  Data preprocessing failed")
        else:
            logger.warning("⚠️  System initialization failed")
    except Exception as e:
        logger.error(f"❌ Error checking preprocessed data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    check_test_set_composition()









