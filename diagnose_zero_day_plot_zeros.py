"""
Diagnostic script to investigate why zero-day performance plot shows zero values
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_zero_day_metrics():
    """Check why zero-day metrics might be zero"""
    
    # Check current config first
    try:
        from config import SystemConfig
        config = SystemConfig()
        logger.info("\n" + "="*80)
        logger.info("CURRENT CONFIGURATION")
        logger.info("="*80)
        logger.info(f"   Zero-day attack: '{config.zero_day_attack}'")
        logger.info(f"   Zero-day attack label: {config.zero_day_attack_label}")
        logger.info(f"   Dataset: {getattr(config, 'dataset_type', 'N/A')}")
        logger.info(f"   Data path: {config.data_path}")
        logger.info(f"   Test path: {config.test_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not load config: {e}")
    
    # Check saved results (excluding old edgeiiot files)
    result_files = [f for f in list(Path(".").glob("evaluation_results_*.json")) + list(Path(".").glob("*results*.json"))
                    if "edgeiiot" not in f.name.lower() and "decentralized" not in f.name.lower()]
    
    if not result_files:
        logger.warning("No result files found. Checking preprocessed data...")
        
        # Check preprocessed data
        preprocessed_path = Path("preprocessed_data.pkl")
        if preprocessed_path.exists():
            with open(preprocessed_path, 'rb') as f:
                preprocessed_data = pickle.load(f)
            
            logger.info("\n" + "="*80)
            logger.info("PREPROCESSED DATA DIAGNOSTIC")
            logger.info("="*80)
            
            # Check test set
            if 'X_test' in preprocessed_data:
                X_test = preprocessed_data['X_test']
                y_test = preprocessed_data['y_test']
                logger.info(f"✅ Test set found: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
                
                # Check multiclass labels
                if 'y_test_multiclass' in preprocessed_data:
                    y_test_multiclass = preprocessed_data['y_test_multiclass']
                    logger.info(f"✅ Multiclass labels found: shape={y_test_multiclass.shape if hasattr(y_test_multiclass, 'shape') else len(y_test_multiclass)}")
                    
                    # Convert to numpy/tensor for analysis
                    if torch.is_tensor(y_test_multiclass):
                        y_test_multiclass_np = y_test_multiclass.cpu().numpy()
                    else:
                        y_test_multiclass_np = np.array(y_test_multiclass)
                    
                    # Check unique labels
                    unique_labels = np.unique(y_test_multiclass_np)
                    logger.info(f"   Unique multiclass labels: {unique_labels}")
                    
                    # Count samples per label
                    for label in unique_labels:
                        count = np.sum(y_test_multiclass_np == label)
                        percentage = 100 * count / len(y_test_multiclass_np)
                        logger.info(f"   Label {label}: {count} samples ({percentage:.1f}%)")
                    
                    # Check for zero-day attack label (assuming PortScan=10 for CICIDS)
                    zero_day_labels = [10]  # PortScan for CICIDS
                    for zd_label in zero_day_labels:
                        zero_day_count = np.sum(y_test_multiclass_np == zd_label)
                        logger.info(f"\n🔍 Zero-day check (label {zd_label}): {zero_day_count} samples")
                        if zero_day_count == 0:
                            logger.warning(f"⚠️  NO zero-day samples found with label {zd_label}!")
                else:
                    logger.warning("⚠️  No 'y_test_multiclass' in preprocessed data!")
                
                # Check attack categories
                if 'test_attack_cat' in preprocessed_data:
                    test_attack_cat = preprocessed_data['test_attack_cat']
                    logger.info(f"✅ Attack categories found: {len(test_attack_cat)} categories")
                    
                    if isinstance(test_attack_cat, (list, np.ndarray)):
                        unique_attacks = np.unique(test_attack_cat)
                        logger.info(f"   Unique attack categories: {unique_attacks}")
                        
                        for attack in unique_attacks:
                            count = np.sum(np.array(test_attack_cat) == attack)
                            percentage = 100 * count / len(test_attack_cat)
                            logger.info(f"   '{attack}': {count} samples ({percentage:.1f}%)")
                        
                        # Check for PortScan (zero-day)
                        if 'PortScan' in unique_attacks:
                            portscan_count = np.sum(np.array(test_attack_cat) == 'PortScan')
                            logger.info(f"\n🔍 Zero-day check (PortScan): {portscan_count} samples")
                            if portscan_count == 0:
                                logger.warning("⚠️  NO PortScan samples found in test_attack_cat!")
                        else:
                            logger.warning("⚠️  PortScan NOT found in attack categories!")
                else:
                    logger.warning("⚠️  No 'test_attack_cat' in preprocessed data!")
                
                # Check test_attack_cat_original
                if 'test_attack_cat_original' in preprocessed_data:
                    test_attack_cat_original = preprocessed_data['test_attack_cat_original']
                    logger.info(f"\n✅ Original attack categories found: {len(test_attack_cat_original)} categories")
                    if isinstance(test_attack_cat_original, (list, np.ndarray)):
                        unique_attacks_orig = np.unique(test_attack_cat_original)
                        logger.info(f"   Unique original attack categories: {unique_attacks_orig[:10]}...")  # Show first 10
            else:
                logger.error("❌ No 'X_test' in preprocessed data!")
            
            logger.info("="*80)
        
    
    # Check evaluation results
    else:
        logger.info(f"Found {len(result_files)} result files")
        for result_file in result_files[:3]:  # Check first 3
            logger.info(f"\n📊 Checking {result_file.name}...")
            try:
                import json
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                # Check for zero-day metrics
                if 'base_model' in results:
                    base_results = results['base_model']
                    if 'zero_day_only' in base_results:
                        zero_day_metrics = base_results['zero_day_only']
                        num_samples = zero_day_metrics.get('num_samples', 0)
                        logger.info(f"   Base model zero-day samples: {num_samples}")
                        if num_samples == 0:
                            logger.warning("   ⚠️  Base model has 0 zero-day samples!")
                        else:
                            logger.info(f"   Base model zero-day accuracy: {zero_day_metrics.get('accuracy', 0):.4f}")
                            logger.info(f"   Base model zero-day detection rate: {zero_day_metrics.get('zero_day_detection_rate', 0):.4f}")
                    else:
                        logger.warning("   ⚠️  No 'zero_day_only' key in base_model results!")
                
                if 'adapted_model' in results:
                    adapted_results = results['adapted_model']
                    if 'zero_day_only' in adapted_results:
                        zero_day_metrics = adapted_results['zero_day_only']
                        num_samples = zero_day_metrics.get('num_samples', 0)
                        logger.info(f"   TTT model zero-day samples: {num_samples}")
                        if num_samples == 0:
                            logger.warning("   ⚠️  TTT model has 0 zero-day samples!")
                    else:
                        logger.warning("   ⚠️  No 'zero_day_only' key in adapted_model results!")
                        
            except Exception as e:
                logger.error(f"   ❌ Error reading {result_file}: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*80)
    logger.info("\n💡 Possible causes for zero metrics:")
    logger.info("   1. No zero-day samples found in test set (check multiclass labels)")
    logger.info("   2. Zero-day mask creation failed (size mismatch)")
    logger.info("   3. Zero-day attack label mismatch (check config.zero_day_attack_label)")
    logger.info("   4. Test set composition issue (all samples filtered out)")
    logger.info("\n📝 Check logs for:")
    logger.info("   - 'No zero-day samples found!' warnings")
    logger.info("   - 'Zero-day mask created: X/Y samples' messages")
    logger.info("   - 'Extracted X zero-day samples' debug messages")

if __name__ == "__main__":
    diagnose_zero_day_metrics()

