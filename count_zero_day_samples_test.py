"""
Script to count zero-day samples during TTT/Base model test evaluation
"""
import torch
import numpy as np
import logging
from config import SystemConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_zero_day_samples():
    """Count zero-day samples in test data for evaluation"""
    try:
        # Load config
        config = SystemConfig()
        logger.info(f"Zero-day attack configured: '{config.zero_day_attack}' (label: {config.zero_day_attack_label})")
        
        # Load preprocessed data using the same method as main.py
        logger.info("Initializing system and loading preprocessed data...")
        from main import BlockchainFederatedIncentiveSystem
        
        # Create system instance
        system = BlockchainFederatedIncentiveSystem(config=config)
        
        # Initialize system components
        logger.info("Initializing system components...")
        if not system.initialize_system():
            logger.error("System initialization failed")
            return
        
        # Preprocess data (this loads the preprocessed data)
        logger.info("Preprocessing data (this may take a moment)...")
        if not system.preprocess_data():
            logger.error("Data preprocessing failed")
            return
        
        if not hasattr(system, 'preprocessed_data') or system.preprocessed_data is None:
            logger.error("Preprocessed data not found after preprocessing")
            return
        
        preprocessed_data = system.preprocessed_data
        logger.info("✅ Preprocessed data loaded successfully")
        
        # Get test data
        if 'X_test' not in preprocessed_data or 'y_test' not in preprocessed_data:
            logger.error("Test data not found in preprocessed_data")
            logger.info(f"Available keys: {list(preprocessed_data.keys())}")
            return
        
        X_test = preprocessed_data['X_test']
        y_test = preprocessed_data['y_test']
        
        # Convert to tensors if needed
        if isinstance(X_test, np.ndarray):
            X_test_tensor = torch.FloatTensor(X_test)
        elif hasattr(X_test, 'numpy'):
            X_test_tensor = X_test
        else:
            X_test_tensor = torch.FloatTensor(X_test)
        
        if isinstance(y_test, np.ndarray):
            y_test_tensor = torch.LongTensor(y_test)
        elif hasattr(y_test, 'numpy'):
            y_test_tensor = y_test
        else:
            y_test_tensor = torch.LongTensor(y_test)
        
        logger.info(f"Test data shape: X={X_test_tensor.shape}, y={y_test_tensor.shape}")
        
        # Create zero-day mask
        zero_day_mask = None
        
        # Method 1: Use multiclass labels if available
        if 'y_test_multiclass' in preprocessed_data:
            y_test_multiclass = preprocessed_data['y_test_multiclass']
            if isinstance(y_test_multiclass, np.ndarray):
                y_test_multiclass_tensor = torch.LongTensor(y_test_multiclass)
            else:
                y_test_multiclass_tensor = y_test_multiclass
            
            logger.info("Using multiclass labels to identify zero-day samples")
            zero_day_mask = (y_test_multiclass_tensor == config.zero_day_attack_label)
            logger.info(f"Multiclass label distribution: {torch.bincount(y_test_multiclass_tensor)}")
        
        # Method 2: Use attack_cat if available
        elif 'test_attack_cat' in preprocessed_data:
            test_attack_cat = preprocessed_data['test_attack_cat']
            logger.info("Using attack_cat to identify zero-day samples")
            
            if isinstance(test_attack_cat, np.ndarray):
                # For sequence data, need to handle properly
                sequence_length = config.sequence_length
                sequence_stride = config.sequence_stride
                num_original_samples = len(test_attack_cat)
                
                zero_day_mask = torch.zeros(len(y_test_tensor), dtype=torch.bool)
                zero_day_count = 0
                
                for seq_idx in range(len(y_test_tensor)):
                    original_idx = seq_idx * sequence_stride + (sequence_length - 1)
                    if original_idx < num_original_samples:
                        if test_attack_cat[original_idx] == config.zero_day_attack:
                            zero_day_mask[seq_idx] = True
                            zero_day_count += 1
                
                logger.info(f"Identified {zero_day_count} zero-day sequences from {num_original_samples} original samples")
            else:
                # Direct comparison
                zero_day_mask = torch.tensor([cat == config.zero_day_attack for cat in test_attack_cat], dtype=torch.bool)
        
        # Method 3: Fallback - check unique values
        else:
            logger.warning("No multiclass labels or attack_cat available. Checking unique labels...")
            unique_labels = torch.unique(y_test_tensor)
            logger.info(f"Unique labels in test data: {unique_labels.tolist()}")
            
            if config.zero_day_attack_label in unique_labels:
                zero_day_mask = (y_test_tensor == config.zero_day_attack_label)
            else:
                logger.error(f"Zero-day attack label {config.zero_day_attack_label} not found in test data!")
                return
        
        # Count zero-day samples
        if zero_day_mask is not None:
            num_zero_day = zero_day_mask.sum().item()
            num_non_zero_day = (~zero_day_mask).sum().item()
            total_test = len(y_test_tensor)
            
            logger.info("=" * 70)
            logger.info("📊 ZERO-DAY SAMPLE COUNT FOR BASE MODEL TEST")
            logger.info("=" * 70)
            logger.info(f"Total test samples: {total_test}")
            logger.info(f"Zero-day samples ({config.zero_day_attack}): {num_zero_day}")
            logger.info(f"Non-zero-day samples: {num_non_zero_day}")
            logger.info(f"Zero-day percentage: {100*num_zero_day/total_test:.2f}%")
            logger.info(f"Non-zero-day percentage: {100*num_non_zero_day/total_test:.2f}%")
            
            # Check label distribution
            logger.info(f"\nLabel distribution in test data:")
            label_counts = torch.bincount(y_test_tensor)
            for label_id, count in enumerate(label_counts):
                logger.info(f"  Label {label_id}: {count} samples ({100*count/total_test:.2f}%)")
            
            # For TTT evaluation - query set typically uses all test samples
            logger.info("\n" + "=" * 70)
            logger.info("📊 ZERO-DAY SAMPLE COUNT FOR TTT EVALUATION (Query Set)")
            logger.info("=" * 70)
            logger.info(f"TTT query set typically uses ALL test samples: {total_test}")
            logger.info(f"Zero-day samples in query set: {num_zero_day}")
            logger.info(f"Non-zero-day samples in query set: {num_non_zero_day}")
            
            # Additional info about support set (if applicable)
            support_size_typical = min(200, total_test // 3)
            logger.info(f"\nTypical TTT support set size: ~{support_size_typical} samples")
            logger.info(f"(Support set may contain both zero-day and non-zero-day samples)")
            
            logger.info("=" * 70)
            
            return {
                'total_test_samples': total_test,
                'zero_day_samples': num_zero_day,
                'non_zero_day_samples': num_non_zero_day,
                'zero_day_percentage': 100*num_zero_day/total_test,
                'zero_day_attack': config.zero_day_attack,
                'zero_day_attack_label': config.zero_day_attack_label
            }
        else:
            logger.error("Failed to create zero-day mask")
            return None
            
    except Exception as e:
        logger.error(f"Error counting zero-day samples: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    result = count_zero_day_samples()
    if result:
        print("\n✅ Zero-day sample count completed successfully!")
    else:
        print("\n❌ Failed to count zero-day samples")

