#!/usr/bin/env python3
"""
Diagnose why zero_day_mask is not identifying Backdoor samples
"""
import logging
import sys
from config_loader import get_dataset_config
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose():
    logger.info("="*80)
    logger.info("DIAGNOSING ZERO-DAY MASK ISSUE")
    logger.info("="*80)

    # Load config
    config = get_dataset_config()
    logger.info(f"\n1. CONFIGURATION:")
    logger.info(f"   zero_day_attack: '{config.zero_day_attack}'")
    logger.info(f"   use_category_grouping: {config.use_category_grouping}")
    logger.info(f"   data_path: {config.data_path}")
    logger.info(f"   test_path: {config.test_path}")

    # Load preprocessor
    logger.info(f"\n2. LOADING PREPROCESSOR:")
    preprocessor = UNSWPreprocessor(config.input_dim, config)

    # Check attack types mapping
    logger.info(f"\n3. ATTACK TYPES MAPPING:")
    for attack_name, label in preprocessor.attack_types.items():
        logger.info(f"   '{attack_name}': {label}")

    # Check if Backdoor is in mapping
    if config.zero_day_attack in preprocessor.attack_types:
        logger.info(f"\n   ✅ '{config.zero_day_attack}' found in attack_types")
        logger.info(f"   Label: {preprocessor.attack_types[config.zero_day_attack]}")
    else:
        logger.error(f"\n   ❌ '{config.zero_day_attack}' NOT found in attack_types!")
        logger.error(f"   Available attacks: {list(preprocessor.attack_types.keys())}")

    # Load and preprocess data
    logger.info(f"\n4. LOADING AND PREPROCESSING DATA:")
    data_dict = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)

    test_df = data_dict.get('test_df')
    if test_df is None:
        logger.error("   ❌ test_df not found in data_dict!")
        logger.info(f"   Available keys: {list(data_dict.keys())}")
        return

    logger.info(f"   Test set shape: {test_df.shape}")

    # Check test_df attack categories
    if 'attack_cat' in test_df.columns:
        attack_cats = test_df['attack_cat'].values
        logger.info(f"\n5. TEST SET ATTACK CATEGORIES:")
        logger.info(f"   Total samples: {len(attack_cats)}")

        from collections import Counter
        cat_counts = Counter(attack_cats)
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(attack_cats)
            logger.info(f"   {cat}: {count} ({pct:.1f}%)")

        # Check for Backdoor
        backdoor_count = (attack_cats == config.zero_day_attack).sum()
        logger.info(f"\n   🎯 '{config.zero_day_attack}' samples: {backdoor_count} ({100*backdoor_count/len(attack_cats):.1f}%)")

        if backdoor_count > 0:
            logger.info(f"   ✅ Backdoor samples EXIST in test set!")

            # Check binary labels for Backdoor
            backdoor_mask = attack_cats == config.zero_day_attack
            backdoor_binary_labels = test_df.loc[backdoor_mask, 'binary_label'].values
            logger.info(f"   Binary labels for Backdoor: {Counter(backdoor_binary_labels)}")

            if (backdoor_binary_labels == 1).all():
                logger.info(f"   ✅ All Backdoor samples have binary_label=1 (correct)")
            else:
                logger.error(f"   ❌ Some Backdoor samples have binary_label=0 (WRONG!)")
        else:
            logger.error(f"   ❌ NO Backdoor samples in test set!")
    else:
        logger.error(f"   ❌ 'attack_cat' column not found in test_df!")
        logger.info(f"   Available columns: {list(test_df.columns)}")

    # Test _is_zero_day_attack logic
    logger.info(f"\n6. TESTING _is_zero_day_attack() LOGIC:")

    # Simulate the logic from main.py
    def _is_zero_day_attack(attack_name: str) -> bool:
        if config.use_category_grouping and config.attack_category_mapping:
            attack_category = config.attack_category_mapping.get(attack_name, None)
            zero_day_category = config.zero_day_category
            return attack_category == zero_day_category
        else:
            # Specific attack: Direct comparison
            return attack_name == config.zero_day_attack

    # Test with sample attack names
    test_attacks = ['Backdoor', 'backdoor', 'Normal', 'Fuzzers', 'DoS']
    for attack in test_attacks:
        result = _is_zero_day_attack(attack)
        logger.info(f"   _is_zero_day_attack('{attack}'): {result}")

    # Check if test_df has attack_cat that matches
    if 'attack_cat' in test_df.columns:
        unique_cats = test_df['attack_cat'].unique()
        logger.info(f"\n7. CHECKING ACTUAL TEST SET ATTACK CATEGORIES:")
        for cat in unique_cats:
            is_zd = _is_zero_day_attack(cat)
            count = (test_df['attack_cat'] == cat).sum()
            logger.info(f"   '{cat}': {count} samples, is_zero_day={is_zd}")

    logger.info(f"\n" + "="*80)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    diagnose()
