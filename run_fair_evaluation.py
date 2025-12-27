"""
Run Fair Binary Evaluation for Zero-Day Detection
==================================================

This script runs a fair comparison between base and TTT models
using the same binary classifier for both evaluations.

Usage:
    python run_fair_evaluation.py --dataset CICIDS2017

Author: PhD Research
Date: 2025-01-17
"""

import torch
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import get_dataset_config
from fair_binary_evaluation import FairBinaryEvaluator
from main import BlockchainFederatedIncentiveSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fair_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run fair binary evaluation"""
    logger.info("=" * 80)
    logger.info("🚀 FAIR BINARY EVALUATION FOR ZERO-DAY DETECTION")
    logger.info("=" * 80)

    # Load configuration
    config = get_dataset_config()

    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Dataset: {config.data_path}")
    logger.info(f"  Zero-day attack: {config.zero_day_attack}")
    logger.info(f"  Category grouping: {config.use_category_grouping}")
    logger.info(f"  Meta epochs: {config.meta_epochs}")
    logger.info(f"  TTT steps: {config.ttt_base_steps}")
    logger.info(f"  TTT learning rate: {config.ttt_lr}")

    # Initialize system to load and preprocess data
    logger.info(f"\n📊 Loading and preprocessing data...")
    system = BlockchainFederatedIncentiveSystem(config)

    # Initialize system components
    success = system.initialize_system()
    if not success:
        logger.error("❌ Failed to initialize system")
        return

    # Preprocess data
    success = system.preprocess_data()
    if not success:
        logger.error("❌ Failed to preprocess data")
        return

    # Get preprocessed data
    preprocessed_data = system.preprocessed_data

    if preprocessed_data is None:
        logger.error("❌ Failed to load preprocessed data")
        return

    # Extract training data
    X_train = torch.FloatTensor(preprocessed_data['X_train'])
    y_train_binary = torch.LongTensor(preprocessed_data['y_train'])
    y_train_multiclass = torch.LongTensor(preprocessed_data.get('y_train_multiclass', preprocessed_data['y_train']))

    # Extract test data
    X_test = torch.FloatTensor(preprocessed_data['X_test'])
    y_test_binary = torch.LongTensor(preprocessed_data['y_test'])
    y_test_multiclass = torch.LongTensor(preprocessed_data.get('y_test_multiclass', preprocessed_data['y_test']))

    logger.info(f"\n✅ Data loaded:")
    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Training shape: {X_train.shape}")
    logger.info(f"  Test shape: {X_test.shape}")

    # Create zero-day mask
    # CRITICAL: Identify zero-day samples in test set
    zero_day_attack_label = config.zero_day_attack_label

    logger.info(f"\n🔍 Creating zero-day mask:")
    logger.info(f"  Zero-day attack: {config.zero_day_attack}")
    logger.info(f"  Zero-day label: {zero_day_attack_label}")

    # Zero-day mask: samples with the zero-day attack label
    zero_day_mask = (y_test_multiclass == zero_day_attack_label)

    # Verify zero-day mask
    zero_day_count = zero_day_mask.sum().item()
    logger.info(f"  Zero-day samples found: {zero_day_count:,} ({100*zero_day_count/len(X_test):.2f}%)")

    if zero_day_count == 0:
        logger.warning(f"⚠️ WARNING: No zero-day samples found in test set!")
        logger.warning(f"   Check if zero-day attack '{config.zero_day_attack}' (label {zero_day_attack_label}) exists in test data")
        logger.warning(f"   Available labels in test set: {torch.unique(y_test_multiclass).tolist()}")

    # Verify zero-day samples are actually attacks (not normal traffic)
    if zero_day_count > 0:
        zero_day_binary_labels = y_test_binary[zero_day_mask]
        zero_day_attacks = (zero_day_binary_labels == 1).sum().item()
        logger.info(f"  Zero-day samples that are attacks: {zero_day_attacks:,} ({100*zero_day_attacks/zero_day_count:.2f}%)")

        if zero_day_attacks == 0:
            logger.error(f"❌ ERROR: All zero-day samples are labeled as Normal!")
            logger.error(f"   This indicates a labeling error. Zero-day attacks should be labeled as Attack (1), not Normal (0)")
            return

    # Initialize fair evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = FairBinaryEvaluator(config, device=device)

    # Run full evaluation pipeline
    logger.info(f"\n🚀 Running full fair evaluation pipeline...")
    results = evaluator.run_full_evaluation(
        X_train=X_train,
        y_train_binary=y_train_binary,
        X_test=X_test,
        y_test_binary=y_test_binary,
        zero_day_mask=zero_day_mask,
        y_train_multiclass=y_train_multiclass
    )

    # Save results
    import json
    import numpy as np

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    results_json = convert_for_json(results)

    # Save to file
    output_file = 'fair_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 80)

    comparison = results['comparison']

    logger.info(f"\n🎯 Key Findings:")
    logger.info(f"  Overall Accuracy Improvement: {comparison['accuracy_improvement']:+.4f} ({comparison['accuracy_improvement_pct']:+.2f}%)")
    logger.info(f"  Zero-Day Detection Rate Improvement: {comparison['zero_day_detection_rate_improvement']:+.4f} ({comparison['zero_day_detection_rate_improvement_pct']:+.2f}%)")
    logger.info(f"  F1-Score Improvement: {comparison['f1_score_improvement']:+.4f} ({comparison['f1_score_improvement_pct']:+.2f}%)")
    logger.info(f"  FAR Reduction: {comparison['far_reduction']:+.4f} ({comparison['far_reduction_pct']:+.2f}%)")

    # Interpretation
    logger.info(f"\n💡 Interpretation:")
    if comparison['zero_day_detection_rate_improvement'] > 0.05:
        logger.info("  ✅ TTT provides SIGNIFICANT improvement for zero-day detection (+5%+)")
    elif comparison['zero_day_detection_rate_improvement'] > 0.01:
        logger.info("  ⚠️ TTT provides MARGINAL improvement for zero-day detection (+1-5%)")
    elif comparison['zero_day_detection_rate_improvement'] > -0.01:
        logger.info("  ⚪ TTT provides NO meaningful improvement for zero-day detection")
    else:
        logger.info("  ❌ TTT DEGRADES zero-day detection performance")

    logger.info("=" * 80)
    logger.info("✅ FAIR EVALUATION COMPLETED")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error during evaluation: {str(e)}", exc_info=True)
        raise
