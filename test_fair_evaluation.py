"""
Quick Test for Fair Binary Evaluation
======================================

This script tests the fair evaluation implementation with synthetic data
before running on real CICIDS2017 dataset.

Usage:
    python test_fair_evaluation.py
"""

import torch
import numpy as np
import logging
from config_loader import get_dataset_config
from fair_binary_evaluation import FairBinaryEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_data(n_train=1000, n_test=200, input_dim=10, seq_len=20):
    """Create synthetic data for testing"""
    logger.info("Creating synthetic data for testing...")

    # Training data
    X_train = torch.randn(n_train, seq_len, input_dim)
    y_train_binary = torch.randint(0, 2, (n_train,))  # Binary: 0=Normal, 1=Attack

    # Test data
    X_test = torch.randn(n_test, seq_len, input_dim)
    y_test_binary = torch.randint(0, 2, (n_test,))

    # Zero-day mask (30% of test data)
    zero_day_mask = torch.zeros(n_test, dtype=torch.bool)
    zero_day_indices = torch.randperm(n_test)[:int(n_test * 0.3)]
    zero_day_mask[zero_day_indices] = True

    logger.info(f"  Train: {n_train} samples, Test: {n_test} samples")
    logger.info(f"  Zero-day: {zero_day_mask.sum()} samples ({100*zero_day_mask.sum()/n_test:.1f}%)")

    return X_train, y_train_binary, X_test, y_test_binary, zero_day_mask


def test_fair_evaluator():
    """Test fair binary evaluator with synthetic data"""
    logger.info("=" * 80)
    logger.info("TESTING FAIR BINARY EVALUATOR")
    logger.info("=" * 80)

    # Load config (for hyperparameters)
    try:
        config = get_dataset_config('CICIDS2017')
    except:
        logger.warning("Could not load CICIDS2017 config, using default")
        from config import SystemConfig
        config = SystemConfig()

    # Override config for quick testing
    config.meta_epochs = 2  # Quick test (normally 20+)
    config.num_meta_tasks = 10  # Fewer tasks
    config.k_shot = 20
    config.n_query = 10
    config.ttt_base_steps = 10  # Quick adaptation
    config.input_dim = 10
    config.sequence_length = 20

    logger.info(f"\nTest Configuration:")
    logger.info(f"  Meta epochs: {config.meta_epochs}")
    logger.info(f"  TTT steps: {config.ttt_base_steps}")

    # Create synthetic data
    X_train, y_train_binary, X_test, y_test_binary, zero_day_mask = create_synthetic_data(
        n_train=1000,
        n_test=200,
        input_dim=config.input_dim,
        seq_len=config.sequence_length
    )

    # Initialize evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nUsing device: {device}")

    evaluator = FairBinaryEvaluator(config, device=device)

    # Test 1: Train binary model
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Training Binary Model")
    logger.info("=" * 80)
    try:
        model = evaluator.train_binary_model(X_train, y_train_binary)
        logger.info("✅ Test 1 PASSED: Binary model trained successfully")
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {str(e)}")
        raise

    # Test 2: Evaluate base model
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Evaluating Base Model")
    logger.info("=" * 80)
    try:
        base_results = evaluator.evaluate_base_model(X_test, y_test_binary, zero_day_mask)
        logger.info("✅ Test 2 PASSED: Base model evaluated successfully")
        logger.info(f"   Base Accuracy: {base_results['accuracy']:.4f}")
        logger.info(f"   Base ZDR: {base_results['zero_day_detection_rate']:.4f}")
    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: {str(e)}")
        raise

    # Test 3: Apply TTT adaptation
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Applying TTT Adaptation")
    logger.info("=" * 80)
    try:
        adapted_model = evaluator.apply_ttt_adaptation(X_test, support_ratio=0.3)
        logger.info("✅ Test 3 PASSED: TTT adaptation applied successfully")
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {str(e)}")
        raise

    # Test 4: Evaluate TTT model
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Evaluating TTT Model")
    logger.info("=" * 80)
    try:
        ttt_results = evaluator.evaluate_ttt_model(adapted_model, X_test, y_test_binary, zero_day_mask)
        logger.info("✅ Test 4 PASSED: TTT model evaluated successfully")
        logger.info(f"   TTT Accuracy: {ttt_results['accuracy']:.4f}")
        logger.info(f"   TTT ZDR: {ttt_results['zero_day_detection_rate']:.4f}")
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {str(e)}")
        raise

    # Test 5: Compare results
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Comparing Results")
    logger.info("=" * 80)
    try:
        comparison = evaluator.compare_results(base_results, ttt_results)
        logger.info("✅ Test 5 PASSED: Results compared successfully")
        logger.info(f"   Accuracy Improvement: {comparison['accuracy_improvement']:+.4f}")
        logger.info(f"   ZDR Improvement: {comparison['zero_day_detection_rate_improvement']:+.4f}")
    except Exception as e:
        logger.error(f"❌ Test 5 FAILED: {str(e)}")
        raise

    # Test 6: Full pipeline
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Running Full Pipeline")
    logger.info("=" * 80)
    try:
        # Create fresh evaluator for full pipeline test
        evaluator2 = FairBinaryEvaluator(config, device=device)
        results = evaluator2.run_full_evaluation(
            X_train, y_train_binary,
            X_test, y_test_binary,
            zero_day_mask
        )
        logger.info("✅ Test 6 PASSED: Full pipeline completed successfully")
        logger.info(f"   Final Accuracy Improvement: {results['comparison']['accuracy_improvement']:+.4f}")
        logger.info(f"   Final ZDR Improvement: {results['comparison']['zero_day_detection_rate_improvement']:+.4f}")
    except Exception as e:
        logger.error(f"❌ Test 6 FAILED: {str(e)}")
        raise

    # All tests passed
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("=" * 80)
    logger.info("\nFair Binary Evaluator is working correctly!")
    logger.info("Ready to run on real CICIDS2017 data:")
    logger.info("  python run_fair_evaluation.py --dataset CICIDS2017")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        test_fair_evaluator()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Test failed: {str(e)}", exc_info=True)
        logger.error("\n⚠️  Fix the error above before running on real data")
        raise
