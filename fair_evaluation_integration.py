#!/usr/bin/env python3
"""
Fair Evaluation Integration
===========================

This module provides a simple integration function to replace the current
evaluation in main.py with fair comparison between base and TTT models.

Key Features:
- Fixes the sample size discrepancy issue
- Maintains the same interface as current evaluation
- Provides comprehensive metrics
- Ensures reproducible results
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional
from comprehensive_evaluation import ComprehensiveEvaluator

logger = logging.getLogger(__name__)

def evaluate_zero_day_detection_fair(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    zero_day_mask: torch.Tensor,
    base_model: torch.nn.Module,
    ttt_model: torch.nn.Module,
    preprocessor: Any = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Fair evaluation function that ensures equal sample sizes for base and TTT models
    
    This function replaces the current evaluation in main.py and fixes the
    sample size discrepancy issue.
    
    Args:
        X_test: Test features
        y_test: Test labels
        zero_day_mask: Boolean mask for zero-day samples
        base_model: Base model for evaluation
        ttt_model: TTT model for evaluation
        preprocessor: Data preprocessor (optional)
        sample_size: Number of samples to use for evaluation (default: 1000)
        
    Returns:
        evaluation_results: Fair evaluation metrics with equal sample sizes
    """
    logger.info("🔍 Starting fair zero-day detection evaluation...")
    logger.info(f"📊 Using {sample_size} samples for both base and TTT models")
    
    try:
        # Create comprehensive evaluator with fair configuration
        evaluator = ComprehensiveEvaluator()
        evaluator.evaluation_config.update({
            'base_model_samples': sample_size,
            'ttt_model_samples': sample_size,
            'k_fold_splits': 3,
            'random_seed': 42,
            'stratified_sampling': True,
            'threshold_optimization': True
        })
        
        # Run comprehensive evaluation
        results = evaluator.evaluate_comprehensive_performance(
            X_test, y_test, zero_day_mask, base_model, ttt_model, preprocessor
        )
        
        # Log the key improvement
        base_accuracy = results['base_model'].get('accuracy', 0)
        ttt_accuracy = results['ttt_model'].get('accuracy', 0)
        accuracy_improvement = ttt_accuracy - base_accuracy
        
        base_f1 = results['base_model'].get('f1_score_macro', 0)
        ttt_f1 = results['ttt_model'].get('f1_score_macro', 0)
        f1_improvement = ttt_f1 - base_f1
        
        logger.info("📈 Fair Evaluation Results:")
        logger.info(f"  Base Model - Accuracy: {base_accuracy:.4f}, F1-Macro: {base_f1:.4f}")
        logger.info(f"  TTT Model - Accuracy: {ttt_accuracy:.4f}, F1-Macro: {ttt_f1:.4f}")
        logger.info(f"  Improvement - Accuracy: {accuracy_improvement:+.4f}, F1-Macro: {f1_improvement:+.4f}")
        
        # Log sample size information
        base_samples = results['base_model'].get('test_samples', 0)
        ttt_samples = results['ttt_model'].get('test_samples', 0)
        logger.info(f"  Sample Sizes - Base: {base_samples}, TTT: {ttt_samples}")
        
        if base_samples != ttt_samples:
            logger.warning(f"⚠️ Sample size mismatch detected: Base={base_samples}, TTT={ttt_samples}")
        else:
            logger.info("✅ Fair comparison achieved - both models evaluated on same sample size")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Fair evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def integrate_fair_evaluation_in_main():
    """
    Integration guide for replacing current evaluation in main.py
    
    To use this in your main.py file:
    
    1. Import this module:
       from fair_evaluation_integration import evaluate_zero_day_detection_fair
    
    2. Replace the current evaluate_zero_day_detection method with:
       def evaluate_zero_day_detection(self) -> Dict:
           return evaluate_zero_day_detection_fair(
               X_test=self.preprocessed_data['X_test'],
               y_test=self.preprocessed_data['y_test'],
               zero_day_mask=zero_day_mask,
               base_model=self.coordinator.model,
               ttt_model=self.ttt_model,
               preprocessor=self.preprocessor,
               sample_size=1000  # Adjust as needed
           )
    
    3. The results will have the same structure as before, but with fair comparison
    """
    print("Fair Evaluation Integration Guide")
    print("=" * 40)
    print("This module fixes the sample size discrepancy between base and TTT models.")
    print("Both models will be evaluated on the same number of samples for fair comparison.")
    print("\nIntegration steps:")
    print("1. Import: from fair_evaluation_integration import evaluate_zero_day_detection_fair")
    print("2. Replace your current evaluation method with the fair version")
    print("3. Both models will use the same sample size (default: 1000 samples)")


def create_fair_evaluation_patch():
    """
    Create a patch file that can be applied to main.py to fix the evaluation issue
    """
    patch_content = '''
# Add this import at the top of main.py
from fair_evaluation_integration import evaluate_zero_day_detection_fair

# Replace the evaluate_zero_day_detection method with this:
def evaluate_zero_day_detection(self) -> Dict:
    """
    Fair evaluation that ensures equal sample sizes for base and TTT models
    """
    try:
        logger.info("🔍 Starting fair zero-day detection evaluation...")
        
        if not hasattr(self, 'preprocessed_data') or not self.preprocessed_data:
            logger.error("No preprocessed data available for evaluation")
            return {}
        
        # Get test data
        X_test = self.preprocessed_data['X_test']
        y_test = self.preprocessed_data['y_test']
        zero_day_indices = self.preprocessed_data.get('zero_day_indices', [])
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create zero-day mask
        if len(zero_day_indices) == 0:
            zero_day_indices = list(range(len(y_test)))
        
        zero_day_mask = torch.zeros(len(y_test), dtype=torch.bool)
        zero_day_mask[zero_day_indices] = True
        
        # Use fair evaluation
        return evaluate_zero_day_detection_fair(
            X_test=X_test_tensor,
            y_test=y_test_tensor,
            zero_day_mask=zero_day_mask,
            base_model=self.coordinator.model,
            ttt_model=self.ttt_model,
            preprocessor=self.preprocessor,
            sample_size=1000  # Adjust as needed
        )
        
    except Exception as e:
        logger.error(f"❌ Fair evaluation failed: {str(e)}")
        return {}
'''
    
    with open('fair_evaluation_patch.py', 'w') as f:
        f.write(patch_content)
    
    print("✅ Fair evaluation patch created: fair_evaluation_patch.py")
    print("Copy the content and replace your evaluate_zero_day_detection method in main.py")


if __name__ == "__main__":
    print("Fair Evaluation Integration Module")
    print("=" * 40)
    print("This module provides fair comparison between base and TTT models")
    print("by ensuring both models are evaluated on the same sample size.")
    print("\nKey Benefits:")
    print("✅ Fixes sample size discrepancy")
    print("✅ Ensures fair comparison")
    print("✅ Maintains same interface")
    print("✅ Provides comprehensive metrics")
    print("\nTo integrate:")
    integrate_fair_evaluation_in_main()
    print("\nTo create patch:")
    create_fair_evaluation_patch()
