#!/usr/bin/env python3
"""
Example: Fair Evaluation Usage
=============================

This example shows how to use the fair evaluation functions to get
comprehensive performance metrics with equal sample sizes for both
base and TTT models.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_fair_evaluation():
    """
    Example of how to use the fair evaluation functions
    """
    print("🔍 Fair Evaluation Example")
    print("=" * 30)
    
    # Create dummy data for demonstration
    n_samples = 5000
    n_features = 100
    n_classes = 10
    
    # Generate synthetic test data
    X_test = torch.randn(n_samples, n_features)
    y_test = torch.randint(0, n_classes, (n_samples,))
    
    # Create zero-day mask (20% of samples are zero-day)
    zero_day_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    zero_day_mask = torch.zeros(n_samples, dtype=torch.bool)
    zero_day_mask[zero_day_indices] = True
    
    # Create dummy models (replace with your actual models)
    class DummyModel(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            return self.fc(x)
        
        def get_embeddings(self, x):
            return self.fc(x)
        
        def adapt_to_support_set(self, support_x, support_y):
            # Simple adaptation - just return self
            return self
    
    base_model = DummyModel(n_features, n_classes)
    ttt_model = DummyModel(n_features, n_classes)
    
    print(f"📊 Generated {n_samples} test samples with {n_features} features")
    print(f"📊 Zero-day samples: {zero_day_mask.sum().item()}")
    print(f"📊 Class distribution: {torch.bincount(y_test).tolist()}")
    
    # Method 1: Using the comprehensive evaluator directly
    print("\n🔧 Method 1: Using ComprehensiveEvaluator directly")
    try:
        from comprehensive_evaluation import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator()
        results = evaluator.evaluate_comprehensive_performance(
            X_test, y_test, zero_day_mask, base_model, ttt_model
        )
        
        print("✅ Comprehensive evaluation completed")
        print(f"📈 Base Model Accuracy: {results['base_model']['accuracy']:.4f}")
        print(f"📈 TTT Model Accuracy: {results['ttt_model']['accuracy']:.4f}")
        print(f"📈 Improvement: {results['improvements']['accuracy_improvement']:+.4f}")
        
    except ImportError as e:
        print(f"❌ Could not import ComprehensiveEvaluator: {e}")
    
    # Method 2: Using the fair evaluation integration
    print("\n🔧 Method 2: Using fair evaluation integration")
    try:
        from fair_evaluation_integration import evaluate_zero_day_detection_fair
        
        results = evaluate_zero_day_detection_fair(
            X_test, y_test, zero_day_mask, base_model, ttt_model,
            sample_size=1000
        )
        
        print("✅ Fair evaluation completed")
        print(f"📈 Base Model Accuracy: {results['base_model']['accuracy']:.4f}")
        print(f"📈 TTT Model Accuracy: {results['ttt_model']['accuracy']:.4f}")
        print(f"📈 Improvement: {results['improvements']['accuracy_improvement']:+.4f}")
        
    except ImportError as e:
        print(f"❌ Could not import fair evaluation: {e}")
    
    # Method 3: Using the run_comprehensive_evaluation function
    print("\n🔧 Method 3: Using run_comprehensive_evaluation")
    try:
        from comprehensive_evaluation import run_comprehensive_evaluation
        
        results = run_comprehensive_evaluation(
            X_test, y_test, zero_day_mask, base_model, ttt_model
        )
        
        print("✅ Comprehensive evaluation completed")
        print(f"📈 Base Model Accuracy: {results['base_model']['accuracy']:.4f}")
        print(f"📈 TTT Model Accuracy: {results['ttt_model']['accuracy']:.4f}")
        print(f"📈 Improvement: {results['improvements']['accuracy_improvement']:+.4f}")
        
    except ImportError as e:
        print(f"❌ Could not import run_comprehensive_evaluation: {e}")


def show_integration_guide():
    """
    Show how to integrate the fair evaluation into your main.py
    """
    print("\n🔧 Integration Guide for main.py")
    print("=" * 40)
    
    print("1. Add these imports at the top of your main.py:")
    print("   from fair_evaluation_integration import evaluate_zero_day_detection_fair")
    print("   from comprehensive_evaluation import ComprehensiveEvaluator")
    
    print("\n2. Replace your evaluate_zero_day_detection method with:")
    print("""
    def evaluate_zero_day_detection(self) -> Dict:
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
    """)
    
    print("\n3. The results will have the same structure as before, but with fair comparison")
    print("4. Both base and TTT models will be evaluated on the same sample size")


def show_metrics_explanation():
    """
    Show what metrics are available in the comprehensive evaluation
    """
    print("\n📊 Available Metrics")
    print("=" * 20)
    
    metrics = {
        'Basic Metrics': [
            'accuracy - Overall classification accuracy',
            'mcc - Matthews Correlation Coefficient'
        ],
        'Multiclass Metrics': [
            'precision_macro - Macro-averaged precision',
            'recall_macro - Macro-averaged recall',
            'f1_score_macro - Macro-averaged F1-score',
            'precision_weighted - Weighted-averaged precision',
            'recall_weighted - Weighted-averaged recall',
            'f1_score_weighted - Weighted-averaged F1-score',
            'precision_micro - Micro-averaged precision',
            'recall_micro - Micro-averaged recall',
            'f1_score_micro - Micro-averaged F1-score'
        ],
        'Binary Metrics (for zero-day detection)': [
            'precision_binary - Binary precision',
            'recall_binary - Binary recall',
            'f1_score_binary - Binary F1-score'
        ],
        'Specialized Metrics': [
            'zero_day_detection_rate - Rate of detecting zero-day attacks',
            'roc_auc - Area under ROC curve',
            'confusion_matrix - Full confusion matrix',
            'classification_report - Detailed per-class metrics'
        ],
        'Statistical Robustness': [
            'base_model_kfold - K-fold cross-validation for base model',
            'ttt_model_kfold - K-fold cross-validation for TTT model',
            'All metrics with mean and standard deviation'
        ]
    }
    
    for category, metric_list in metrics.items():
        print(f"\n{category}:")
        for metric in metric_list:
            print(f"  • {metric}")


if __name__ == "__main__":
    print("Fair Evaluation Example and Integration Guide")
    print("=" * 50)
    
    # Show the example
    example_fair_evaluation()
    
    # Show integration guide
    show_integration_guide()
    
    # Show metrics explanation
    show_metrics_explanation()
    
    print("\n✅ Example completed!")
    print("Use these functions in your main.py to get fair comparison between base and TTT models.")
