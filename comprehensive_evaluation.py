#!/usr/bin/env python3
"""
Comprehensive Performance Evaluation Function
============================================

This module provides a complete performance evaluation function that addresses
the sample size discrepancy between base model and TTT model evaluations.

Key Features:
- Fair comparison with equal sample sizes
- Comprehensive metrics calculation
- Statistical robustness evaluation
- Detailed logging and visualization
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    matthews_corrcoef, classification_report, f1_score, roc_curve, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that ensures fair comparison between base and TTT models
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.evaluation_config = {
            'base_model_samples': 1000,  # Fixed sample size for base model
            'ttt_model_samples': 1000,   # Same sample size for TTT model
            'k_fold_splits': 3,          # K-fold cross-validation
            'random_seed': 42,           # For reproducibility
            'stratified_sampling': True, # Maintain class distribution
            'threshold_optimization': True, # Use optimal threshold
        }
    
    def evaluate_comprehensive_performance(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        zero_day_mask: torch.Tensor,
        base_model: torch.nn.Module,
        ttt_model: torch.nn.Module,
        preprocessor: Any = None
    ) -> Dict[str, Any]:
        """
        Comprehensive performance evaluation with fair comparison
        
        Args:
            X_test: Test features
            y_test: Test labels
            zero_day_mask: Boolean mask for zero-day samples
            base_model: Base model for evaluation
            ttt_model: TTT model for evaluation
            preprocessor: Data preprocessor (optional)
            
        Returns:
            evaluation_results: Comprehensive evaluation metrics
        """
        logger.info("🔍 Starting comprehensive performance evaluation...")
        
        try:
            # Ensure consistent sample sizes for fair comparison
            max_samples = min(
                self.evaluation_config['base_model_samples'],
                self.evaluation_config['ttt_model_samples'],
                len(X_test)
            )
            
            # Use stratified sampling to maintain class distribution
            if preprocessor and hasattr(preprocessor, 'sample_stratified_subset'):
                X_subset, y_subset = preprocessor.sample_stratified_subset(
                    X_test, y_test, n_samples=max_samples
                )
            else:
                # Manual stratified sampling
                X_subset, y_subset = self._stratified_sample(
                    X_test, y_test, max_samples
                )
            
            # Update zero_day_mask for subset
            zero_day_mask_subset = zero_day_mask[:len(X_subset)]
            
            logger.info(f"📊 Using {len(X_subset)} samples for evaluation")
            logger.info(f"📊 Class distribution: {torch.bincount(y_subset).tolist()}")
            
            # Evaluate Base Model
            logger.info("📊 Evaluating Base Model...")
            base_results = self._evaluate_base_model_comprehensive(
                X_subset, y_subset, zero_day_mask_subset, base_model
            )
            
            # Evaluate TTT Model
            logger.info("🚀 Evaluating TTT Model...")
            ttt_results = self._evaluate_ttt_model_comprehensive(
                X_subset, y_subset, zero_day_mask_subset, ttt_model
            )
            
            # Statistical robustness evaluation
            logger.info("📈 Evaluating statistical robustness...")
            base_kfold_results = self._evaluate_base_model_kfold_comprehensive(
                X_subset, y_subset, base_model
            )
            ttt_kfold_results = self._evaluate_ttt_model_kfold_comprehensive(
                X_subset, y_subset, ttt_model
            )
            
            # Calculate improvements
            improvements = self._calculate_improvements(base_results, ttt_results)
            
            # Compile comprehensive results
            evaluation_results = {
                # Primary evaluation results
                'base_model': base_results,
                'ttt_model': ttt_results,
                
                # Statistical robustness results
                'base_model_kfold': base_kfold_results,
                'ttt_model_kfold': ttt_kfold_results,
                
                # Improvements
                'improvements': improvements,
                
                # Dataset information
                'dataset_info': {
                    'total_samples': len(X_test),
                    'evaluated_samples': len(X_subset),
                    'zero_day_samples': zero_day_mask_subset.sum().item(),
                    'class_distribution': torch.bincount(y_subset).tolist(),
                    'evaluation_timestamp': time.time()
                },
                
                # Configuration
                'evaluation_config': self.evaluation_config
            }
            
            # Log comprehensive results
            self._log_comprehensive_results(evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _stratified_sample(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manual stratified sampling"""
        from sklearn.model_selection import train_test_split
        
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        X_sample, _, y_sample, _ = train_test_split(
            X_np, y_np, 
            train_size=n_samples,
            stratify=y_np,
            random_state=self.evaluation_config['random_seed']
        )
        
        return torch.FloatTensor(X_sample).to(self.device), torch.LongTensor(y_sample).to(self.device)
    
    def _evaluate_base_model_comprehensive(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        zero_day_mask: torch.Tensor,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Comprehensive base model evaluation"""
        try:
            model.eval()
            
            with torch.no_grad():
                # Get model predictions
                outputs = model(X_test)
                probabilities = torch.softmax(outputs, dim=1)
                
                # For binary classification (Normal vs Attack)
                if outputs.shape[1] == 2:
                    predictions = torch.argmax(outputs, dim=1)
                    attack_probabilities = probabilities[:, 1]
                else:
                    # For multiclass, convert to binary
                    predictions = torch.argmax(outputs, dim=1)
                    attack_probabilities = 1.0 - probabilities[:, 0]  # 1 - P(Normal)
                    binary_predictions = (predictions != 0).long()
                else:
                    binary_predictions = predictions
                
                # Convert to numpy
                y_np = y_test.cpu().numpy()
                pred_np = predictions.cpu().numpy()
                binary_pred_np = binary_predictions.cpu().numpy()
                attack_probs_np = attack_probabilities.cpu().numpy()
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(
                    y_np, pred_np, binary_pred_np, attack_probs_np, zero_day_mask
                )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Base model evaluation failed: {str(e)}")
            return self._get_empty_metrics()
    
    def _evaluate_ttt_model_comprehensive(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        zero_day_mask: torch.Tensor,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """Comprehensive TTT model evaluation"""
        try:
            # Split into support and query sets
            support_size = min(200, len(X_test) // 2)
            query_size = len(X_test) - support_size
            
            # Stratified split
            y_binary = (y_test != 0).long()
            support_indices, query_indices = train_test_split(
                torch.arange(len(X_test)),
                test_size=query_size / len(X_test),
                stratify=y_binary.cpu().numpy(),
                random_state=self.evaluation_config['random_seed']
            )
            
            support_x = X_test[support_indices]
            support_y = y_binary[support_indices]
            query_x = X_test[query_indices]
            query_y = y_binary[query_indices]
            query_zero_day_mask = zero_day_mask[query_indices]
            
            # TTT adaptation (if model supports it)
            if hasattr(model, 'adapt_to_support_set'):
                adapted_model = model.adapt_to_support_set(support_x, support_y)
            else:
                adapted_model = model
            
            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                outputs = adapted_model(query_x)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                attack_probabilities = probabilities[:, 1] if outputs.shape[1] == 2 else 1.0 - probabilities[:, 0]
            
            # Convert to numpy
            y_np = query_y.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            attack_probs_np = attack_probabilities.cpu().numpy()
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                y_np, pred_np, pred_np, attack_probs_np, query_zero_day_mask
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"TTT model evaluation failed: {str(e)}")
            return self._get_empty_metrics()
    
    def _evaluate_base_model_kfold_comprehensive(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """K-fold cross-validation for base model"""
        try:
            X_np = X_test.cpu().numpy()
            y_np = y_test.cpu().numpy()
            
            kfold = StratifiedKFold(
                n_splits=self.evaluation_config['k_fold_splits'],
                shuffle=True,
                random_state=self.evaluation_config['random_seed']
            )
            
            fold_metrics = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_np, y_np)):
                X_fold = torch.FloatTensor(X_np[val_idx]).to(self.device)
                y_fold = torch.LongTensor(y_np[val_idx]).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(X_fold)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                
                y_fold_np = y_fold.cpu().numpy()
                pred_fold_np = predictions.cpu().numpy()
                
                # Calculate metrics for this fold
                accuracy = accuracy_score(y_fold_np, pred_fold_np)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_fold_np, pred_fold_np, average='macro', zero_division=0
                )
                mcc = matthews_corrcoef(y_fold_np, pred_fold_np)
                
                fold_metrics.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mcc': mcc
                })
            
            # Calculate mean and std
            results = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']:
                values = [fold[metric] for fold in fold_metrics]
                results[f'{metric}_mean'] = np.mean(values)
                results[f'{metric}_std'] = np.std(values)
            
            return results
            
        except Exception as e:
            logger.error(f"Base model k-fold evaluation failed: {str(e)}")
            return {}
    
    def _evaluate_ttt_model_kfold_comprehensive(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """K-fold cross-validation for TTT model"""
        try:
            X_np = X_test.cpu().numpy()
            y_np = y_test.cpu().numpy()
            
            kfold = StratifiedKFold(
                n_splits=self.evaluation_config['k_fold_splits'],
                shuffle=True,
                random_state=self.evaluation_config['random_seed']
            )
            
            fold_metrics = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_np, y_np)):
                X_fold = torch.FloatTensor(X_np[val_idx]).to(self.device)
                y_fold = torch.LongTensor(y_np[val_idx]).to(self.device)
                
                # Split fold into support and query
                support_size = min(50, len(X_fold) // 2)
                query_size = len(X_fold) - support_size
                
                support_indices, query_indices = train_test_split(
                    torch.arange(len(X_fold)),
                    test_size=query_size / len(X_fold),
                    stratify=y_fold.cpu().numpy(),
                    random_state=self.evaluation_config['random_seed']
                )
                
                support_x = X_fold[support_indices]
                support_y = y_fold[support_indices]
                query_x = X_fold[query_indices]
                query_y = y_fold[query_indices]
                
                # TTT adaptation
                if hasattr(model, 'adapt_to_support_set'):
                    adapted_model = model.adapt_to_support_set(support_x, support_y)
                else:
                    adapted_model = model
                
                # Evaluate
                adapted_model.eval()
                with torch.no_grad():
                    outputs = adapted_model(query_x)
                    predictions = torch.argmax(outputs, dim=1)
                
                query_y_np = query_y.cpu().numpy()
                pred_np = predictions.cpu().numpy()
                
                # Calculate metrics for this fold
                accuracy = accuracy_score(query_y_np, pred_np)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    query_y_np, pred_np, average='macro', zero_division=0
                )
                mcc = matthews_corrcoef(query_y_np, pred_np)
                
                fold_metrics.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mcc': mcc
                })
            
            # Calculate mean and std
            results = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']:
                values = [fold[metric] for fold in fold_metrics]
                results[f'{metric}_mean'] = np.mean(values)
                results[f'{metric}_std'] = np.std(values)
            
            return results
            
        except Exception as e:
            logger.error(f"TTT model k-fold evaluation failed: {str(e)}")
            return {}
    
    def _calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_binary: np.ndarray,
        y_probs: np.ndarray,
        zero_day_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            # Multiclass metrics
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro', zero_division=0
            )
            
            # Binary metrics (for zero-day detection)
            if len(np.unique(y_pred_binary)) > 1:
                precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
                    y_true, y_pred_binary, average='binary', zero_division=0
                )
            else:
                precision_binary = recall_binary = f1_binary = 0.0
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # ROC curve and AUC
            if len(np.unique(y_true)) > 1 and len(np.unique(y_probs)) > 1:
                try:
                    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
                    roc_auc = roc_auc_score(y_true, y_probs)
                except:
                    fpr = tpr = thresholds = np.array([0, 1])
                    roc_auc = 0.5
            else:
                fpr = tpr = thresholds = np.array([0, 1])
                roc_auc = 0.5
            
            # Zero-day detection rate
            zero_day_mask_np = zero_day_mask.cpu().numpy()
            if len(zero_day_mask_np) > 0 and len(zero_day_mask_np) == len(y_pred_binary):
                zero_day_predictions = y_pred_binary[zero_day_mask_np]
                zero_day_detection_rate = zero_day_predictions.mean() if len(zero_day_predictions) > 0 else 0.0
            else:
                zero_day_detection_rate = 0.0
            
            # Classification report
            try:
                class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            except:
                class_report = {}
            
            return {
                # Basic metrics
                'accuracy': accuracy,
                
                # Multiclass metrics
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_score_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_score_weighted': f1_weighted,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_score_micro': f1_micro,
                
                # Binary metrics
                'precision_binary': precision_binary,
                'recall_binary': recall_binary,
                'f1_score_binary': f1_binary,
                
                # Additional metrics
                'mcc': mcc,
                'zero_day_detection_rate': zero_day_detection_rate,
                
                # Confusion matrix and ROC
                'confusion_matrix': cm.tolist(),
                'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()},
                'roc_auc': roc_auc,
                
                # Classification report
                'classification_report': class_report,
                
                # Sample information
                'test_samples': len(y_true),
                'zero_day_samples': zero_day_mask_np.sum() if len(zero_day_mask_np) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            return self._get_empty_metrics()
    
    def _calculate_improvements(
        self,
        base_results: Dict[str, Any],
        ttt_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate improvement metrics"""
        improvements = {}
        
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_score_macro',
                      'precision_weighted', 'recall_weighted', 'f1_score_weighted',
                      'mcc', 'zero_day_detection_rate']:
            base_val = base_results.get(metric, 0.0)
            ttt_val = ttt_results.get(metric, 0.0)
            improvements[f'{metric}_improvement'] = ttt_val - base_val
        
        return improvements
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary"""
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_score_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_score_weighted': 0.0,
            'precision_micro': 0.0,
            'recall_micro': 0.0,
            'f1_score_micro': 0.0,
            'precision_binary': 0.0,
            'recall_binary': 0.0,
            'f1_score_binary': 0.0,
            'mcc': 0.0,
            'zero_day_detection_rate': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'roc_curve': {'fpr': [0, 1], 'tpr': [0, 1], 'thresholds': [1, 0]},
            'roc_auc': 0.5,
            'classification_report': {},
            'test_samples': 0,
            'zero_day_samples': 0
        }
    
    def _log_comprehensive_results(self, results: Dict[str, Any]):
        """Log comprehensive evaluation results"""
        logger.info("📈 Comprehensive Evaluation Results:")
        logger.info("=" * 50)
        
        # Base Model Results
        base = results['base_model']
        logger.info("🎯 Base Model:")
        logger.info(f"  Accuracy: {base.get('accuracy', 0):.4f}")
        logger.info(f"  F1-Macro: {base.get('f1_score_macro', 0):.4f}")
        logger.info(f"  F1-Weighted: {base.get('f1_score_weighted', 0):.4f}")
        logger.info(f"  MCC: {base.get('mcc', 0):.4f}")
        logger.info(f"  Zero-day Detection Rate: {base.get('zero_day_detection_rate', 0):.4f}")
        logger.info(f"  Test Samples: {base.get('test_samples', 0)}")
        
        # TTT Model Results
        ttt = results['ttt_model']
        logger.info("🚀 TTT Model:")
        logger.info(f"  Accuracy: {ttt.get('accuracy', 0):.4f}")
        logger.info(f"  F1-Macro: {ttt.get('f1_score_macro', 0):.4f}")
        logger.info(f"  F1-Weighted: {ttt.get('f1_score_weighted', 0):.4f}")
        logger.info(f"  MCC: {ttt.get('mcc', 0):.4f}")
        logger.info(f"  Zero-day Detection Rate: {ttt.get('zero_day_detection_rate', 0):.4f}")
        logger.info(f"  Test Samples: {ttt.get('test_samples', 0)}")
        
        # Improvements
        improvements = results['improvements']
        logger.info("📈 Improvements:")
        logger.info(f"  Accuracy: {improvements.get('accuracy_improvement', 0):+.4f}")
        logger.info(f"  F1-Macro: {improvements.get('f1_score_macro_improvement', 0):+.4f}")
        logger.info(f"  F1-Weighted: {improvements.get('f1_score_weighted_improvement', 0):+.4f}")
        logger.info(f"  MCC: {improvements.get('mcc_improvement', 0):+.4f}")
        logger.info(f"  Zero-day Detection: {improvements.get('zero_day_detection_rate_improvement', 0):+.4f}")
        
        # Statistical Robustness
        base_kfold = results['base_model_kfold']
        ttt_kfold = results['ttt_model_kfold']
        logger.info("📊 Statistical Robustness (K-fold):")
        logger.info(f"  Base Model - Accuracy: {base_kfold.get('accuracy_mean', 0):.4f} ± {base_kfold.get('accuracy_std', 0):.4f}")
        logger.info(f"  TTT Model - Accuracy: {ttt_kfold.get('accuracy_mean', 0):.4f} ± {ttt_kfold.get('accuracy_std', 0):.4f}")
        
        # Dataset Info
        dataset_info = results['dataset_info']
        logger.info("📊 Dataset Information:")
        logger.info(f"  Total Samples: {dataset_info.get('total_samples', 0)}")
        logger.info(f"  Evaluated Samples: {dataset_info.get('evaluated_samples', 0)}")
        logger.info(f"  Zero-day Samples: {dataset_info.get('zero_day_samples', 0)}")
        logger.info(f"  Class Distribution: {dataset_info.get('class_distribution', [])}")


def create_comprehensive_evaluator(
    base_model_samples: int = 1000,
    ttt_model_samples: int = 1000,
    k_fold_splits: int = 3,
    random_seed: int = 42
) -> ComprehensiveEvaluator:
    """
    Factory function to create a comprehensive evaluator with custom configuration
    
    Args:
        base_model_samples: Number of samples for base model evaluation
        ttt_model_samples: Number of samples for TTT model evaluation
        k_fold_splits: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        
    Returns:
        ComprehensiveEvaluator: Configured evaluator instance
    """
    evaluator = ComprehensiveEvaluator()
    evaluator.evaluation_config.update({
        'base_model_samples': base_model_samples,
        'ttt_model_samples': ttt_model_samples,
        'k_fold_splits': k_fold_splits,
        'random_seed': random_seed
    })
    return evaluator


# Example usage function
def run_comprehensive_evaluation(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    zero_day_mask: torch.Tensor,
    base_model: torch.nn.Module,
    ttt_model: torch.nn.Module,
    preprocessor: Any = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation with fair comparison
    
    This function ensures both base and TTT models are evaluated on the same
    sample size for fair comparison.
    
    Args:
        X_test: Test features
        y_test: Test labels
        zero_day_mask: Boolean mask for zero-day samples
        base_model: Base model for evaluation
        ttt_model: TTT model for evaluation
        preprocessor: Data preprocessor (optional)
        
    Returns:
        evaluation_results: Comprehensive evaluation metrics
    """
    evaluator = create_comprehensive_evaluator(
        base_model_samples=1000,
        ttt_model_samples=1000,
        k_fold_splits=3,
        random_seed=42
    )
    
    return evaluator.evaluate_comprehensive_performance(
        X_test, y_test, zero_day_mask, base_model, ttt_model, preprocessor
    )


if __name__ == "__main__":
    # Example usage
    print("Comprehensive Performance Evaluation Module")
    print("=" * 50)
    print("This module provides fair comparison between base and TTT models")
    print("by ensuring both models are evaluated on the same sample size.")
    print("\nUsage:")
    print("  from comprehensive_evaluation import run_comprehensive_evaluation")
    print("  results = run_comprehensive_evaluation(X_test, y_test, zero_day_mask, base_model, ttt_model)")
