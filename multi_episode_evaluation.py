"""
Multi-Episode Evaluation for Transductive Meta-Learning

This implements philosophically correct multi-episode evaluation that aligns with
the transductive meta-learning paradigm used during training.

Usage:
    python multi_episode_evaluation.py --attack DoS --episodes 10

    Or for comprehensive evaluation of all attacks:
    python run_comprehensive_multi_episode_evaluation.py
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

from config_loader import get_dataset_config
from main import BlockchainFederatedIncentiveSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiEpisodeEvaluator:
    """
    Evaluates a meta-learned model across multiple test episodes.

    This aligns with transductive meta-learning philosophy:
    - Training: Multiple episodes (meta_epochs)
    - Evaluation: Multiple episodes (eval_episodes)
    """

    def __init__(self, config, n_episodes=10, episode_size_target=300):  # REDUCED: 800→300 for bootstrap variance with limited samples
        """
        Initialize multi-episode evaluator.

        Args:
            config: Configuration object
            n_episodes: Number of test episodes to evaluate (default: 10)
            episode_size_target: Target size per episode after sequence creation (default: 800)
        """
        self.config = config
        self.n_episodes = n_episodes
        self.episode_size_target = episode_size_target

        # Calculate pre-sequence size needed
        # Sequence creation typically reduces samples by ~4x
        self.pre_sequence_size = episode_size_target * 4

        logger.info(f"Initialized MultiEpisodeEvaluator:")
        logger.info(f"  Episodes: {n_episodes}")
        logger.info(f"  Target episode size: {episode_size_target} samples")
        logger.info(f"  Pre-sequence sampling: {self.pre_sequence_size} samples")

    def evaluate_single_episode(self, system, episode_idx, test_pool):
        """
        Evaluate on a single episode.

        Args:
            system: CentralizedBlockchainFL system instance
            episode_idx: Episode index for seed
            test_pool: Full test data pool to sample from

        Returns:
            dict: Episode results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"EPISODE {episode_idx + 1}/{self.n_episodes}")
        logger.info(f"{'='*70}\n")

        # Set episode-specific seed for reproducibility
        # Use global SEED constant (42) instead of config.seed
        base_seed = 42
        episode_seed = base_seed + episode_idx
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(episode_seed)

        logger.info(f"🎲 Episode seed: {episode_seed}")

        # Sample episode from full test pool using stratified sampling
        episode_data = self._sample_episode(test_pool, episode_seed, episode_idx)

        logger.info(f"📊 Episode {episode_idx + 1} samples: {len(episode_data['X_test'])} sequences")

        # Store episode data in system for evaluation
        system.preprocessed_data['X_test'] = episode_data.get('X_test')
        system.preprocessed_data['y_test'] = episode_data.get('y_test')
        system.preprocessed_data['y_test_multiclass'] = episode_data.get('y_test_multiclass')
        system.preprocessed_data['test_attack_cat'] = episode_data.get('test_attack_cat')

        # Evaluate base model
        logger.info("🔍 Evaluating Base Model (Transductive Meta-Learning)...")
        base_eval_results = system.evaluate_base_model_only()

        # Perform TTT adaptation
        logger.info("🚀 Performing TTT Adaptation...")
        adapted_model = system.perform_coordinator_side_ttt_adaptation()

        # Evaluate adapted model
        logger.info("📈 Evaluating Adapted Model (TTT Enhanced)...")
        adapted_eval_results = system.evaluate_adapted_model(adapted_model, seed=episode_seed)

        # Ensemble evaluation (if enabled)
        ensemble_eval_results = None
        if self.config.use_ensemble:
            logger.info("🎯 Evaluating Base + TTT Ensemble...")
            from base_ttt_ensemble import BaseTTTEnsemble

            ensemble = BaseTTTEnsemble(
                ensemble_method=self.config.ensemble_method,
                base_weight=self.config.ensemble_base_weight,
                base_confidence_threshold=self.config.ensemble_base_conf_threshold,
                ttt_confidence_threshold=self.config.ensemble_ttt_conf_threshold,
                decision_threshold=self.config.ensemble_decision_threshold
            )

            # Get predictions and probabilities from both models
            base_probs = base_eval_results.get('probabilities', None)
            ttt_probs = adapted_eval_results.get('probabilities', None)

            if base_probs is not None and ttt_probs is not None:
                # Convert from list to numpy array if needed
                if isinstance(base_probs, list):
                    base_probs = np.array(base_probs)
                if isinstance(ttt_probs, list):
                    ttt_probs = np.array(ttt_probs)

                # Get attack class index from zero-day attack type
                attack_class_idx = system.preprocessed_data.get('zero_day_attack_class_idx', 1)

                # Ensemble predictions
                ensemble_predictions, ensemble_probs, ensemble_stats = ensemble.predict(
                    base_probs=base_probs,
                    ttt_probs=ttt_probs,
                    attack_class_idx=attack_class_idx
                )

                # Compute metrics for ensemble
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                y_test = system.preprocessed_data['y_test']

                accuracy = accuracy_score(y_test, ensemble_predictions)
                precision = precision_score(y_test, ensemble_predictions, zero_division=0)
                recall = recall_score(y_test, ensemble_predictions, zero_division=0)
                f1 = f1_score(y_test, ensemble_predictions, zero_division=0)
                cm = confusion_matrix(y_test, ensemble_predictions)

                # Calculate ZDR and FAR
                zero_day_mask = (y_test == 1)
                non_zero_day_mask = (y_test == 0)

                if zero_day_mask.sum() > 0:
                    zdr = ensemble_predictions[zero_day_mask].sum() / zero_day_mask.sum()
                else:
                    zdr = 0.0

                if non_zero_day_mask.sum() > 0:
                    far = ensemble_predictions[non_zero_day_mask].sum() / non_zero_day_mask.sum()
                else:
                    far = 0.0

                ensemble_eval_results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'zero_day_detection_rate': zdr,
                    'far': far,
                    'confusion_matrix': cm.tolist(),
                    'ensemble_stats': ensemble_stats
                }

                logger.info(f"✅ Episode {episode_idx + 1} Ensemble Results:")
                logger.info(f"  Accuracy: {accuracy:.2%}")
                logger.info(f"  ZDR: {zdr:.2%}")
                logger.info(f"  FAR: {far:.2%}")
                logger.info(f"  Method: {self.config.ensemble_method}")
            else:
                logger.warning("⚠️ Probabilities not available for ensemble evaluation")

        logger.info(f"✅ Episode {episode_idx + 1} Base Results:")
        logger.info(f"  Accuracy: {base_eval_results.get('accuracy', 0.0):.2%}")
        logger.info(f"  ZDR: {base_eval_results.get('zero_day_detection_rate', 0.0):.2%}")
        logger.info(f"  FAR: {base_eval_results.get('far', 0.0):.2%}")

        logger.info(f"✅ Episode {episode_idx + 1} TTT Results:")
        logger.info(f"  Accuracy: {adapted_eval_results.get('accuracy', 0.0):.2%}")
        logger.info(f"  ZDR: {adapted_eval_results.get('zero_day_detection_rate', 0.0):.2%}")
        logger.info(f"  FAR: {adapted_eval_results.get('far', 0.0):.2%}")

        # Calculate ROC AUC and AUC-PR if probabilities are available
        base_roc_auc = None
        ttt_roc_auc = None
        base_auc_pr = None
        ttt_auc_pr = None

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            # Get probabilities and true labels
            base_probs = base_eval_results.get('probabilities', None)
            ttt_probs = adapted_eval_results.get('probabilities', None)
            y_test = system.preprocessed_data['y_test']
            
            # Ensure y_test is numpy
            if hasattr(y_test, 'cpu'):
                y_test = y_test.cpu().numpy()

            # Calculate ROC AUC for base model
            if base_probs is not None and len(base_probs) > 0:
                if isinstance(base_probs, list):
                    base_probs = np.array(base_probs)
                
                # Handle multiclass probabilities (N, 2) -> (N,) - take probability of class 1 (Attack)
                if len(base_probs.shape) > 1 and base_probs.shape[1] == 2:
                    base_probs = base_probs[:, 1]
                    
                # Ensure we have binary labels and valid probabilities
                if len(np.unique(y_test)) == 2 and len(base_probs) == len(y_test):
                    base_roc_auc = roc_auc_score(y_test, base_probs)
                    base_auc_pr = average_precision_score(y_test, base_probs)
                    logger.info(f"  Base Model ROC AUC: {base_roc_auc:.4f}")
                    logger.info(f"  Base Model AUC-PR: {base_auc_pr:.4f}")

            # Calculate ROC AUC for TTT model
            if ttt_probs is not None and len(ttt_probs) > 0:
                if isinstance(ttt_probs, list):
                    ttt_probs = np.array(ttt_probs)
                
                # Handle multiclass probabilities (N, 2) -> (N,) - take probability of class 1 (Attack)
                if len(ttt_probs.shape) > 1 and ttt_probs.shape[1] == 2:
                    ttt_probs = ttt_probs[:, 1]
                    
                # Ensure we have binary labels and valid probabilities
                if len(np.unique(y_test)) == 2 and len(ttt_probs) == len(y_test):
                    ttt_roc_auc = roc_auc_score(y_test, ttt_probs)
                    ttt_auc_pr = average_precision_score(y_test, ttt_probs)
                    logger.info(f"  TTT Model ROC AUC: {ttt_roc_auc:.4f}")
                    logger.info(f"  TTT Model AUC-PR: {ttt_auc_pr:.4f}")

        except Exception as e:
            logger.warning(f"⚠️ Could not calculate ROC/PR AUC: {str(e)}")

        # Extract relevant metrics
        episode_result = {
            'episode_id': episode_idx,
            'episode_seed': episode_seed,
            'samples': len(episode_data['X_test']),
            'zero_day_samples': base_eval_results.get('zero_day_only', {}).get('num_samples', 0),
            'non_zero_day_samples': base_eval_results.get('non_zero_day', {}).get('num_samples', 0),

            'base_model': {
                'accuracy': base_eval_results.get('accuracy', 0.0),
                'precision': base_eval_results.get('precision', 0.0),
                'recall': base_eval_results.get('recall', 0.0),
                'f1_score': base_eval_results.get('f1_score', 0.0),
                'zero_day_detection_rate': base_eval_results.get('zero_day_detection_rate', 0.0),
                'far': base_eval_results.get('far', 0.0),
                'confusion_matrix': base_eval_results.get('confusion_matrix', [[0, 0], [0, 0]]),
                'auc_pr': base_eval_results.get('auc_pr', 0.0),
                'roc_auc': base_eval_results.get('roc_auc', 0.0),
            },

            'ttt_model': {
                'accuracy': adapted_eval_results.get('accuracy', 0.0),
                'precision': adapted_eval_results.get('precision', 0.0),
                'recall': adapted_eval_results.get('recall', 0.0),
                'f1_score': adapted_eval_results.get('f1_score', 0.0),
                'zero_day_detection_rate': adapted_eval_results.get('zero_day_detection_rate', 0.0),
                'far': adapted_eval_results.get('far', 0.0),
                'confusion_matrix': adapted_eval_results.get('confusion_matrix', [[0, 0], [0, 0]]),
                'auc_pr': adapted_eval_results.get('auc_pr', 0.0),
                'roc_auc': adapted_eval_results.get('roc_auc', 0.0),
            },

            'improvement': {
                'accuracy': adapted_eval_results.get('accuracy', 0.0) - base_eval_results.get('accuracy', 0.0),
                'zdr': adapted_eval_results.get('zero_day_detection_rate', 0.0) - base_eval_results.get('zero_day_detection_rate', 0.0),
                'far': base_eval_results.get('far', 0.0) - adapted_eval_results.get('far', 0.0),
            }
        }

        # Add ROC AUC and AUC-PR if calculated locally (overriding potential 0.0 defaults)
        if base_roc_auc is not None:
            episode_result['base_model']['roc_auc'] = base_roc_auc
        if base_auc_pr is not None:
            episode_result['base_model']['auc_pr'] = base_auc_pr
        if ttt_roc_auc is not None:
            episode_result['ttt_model']['roc_auc'] = ttt_roc_auc
        if ttt_auc_pr is not None:
            episode_result['ttt_model']['auc_pr'] = ttt_auc_pr
        if base_roc_auc is not None and ttt_roc_auc is not None:
            episode_result['improvement']['roc_auc'] = ttt_roc_auc - base_roc_auc

        # Add ensemble results if available
        if ensemble_eval_results is not None:
            episode_result['ensemble_model'] = {
                'accuracy': ensemble_eval_results.get('accuracy', 0.0),
                'precision': ensemble_eval_results.get('precision', 0.0),
                'recall': ensemble_eval_results.get('recall', 0.0),
                'f1_score': ensemble_eval_results.get('f1_score', 0.0),
                'zero_day_detection_rate': ensemble_eval_results.get('zero_day_detection_rate', 0.0),
                'far': ensemble_eval_results.get('far', 0.0),
                'confusion_matrix': ensemble_eval_results.get('confusion_matrix', [[0, 0], [0, 0]]),
                'ensemble_stats': ensemble_eval_results.get('ensemble_stats', {}),
            }

            episode_result['ensemble_improvement'] = {
                'accuracy': ensemble_eval_results.get('accuracy', 0.0) - base_eval_results.get('accuracy', 0.0),
                'zdr': ensemble_eval_results.get('zero_day_detection_rate', 0.0) - base_eval_results.get('zero_day_detection_rate', 0.0),
                'far': base_eval_results.get('far', 0.0) - ensemble_eval_results.get('far', 0.0),
            }

        return episode_result

    def _sample_episode(self, test_pool, seed, episode_idx):
        """
        Sample a single episode from the test pool using simple random sampling.

        Args:
            test_pool: Dictionary with full test data (already sequences)
            seed: Random seed for reproducibility

        Returns:
            dict: Episode data with sequences
        """
        # Sample from full test pool (data is already in sequence format)
        X_test_full = test_pool['X_test']
        y_test_full = test_pool['y_test']
        y_test_multiclass_full = test_pool['y_test_multiclass']
        test_attack_cat_full = test_pool['test_attack_cat']

        # Simple stratified sampling by binary label (Normal vs Attack)
        # to maintain zero-day proportion
        np.random.seed(seed)

        # Get indices for each class
        zero_day_mask = y_test_full == 1
        non_zero_day_mask = y_test_full == 0

        zero_day_indices = np.where(zero_day_mask)[0]
        non_zero_day_indices = np.where(non_zero_day_mask)[0]

        # Target 25% zero-day proportion
        target_zero_day_samples = int(self.episode_size_target * 0.25)
        target_non_zero_day_samples = self.episode_size_target - target_zero_day_samples

        # Sample from each class WITH REPLACEMENT to ensure variation across episodes
        # This is critical for obtaining meaningful confidence intervals
        if len(zero_day_indices) >= target_zero_day_samples:
            sampled_zero_day_indices = np.random.choice(
                zero_day_indices, target_zero_day_samples, replace=True  # CHANGED: replace=True
            )
        else:
            # If not enough samples, use all available
            sampled_zero_day_indices = zero_day_indices
            if episode_idx == 0:  # Log warning only once
                logger.warning(
                    f"⚠️  Insufficient zero-day samples! Target: {target_zero_day_samples}, "
                    f"Available: {len(zero_day_indices)}. Using all available samples."
                )
            target_non_zero_day_samples = self.episode_size_target - len(sampled_zero_day_indices)

        if len(non_zero_day_indices) >= target_non_zero_day_samples:
            sampled_non_zero_day_indices = np.random.choice(
                non_zero_day_indices, target_non_zero_day_samples, replace=True  # CHANGED: replace=True
            )
        else:
            sampled_non_zero_day_indices = non_zero_day_indices

        # Combine indices
        sampled_indices = np.concatenate([sampled_zero_day_indices, sampled_non_zero_day_indices])
        np.random.shuffle(sampled_indices)

        # Extract episode data
        # Handle test_attack_cat which might be a list
        if isinstance(test_attack_cat_full, list):
            test_attack_cat_sampled = [test_attack_cat_full[i] for i in sampled_indices]
        else:
            test_attack_cat_sampled = test_attack_cat_full[sampled_indices]

        episode_data = {
            'X_test': X_test_full[sampled_indices],
            'y_test': y_test_full[sampled_indices],
            'y_test_multiclass': y_test_multiclass_full[sampled_indices],
            'test_attack_cat': test_attack_cat_sampled,
        }

        return episode_data

    def aggregate_results(self, episode_results):
        """
        Aggregate results across multiple episodes.

        Args:
            episode_results: List of episode result dictionaries

        Returns:
            dict: Aggregated results with mean, std, and confidence intervals
        """
        n_episodes = len(episode_results)

        # Extract metrics
        base_accuracy = [ep['base_model']['accuracy'] for ep in episode_results]
        base_precision = [ep['base_model'].get('precision') or 0.0 for ep in episode_results]
        base_recall = [ep['base_model'].get('recall') or 0.0 for ep in episode_results]
        base_zdr = [ep['base_model']['zero_day_detection_rate'] for ep in episode_results]
        base_far = [ep['base_model']['far'] for ep in episode_results]
        base_f1 = [ep['base_model']['f1_score'] for ep in episode_results]
        base_auc_pr = [ep['base_model'].get('auc_pr') or 0.0 for ep in episode_results]
        base_roc_auc = [ep['base_model'].get('roc_auc') or 0.0 for ep in episode_results]

        ttt_accuracy = [ep['ttt_model']['accuracy'] for ep in episode_results]
        ttt_precision = [ep['ttt_model'].get('precision') or 0.0 for ep in episode_results]
        ttt_recall = [ep['ttt_model'].get('recall') or 0.0 for ep in episode_results]
        ttt_zdr = [ep['ttt_model']['zero_day_detection_rate'] for ep in episode_results]
        ttt_far = [ep['ttt_model']['far'] for ep in episode_results]
        ttt_f1 = [ep['ttt_model']['f1_score'] for ep in episode_results]
        ttt_auc_pr = [ep['ttt_model'].get('auc_pr') or 0.0 for ep in episode_results]
        ttt_roc_auc = [ep['ttt_model'].get('roc_auc') or 0.0 for ep in episode_results]

        improvement_zdr = [ep['improvement']['zdr'] for ep in episode_results]
        improvement_acc = [ep['improvement']['accuracy'] for ep in episode_results]
        improvement_roc_auc = [ep.get('improvement', {}).get('roc_auc') or 0.0 for ep in episode_results]

        # Extract ensemble metrics if available
        has_ensemble = 'ensemble_model' in episode_results[0]
        if has_ensemble:
            ensemble_accuracy = [ep['ensemble_model']['accuracy'] for ep in episode_results]
            ensemble_zdr = [ep['ensemble_model']['zero_day_detection_rate'] for ep in episode_results]
            ensemble_far = [ep['ensemble_model']['far'] for ep in episode_results]
            ensemble_f1 = [ep['ensemble_model']['f1_score'] for ep in episode_results]
            ensemble_improvement_zdr = [ep['ensemble_improvement']['zdr'] for ep in episode_results]
            ensemble_improvement_acc = [ep['ensemble_improvement']['accuracy'] for ep in episode_results]

        # Compute statistics
        def stats(values):
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            ci_95 = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            return {
                'mean': float(mean),
                'std': float(std),
                'ci_95': float(ci_95),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        aggregated = {
            'metadata': {
                'n_episodes': n_episodes,
                'total_samples': sum(ep['samples'] for ep in episode_results),
                'total_zero_day_samples': sum(ep['zero_day_samples'] for ep in episode_results),
                'total_non_zero_day_samples': sum(ep['non_zero_day_samples'] for ep in episode_results),
                'evaluated_at': datetime.now().isoformat(),
            },

            'base_model': {
                'accuracy': stats(base_accuracy),
                'precision': stats(base_precision),
                'recall': stats(base_recall),
                'zero_day_detection_rate': stats(base_zdr),
                'false_alarm_rate': stats(base_far),
                'f1_score': stats(base_f1),
                'auc_pr': stats(base_auc_pr),
                'roc_auc': stats(base_roc_auc),
            },

            'ttt_model': {
                'accuracy': stats(ttt_accuracy),
                'precision': stats(ttt_precision),
                'recall': stats(ttt_recall),
                'zero_day_detection_rate': stats(ttt_zdr),
                'false_alarm_rate': stats(ttt_far),
                'f1_score': stats(ttt_f1),
                'auc_pr': stats(ttt_auc_pr),
                'roc_auc': stats(ttt_roc_auc),
            },

            'improvement': {
                'zero_day_detection_rate': stats(improvement_zdr),
                'accuracy': stats(improvement_acc),
                'roc_auc': stats(improvement_roc_auc),
            },

            'per_episode_results': episode_results
        }

        # Add ensemble statistics if available
        if has_ensemble:
            aggregated['ensemble_model'] = {
                'accuracy': stats(ensemble_accuracy),
                'zero_day_detection_rate': stats(ensemble_zdr),
                'false_alarm_rate': stats(ensemble_far),
                'f1_score': stats(ensemble_f1),
            }
            aggregated['ensemble_improvement'] = {
                'zero_day_detection_rate': stats(ensemble_improvement_zdr),
                'accuracy': stats(ensemble_improvement_acc),
            }

        return aggregated

    def print_summary(self, aggregated):
        """Print summary of aggregated results."""
        logger.info(f"\n{'='*70}")
        logger.info("MULTI-EPISODE EVALUATION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"\nEpisodes Evaluated: {aggregated['metadata']['n_episodes']}")
        logger.info(f"Total Samples: {aggregated['metadata']['total_samples']}")
        logger.info(f"  Zero-Day: {aggregated['metadata']['total_zero_day_samples']}")
        logger.info(f"  Non Zero-Day: {aggregated['metadata']['total_non_zero_day_samples']}")

        logger.info(f"\n{'='*70}")
        logger.info("BASE MODEL PERFORMANCE")
        logger.info(f"{'='*70}")
        bm = aggregated['base_model']
        logger.info(f"Accuracy: {bm['accuracy']['mean']:.2%} ± {bm['accuracy']['ci_95']:.2%} (95% CI)")
        logger.info(f"Zero-Day Detection Rate: {bm['zero_day_detection_rate']['mean']:.2%} ± {bm['zero_day_detection_rate']['ci_95']:.2%}")
        logger.info(f"False Alarm Rate: {bm['false_alarm_rate']['mean']:.2%} ± {bm['false_alarm_rate']['ci_95']:.2%}")
        logger.info(f"F1-Score: {bm['f1_score']['mean']:.2%} ± {bm['f1_score']['ci_95']:.2%}")
        if 'roc_auc' in bm:
            logger.info(f"ROC AUC: {bm['roc_auc']['mean']:.4f} ± {bm['roc_auc']['ci_95']:.4f}")

        logger.info(f"\n{'='*70}")
        logger.info("TTT ADAPTED MODEL PERFORMANCE")
        logger.info(f"{'='*70}")
        tm = aggregated['ttt_model']
        logger.info(f"Accuracy: {tm['accuracy']['mean']:.2%} ± {tm['accuracy']['ci_95']:.2%} (95% CI)")
        logger.info(f"Zero-Day Detection Rate: {tm['zero_day_detection_rate']['mean']:.2%} ± {tm['zero_day_detection_rate']['ci_95']:.2%}")
        logger.info(f"False Alarm Rate: {tm['false_alarm_rate']['mean']:.2%} ± {tm['false_alarm_rate']['ci_95']:.2%}")
        logger.info(f"F1-Score: {tm['f1_score']['mean']:.2%} ± {tm['f1_score']['ci_95']:.2%}")
        if 'roc_auc' in tm:
            logger.info(f"ROC AUC: {tm['roc_auc']['mean']:.4f} ± {tm['roc_auc']['ci_95']:.4f}")

        logger.info(f"\n{'='*70}")
        logger.info("TTT IMPROVEMENT")
        logger.info(f"{'='*70}")
        imp = aggregated['improvement']
        logger.info(f"ZDR Improvement: +{imp['zero_day_detection_rate']['mean']:.2%} ± {imp['zero_day_detection_rate']['ci_95']:.2%}")
        logger.info(f"Accuracy Improvement: +{imp['accuracy']['mean']:.2%} ± {imp['accuracy']['ci_95']:.2%}")

        # Print ensemble results if available
        if 'ensemble_model' in aggregated:
            logger.info(f"\n{'='*70}")
            logger.info("ENSEMBLE MODEL PERFORMANCE (Base + TTT)")
            logger.info(f"{'='*70}")
            em = aggregated['ensemble_model']
            logger.info(f"Accuracy: {em['accuracy']['mean']:.2%} ± {em['accuracy']['ci_95']:.2%} (95% CI)")
            logger.info(f"Zero-Day Detection Rate: {em['zero_day_detection_rate']['mean']:.2%} ± {em['zero_day_detection_rate']['ci_95']:.2%}")
            logger.info(f"False Alarm Rate: {em['false_alarm_rate']['mean']:.2%} ± {em['false_alarm_rate']['ci_95']:.2%}")
            logger.info(f"F1-Score: {em['f1_score']['mean']:.2%} ± {em['f1_score']['ci_95']:.2%}")

            logger.info(f"\n{'='*70}")
            logger.info("ENSEMBLE IMPROVEMENT (vs Base)")
            logger.info(f"{'='*70}")
            eimp = aggregated['ensemble_improvement']
            logger.info(f"ZDR Improvement: +{eimp['zero_day_detection_rate']['mean']:.2%} ± {eimp['zero_day_detection_rate']['ci_95']:.2%}")
            logger.info(f"Accuracy Improvement: +{eimp['accuracy']['mean']:.2%} ± {eimp['accuracy']['ci_95']:.2%}")

            # Compare all three models
            logger.info(f"\n{'='*70}")
            logger.info("MODEL COMPARISON")
            logger.info(f"{'='*70}")
            logger.info(f"{'Model':<20s} {'ZDR':>10s} {'FAR':>10s} {'Accuracy':>10s} {'F1-Score':>10s}")
            logger.info('-' * 70)
            logger.info(
                f"{'Base':<20s} "
                f"{bm['zero_day_detection_rate']['mean']:>9.2%} "
                f"{bm['false_alarm_rate']['mean']:>9.2%} "
                f"{bm['accuracy']['mean']:>9.2%} "
                f"{bm['f1_score']['mean']:>9.2%}"
            )
            logger.info(
                f"{'TTT':<20s} "
                f"{tm['zero_day_detection_rate']['mean']:>9.2%} "
                f"{tm['false_alarm_rate']['mean']:>9.2%} "
                f"{tm['accuracy']['mean']:>9.2%} "
                f"{tm['f1_score']['mean']:>9.2%}"
            )
            logger.info(
                f"{'Ensemble (Base+TTT)':<20s} "
                f"{em['zero_day_detection_rate']['mean']:>9.2%} "
                f"{em['false_alarm_rate']['mean']:>9.2%} "
                f"{em['accuracy']['mean']:>9.2%} "
                f"{em['f1_score']['mean']:>9.2%}"
            )

        logger.info(f"\n{'='*70}")
        logger.info("PER-EPISODE BREAKDOWN")
        logger.info(f"{'='*70}")
        logger.info(f"{'Episode':<10s} {'Base ZDR':>10s} {'TTT ZDR':>10s} {'Improvement':>12s} {'Samples':>8s}")
        logger.info('-' * 70)

        for ep in aggregated['per_episode_results']:
            logger.info(
                f"{ep['episode_id']+1:<10d} "
                f"{ep['base_model']['zero_day_detection_rate']:>9.2%} "
                f"{ep['ttt_model']['zero_day_detection_rate']:>9.2%} "
                f"{ep['improvement']['zdr']:>11.2%} "
                f"{ep['samples']:>8d}"
            )

        logger.info('=' * 70)

    def run_evaluation(self):
        """
        Run multi-episode evaluation.

        Returns:
            dict: Aggregated results
        """
        logger.info(f"\n{'='*70}")
        logger.info("STARTING MULTI-EPISODE EVALUATION")
        logger.info(f"{'='*70}")
        logger.info(f"Zero-Day Attack: {self.config.zero_day_attack}")
        logger.info(f"Episodes: {self.n_episodes}")
        logger.info(f"Target Episode Size: {self.episode_size_target}")

        # Initialize system and train model
        logger.info("\n🚀 Initializing system and training model...")
        system = BlockchainFederatedIncentiveSystem(self.config)

        # Initialize system components
        if not system.initialize_system():
            logger.error("System initialization failed")
            return None

        # Preprocess data
        if not system.preprocess_data():
            logger.error("Data preprocessing failed")
            return None

        # Setup centralized learning
        if not system.setup_centralized_learning():
            logger.error("Centralized learning setup failed")
            return None

        # Train model (this runs the 40 meta-epochs)
        logger.info(f"\n🎓 Training model with {self.config.meta_epochs} meta-epochs...")
        round_results = system.coordinator.train_once()

        if not round_results:
            logger.error("Training failed")
            return None

        logger.info("✅ Training completed")
        logger.info(f"   Training loss: {round_results.get('training_loss', 0.0):.4f}")
        logger.info(f"   Validation accuracy: {round_results.get('validation_accuracy', 0.0):.4f}")

        # Load full test pool for episode sampling
        # Use the already-filtered test set (sequences aligned with labels)
        logger.info("\n📦 Loading full test pool for episode sampling...")
        test_pool = {
            'X_test': system.preprocessed_data['X_test'],  # Sequences with aligned labels
            'y_test': system.preprocessed_data['y_test'],
            'y_test_multiclass': system.preprocessed_data.get('y_test_multiclass'),
            'test_attack_cat': system.preprocessed_data.get('test_attack_cat'),
        }

        logger.info(f"Test pool size: {len(test_pool['X_test'])} samples")

        # Run evaluation across multiple episodes
        episode_results = []

        for episode_idx in range(self.n_episodes):
            try:
                result = self.evaluate_single_episode(system, episode_idx, test_pool)
                episode_results.append(result)
            except Exception as e:
                logger.error(f"❌ Episode {episode_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not episode_results:
            logger.error("❌ No episodes completed successfully!")
            return None

        # Aggregate results
        logger.info("\n📊 Aggregating results across episodes...")
        aggregated = self.aggregate_results(episode_results)

        # Print summary
        self.print_summary(aggregated)

        # Generate visualizations for the last episode (provides visual reference)
        logger.info("\n📊 Generating performance visualizations (from last episode)...")
        try:
            # Set evaluation results from last episode for visualization
            last_episode = episode_results[-1]
            system.evaluation_results = {
                'base_model': last_episode['base_model'],
                'adapted_model': last_episode['ttt_model']
            }

            # Generate plots
            plot_paths = system.generate_performance_visualizations()
            logger.info(f"✅ Generated {len(plot_paths)} plots:")
            for plot_type, plot_path in plot_paths.items():
                if plot_path:
                    logger.info(f"   {plot_type}: {plot_path}")
        except Exception as e:
            logger.warning(f"⚠️ Visualization generation failed: {e}")
            logger.warning("Continuing without plots - results are saved to JSON")

        return aggregated


def main():
    """Main entry point for multi-episode evaluation."""
    parser = argparse.ArgumentParser(description='Multi-Episode Evaluation for Transductive Meta-Learning')
    parser.add_argument('--attack', type=str, default=None,
                       help='Zero-day attack type (default: from config)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate (default: 10)')
    parser.add_argument('--episode-size', type=int, default=800,
                       help='Target episode size after sequences (default: 800)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (default: auto-generates based on attack and episodes)')

    args = parser.parse_args()

    # Load config
    config = get_dataset_config('UNSW')  # Use UNSW dataset

    # Override zero-day attack if specified
    if args.attack:
        config.zero_day_attack = args.attack
        logger.info(f"Overriding zero-day attack to: {args.attack}")

    # Create evaluator
    evaluator = MultiEpisodeEvaluator(
        config=config,
        n_episodes=args.episodes,
        episode_size_target=args.episode_size
    )

    # Run evaluation
    results = evaluator.run_evaluation()

    if results:
        # Convert any tensors to Python types before JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        results = convert_to_json_serializable(results)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = Path("multi_episode_results")
            output_dir.mkdir(exist_ok=True)
            attack_name = config.zero_day_attack.lower()
            output_filename = f"{attack_name}_{args.episodes}_episodes_phase1.json"
            output_path = output_dir / output_filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_path}")

        # Print final summary
        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)
        logger.info(f"Attack: {config.zero_day_attack}")
        logger.info(f"Episodes: {results['metadata']['n_episodes']}")
        logger.info(f"TTT ZDR: {results['ttt_model']['zero_day_detection_rate']['mean']:.2%} "
                   f"± {results['ttt_model']['zero_day_detection_rate']['ci_95']:.2%}")
        logger.info(f"Base ZDR: {results['base_model']['zero_day_detection_rate']['mean']:.2%} "
                   f"± {results['base_model']['zero_day_detection_rate']['ci_95']:.2%}")
        logger.info(f"Improvement: +{results['improvement']['zero_day_detection_rate']['mean']:.2%}")
        logger.info("="*70)
    else:
        logger.error("❌ Evaluation failed!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
