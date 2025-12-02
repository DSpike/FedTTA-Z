"""
Hyperparameter Optimization using Optuna and Wandb
Optimizes both meta-learning and TTT parameters for zero-day detection
"""

import optuna
import wandb
import torch
import numpy as np
import logging
from typing import Dict, Any
import json
import os
import pickle
from pathlib import Path

from config import SystemConfig
from main import BlockchainFederatedIncentiveSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optimizes hyperparameters using Optuna with Wandb integration"""
    
    def __init__(self, 
                 study_name: str = "zero_day_detection_optimization",
                 n_trials: int = 20,
                 direction: str = "maximize",
                 metric: str = "balanced_base_ttt"):
        """
        Initialize the optimizer
        
        Args:
            study_name: Name for Optuna study
            n_trials: Number of optimization trials
            direction: "maximize" or "minimize"
            metric: Primary metric to optimize. Options:
                - "balanced_base_ttt": DEFAULT - Balanced base model + TTT (40% base F1, 30% TTT ZDR, 30% TTT F1)
                  Optimizes for BOTH strong federated few-shot base model AND excellent zero-day detection after TTT
                  Recommended for fair and comprehensive optimization
                - "ttt_zero_day_detection_rate": Optimize for zero-day detection only (TTT-adapted)
                - "ttt_auc_pr": Optimize for AUC-PR (TTT-adapted)
                - "ttt_f1_score": Optimize for F1-score (TTT-adapted)
                - "ttt_accuracy": Optimize for accuracy (TTT-adapted)
                - "multi_objective": BALANCED - Zero-day ZDR (30%), Non-zero-day F1 (35%), Overall F1 (35%)
                  Recommended for IDS that needs to detect BOTH known and unknown attacks (TTT-only)
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric
        
        # Initialize Wandb for the entire study
        # Use offline mode to prevent hanging on network issues
        try:
            wandb.init(
                project="zero-day-detection-optimization",
                name=study_name,
                config={
                    "n_trials": n_trials,
                    "optimization_metric": metric,
                    "direction": direction
                },
                mode="offline",  # Use offline mode to prevent hanging
                reinit=True  # Allow reinitialization if script is run multiple times
            )
            logger.info("✅ Wandb initialized successfully (offline mode)")
        except Exception as e:
            logger.warning(f"⚠️ Wandb initialization failed: {e}. Continuing without wandb logging...")
            # Create a dummy wandb object for compatibility
            class DummyWandb:
                def log(self, *args, **kwargs):
                    pass
                def finish(self, *args, **kwargs):
                    pass
            wandb.log = lambda *args, **kwargs: None
            wandb.finish = lambda *args, **kwargs: None
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # === FEDERATED LEARNING HYPERPARAMETERS ===
        num_clients = trial.suggest_int("num_clients", 3, 10)
        num_rounds = trial.suggest_int("num_rounds", 5, 20)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.5, 10.0, log=True)
        
        # === META-LEARNING HYPERPARAMETERS ===
        meta_lr = trial.suggest_float("meta_learning_rate", 1e-4, 1e-2, log=True)
        meta_epochs = trial.suggest_int("meta_epochs", 3, 30)  # Extended range to find optimal meta-epochs
        k_shot = trial.suggest_int("k_shot", 100, 200)
        n_query = trial.suggest_int("n_query", 10, 20)
        num_meta_tasks = trial.suggest_int("num_meta_tasks", 10, 40)  # Number of meta-tasks per client per round
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
        # FIXED: Equal distribution is now a fixed configuration (not optimized)
        # All classes (Normal + each attack type) get equal proportion in support set
        enforce_equal_support_composition = True
        include_all_attack_types_in_support = True  # Fixed: Sample from all attack types for balanced representation
        
        # === TCN CONFIGURATION ===
        sequence_length = trial.suggest_int("sequence_length", 20, 50)
        sequence_stride = trial.suggest_int("sequence_stride", 10, 20)
        tcn_kernel_size_1 = trial.suggest_int("tcn_kernel_size_1", 2, 6)
        tcn_kernel_size_2 = trial.suggest_int("tcn_kernel_size_2", 2, 6)
        tcn_kernel_size_3 = trial.suggest_int("tcn_kernel_size_3", 2, 6)
        use_residual_connections = trial.suggest_categorical("use_residual_connections", [True, False])
        
        # === TTT HYPERPARAMETERS ===
        ttt_lr = trial.suggest_float("ttt_lr", 1e-4, 2e-3, log=True)
        ttt_steps = trial.suggest_int("ttt_base_steps", 200, 400)
        ttt_batch_size = trial.suggest_categorical("ttt_batch_size", [4, 8, 16, 32])
        ttt_adaptation_query_size = trial.suggest_int("ttt_adaptation_query_size", 1000, 2000)
        
        # === TENT + PSEUDO-LABELS CONFIGURATION ===
        use_pseudo_labels = trial.suggest_categorical("use_pseudo_labels", [True, False])
        pseudo_weight = trial.suggest_float("pseudo_weight", 1.5, 3.5)
        entropy_weight = trial.suggest_float("entropy_weight", 0.5, 1.5)
        pseudo_threshold = trial.suggest_float("pseudo_threshold", 0.85, 0.98)
        pseudo_min_threshold = trial.suggest_float("pseudo_min_threshold", 0.70, 0.85)
        use_teacher = trial.suggest_categorical("use_teacher", [True, False])  # EMA teacher model
        ema_decay = trial.suggest_float("ema_decay", 0.95, 0.999)  # EMA decay rate for teacher model
        pseudo_label_temperature = trial.suggest_float("pseudo_label_temperature", 0.3, 0.8)  # Temperature for sharpening pseudo-label logits
        
        # === TTT TEMPERATURE SCALING ===
        ttt_temperature = trial.suggest_float("ttt_temperature", 1.0, 2.0)
        
        # === ADVANCED TTT TECHNIQUES ===
        use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
        focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.0)  # Focal loss gamma
        focal_alpha = trial.suggest_float("focal_alpha", 0.15, 0.35)  # Focal loss alpha
        
        # === FEDPROX (if enabled) ===
        fedprox_mu = trial.suggest_float("fedprox_mu", 0.001, 0.1, log=True)
        
        return {
            # Federated learning
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "dirichlet_alpha": dirichlet_alpha,
            
            # Meta-learning
            "learning_rate": meta_lr,
            "meta_epochs": meta_epochs,
            "k_shot": k_shot,
            "n_query": n_query,
            "num_meta_tasks": num_meta_tasks,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "enforce_equal_support_composition": enforce_equal_support_composition,
            "include_all_attack_types_in_support": include_all_attack_types_in_support,
            
            # TCN
            "sequence_length": sequence_length,
            "sequence_stride": sequence_stride,
            "tcn_kernel_size_1": tcn_kernel_size_1,
            "tcn_kernel_size_2": tcn_kernel_size_2,
            "tcn_kernel_size_3": tcn_kernel_size_3,
            "use_residual_connections": use_residual_connections,
            
            # TTT
            "ttt_lr": ttt_lr,
            "ttt_base_steps": ttt_steps,
            "ttt_batch_size": ttt_batch_size,
            "ttt_adaptation_query_size": ttt_adaptation_query_size,
            
            # TENT + Pseudo-Labels
            "use_pseudo_labels": use_pseudo_labels,
            "pseudo_weight": pseudo_weight,
            "entropy_weight": entropy_weight,
            "pseudo_threshold": pseudo_threshold,
            "pseudo_min_threshold": pseudo_min_threshold,
            "use_teacher": use_teacher,
            "ema_decay": ema_decay,
            "pseudo_label_temperature": pseudo_label_temperature,
            
            # Temperature Scaling
            "ttt_temperature": ttt_temperature,
            
            # Advanced TTT Techniques
            "use_focal_loss": use_focal_loss,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            
            # FedProx
            "fedprox_mu": fedprox_mu
        }
    
    def update_config(self, config: SystemConfig, hyperparams: Dict[str, Any]) -> SystemConfig:
        """
        Update SystemConfig with suggested hyperparameters
        
        Args:
            config: Original SystemConfig
            hyperparams: Dictionary of hyperparameters to apply
            
        Returns:
            Updated SystemConfig
        """
        # Update federated learning parameters
        config.num_clients = hyperparams["num_clients"]
        config.num_rounds = hyperparams["num_rounds"]
        config.dirichlet_alpha = hyperparams["dirichlet_alpha"]
        
        # Update meta-learning parameters
        config.learning_rate = hyperparams["learning_rate"]
        config.meta_epochs = hyperparams["meta_epochs"]
        config.k_shot = hyperparams["k_shot"]
        config.n_query = hyperparams["n_query"]
        config.num_meta_tasks = hyperparams["num_meta_tasks"]
        config.hidden_dim = hyperparams["hidden_dim"]
        config.embedding_dim = hyperparams["embedding_dim"]
        config.enforce_equal_support_composition = hyperparams["enforce_equal_support_composition"]
        config.include_all_attack_types_in_support = hyperparams.get("include_all_attack_types_in_support", True)
        
        # Update TCN parameters
        config.sequence_length = hyperparams["sequence_length"]
        config.sequence_stride = hyperparams["sequence_stride"]
        # Store TCN kernel sizes in config for model initialization
        config.tcn_kernel_sizes = (
            hyperparams["tcn_kernel_size_1"],
            hyperparams["tcn_kernel_size_2"],
            hyperparams["tcn_kernel_size_3"]
        )
        config.use_residual_connections = hyperparams["use_residual_connections"]
        
        # Update TTT parameters
        config.ttt_lr = hyperparams["ttt_lr"]
        config.ttt_base_steps = hyperparams["ttt_base_steps"]
        config.ttt_batch_size = hyperparams["ttt_batch_size"]
        config.ttt_adaptation_query_size = hyperparams["ttt_adaptation_query_size"]
        
        # Update TENT + Pseudo-Labels parameters
        config.use_pseudo_labels = hyperparams["use_pseudo_labels"]
        config.pseudo_weight = hyperparams["pseudo_weight"]
        config.entropy_weight = hyperparams["entropy_weight"]
        config.pseudo_threshold = hyperparams["pseudo_threshold"]
        config.pseudo_min_threshold = hyperparams["pseudo_min_threshold"]
        config.use_teacher = hyperparams["use_teacher"]
        config.ema_decay = hyperparams["ema_decay"]
        config.pseudo_label_temperature = hyperparams["pseudo_label_temperature"]
        
        # Update temperature scaling
        config.ttt_temperature = hyperparams["ttt_temperature"]
        
        # Update advanced TTT techniques
        config.use_focal_loss = hyperparams["use_focal_loss"]
        config.focal_gamma = hyperparams["focal_gamma"]
        config.focal_alpha = hyperparams["focal_alpha"]
        
        # Update FedProx
        config.fedprox_mu = hyperparams["fedprox_mu"]
        
        return config
    
    def _save_test_set(self, preprocessed_data: Dict[str, Any], trial_number: int):
        """
        Save the test set from preprocessing for reproducibility.
        
        Args:
            preprocessed_data: Dictionary containing preprocessed data
            trial_number: Trial number for unique filename
        """
        try:
            test_set_dir = Path("saved_test_sets")
            test_set_dir.mkdir(exist_ok=True)
            
            test_set_data = {
                'X_test': preprocessed_data.get('X_test'),
                'y_test': preprocessed_data.get('y_test'),
                'y_test_multiclass': preprocessed_data.get('y_test_multiclass'),
                'test_attack_cat': preprocessed_data.get('test_attack_cat'),
                'X_test_original': preprocessed_data.get('X_test_original'),
                'y_test_original': preprocessed_data.get('y_test_original'),
                'test_attack_cat_original': preprocessed_data.get('test_attack_cat_original'),
                'zero_day_indices': preprocessed_data.get('zero_day_indices'),
                'zero_day_attack': preprocessed_data.get('zero_day_attack'),
                'trial_number': trial_number,
            }
            
            # Save test set
            test_set_path = test_set_dir / f"test_set_trial_{trial_number}.pkl"
            with open(test_set_path, 'wb') as f:
                pickle.dump(test_set_data, f)
            
            logger.info(f"💾 Test set saved to: {test_set_path}")
            
            # Also save best trial test set separately for easy access
            if trial_number == 13:  # Best trial from optimization
                best_test_set_path = test_set_dir / "test_set_best_trial.pkl"
                with open(best_test_set_path, 'wb') as f:
                    pickle.dump(test_set_data, f)
                logger.info(f"⭐ Best trial test set saved to: {best_test_set_path}")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to save test set: {e}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to optimize (e.g., zero-day detection rate)
        """
        try:
            # Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {trial.number + 1}/{self.n_trials}")
            logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
            logger.info(f"{'='*80}\n")
            
            # Create config with suggested hyperparameters
            config = SystemConfig()
            config = self.update_config(config, hyperparams)
            
            # Initialize and run system
            system = BlockchainFederatedIncentiveSystem(config)
            
            # Verify FP16 is available for faster optimization (if GPU is available)
            if torch.cuda.is_available():
                device = system.device
                logger.info(f"🚀 GPU detected: {device} - FP16 mixed precision will be used for faster optimization")
                logger.info(f"   Meta-training: FP16 enabled (40-70% faster)")
                logger.info(f"   TTT adaptation: FP16 enabled (40-70% faster)")
            else:
                logger.info("⚠️ CPU mode - FP16 disabled (no GPU available). Optimization will run slower.")
            
            # Initialize system components
            if not system.initialize_system():
                logger.error("❌ System initialization failed")
                return float('-inf') if self.direction == "maximize" else float('inf')
            
            # Preprocess data (skip saved test set during optimization - each trial should create its own)
            if not system.preprocess_data(skip_saved_test_set=True):
                logger.error("❌ Data preprocessing failed")
                return float('-inf') if self.direction == "maximize" else float('inf')
            
            # Save test set for this trial (for reproducibility)
            self._save_test_set(system.preprocessed_data, trial.number)
            
            # Setup federated learning
            if not system.setup_federated_learning():
                logger.error("❌ Federated learning setup failed")
                return float('-inf') if self.direction == "maximize" else float('inf')
            
            # Run federated training rounds
            system.training_history = []
            for round_num in range(1, config.num_rounds + 1):
                logger.info(f"🔄 Federated Round {round_num}/{config.num_rounds}")
                round_results = system.coordinator.run_federated_round(
                    epochs=config.local_epochs
                )
                if not round_results:
                    logger.error(f"❌ Federated round {round_num} failed")
                    return float('-inf') if self.direction == "maximize" else float('inf')
            
            # Evaluate base model
            base_results = system.evaluate_base_model_only()
            system.base_evaluation_results = base_results
            
            # Run TTT adaptation
            adapted_model = system.perform_coordinator_side_ttt_adaptation()
            if adapted_model is None:
                logger.error("❌ TTT adaptation failed")
                return float('-inf') if self.direction == "maximize" else float('inf')
            
            # Evaluate adapted model
            adapted_results = system.evaluate_adapted_model(adapted_model)
            system.adapted_evaluation_results = adapted_results
            
            # Extract metrics
            base_accuracy = base_results.get('accuracy', 0.0)
            base_f1 = base_results.get('f1_score', 0.0)
            base_auc_pr = base_results.get('auc_pr', 0.0)
            base_zdr = base_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)
            
            # Extract non-zero-day metrics
            base_non_zero_day = base_results.get('non_zero_day', {})
            base_non_zero_day_acc = base_non_zero_day.get('accuracy', 0.0)
            base_non_zero_day_f1 = base_non_zero_day.get('f1_score', 0.0)
            
            ttt_accuracy = adapted_results.get('accuracy', 0.0)
            ttt_f1 = adapted_results.get('f1_score', 0.0)
            ttt_auc_pr = adapted_results.get('auc_pr', 0.0)
            ttt_zdr = adapted_results.get('zero_day_only', {}).get('zero_day_detection_rate', 0.0)
            
            # Extract non-zero-day metrics for TTT
            ttt_non_zero_day = adapted_results.get('non_zero_day', {})
            ttt_non_zero_day_acc = ttt_non_zero_day.get('accuracy', 0.0)
            ttt_non_zero_day_f1 = ttt_non_zero_day.get('f1_score', 0.0)
            
            # Calculate improvements
            accuracy_improvement = ttt_accuracy - base_accuracy
            f1_improvement = ttt_f1 - base_f1
            auc_pr_improvement = ttt_auc_pr - base_auc_pr
            zdr_improvement = ttt_zdr - base_zdr
            
            # Calculate non-zero-day improvements
            non_zero_day_acc_improvement = ttt_non_zero_day_acc - base_non_zero_day_acc
            non_zero_day_f1_improvement = ttt_non_zero_day_f1 - base_non_zero_day_f1
            
            # Calculate metric_value for optimization (needed before wandb.log)
            # Default to balanced_base_ttt (includes both base and TTT performance)
            metric_value = 0.0
            
            if self.metric == "balanced_base_ttt":
                # BALANCED BASE + TTT: Optimizes for BOTH strong base model AND excellent TTT performance
                # Formula: 0.40 * base_f1_score + 0.30 * ttt_zero_day_detection_rate + 0.30 * ttt_f1_score
                # This ensures we optimize for:
                # - Strong federated few-shot base model (40% weight)
                # - Excellent zero-day detection after TTT (30% weight)
                # - Overall TTT performance (30% weight)
                base_f1_weight = 0.40  # Base model F1-score (federated few-shot performance)
                ttt_zdr_weight = 0.30  # TTT zero-day detection rate
                ttt_f1_weight = 0.30   # TTT overall F1-score
                
                # Normalize all metrics to 0-1 range (they already are)
                base_f1_score = base_f1  # Already 0.0-1.0
                ttt_zdr_score = ttt_zdr  # Already 0.0-1.0
                ttt_f1_score = ttt_f1    # Already 0.0-1.0
                
                # Combined balanced base + TTT score
                metric_value = (
                    base_f1_weight * base_f1_score +
                    ttt_zdr_weight * ttt_zdr_score +
                    ttt_f1_weight * ttt_f1_score
                )
                
                # Log balanced base + TTT breakdown for debugging
                logger.debug(f"  Balanced Base + TTT objective breakdown:")
                logger.debug(f"    Base F1: {base_f1_score:.4f} (weight: {base_f1_weight:.2f}) → {base_f1_weight * base_f1_score:.4f}")
                logger.debug(f"    TTT ZDR: {ttt_zdr_score:.4f} (weight: {ttt_zdr_weight:.2f}) → {ttt_zdr_weight * ttt_zdr_score:.4f}")
                logger.debug(f"    TTT F1: {ttt_f1_score:.4f} (weight: {ttt_f1_weight:.2f}) → {ttt_f1_weight * ttt_f1_score:.4f}")
                logger.debug(f"    Combined: {metric_value:.4f}")
                
                # Store balanced base + TTT components in trial attributes
                trial.set_user_attr("balanced_base_ttt_score", metric_value)
                trial.set_user_attr("balanced_base_f1_component", base_f1_weight * base_f1_score)
                trial.set_user_attr("balanced_ttt_zdr_component", ttt_zdr_weight * ttt_zdr_score)
                trial.set_user_attr("balanced_ttt_f1_component", ttt_f1_weight * ttt_f1_score)
                trial.set_user_attr("balanced_base_f1_weight", base_f1_weight)
                trial.set_user_attr("balanced_ttt_zdr_weight", ttt_zdr_weight)
                trial.set_user_attr("balanced_ttt_f1_weight", ttt_f1_weight)
            elif self.metric == "multi_objective":
                # BALANCED Multi-objective: Equal importance for zero-day and known attack detection
                # Weights: 30% zero-day detection, 35% non-zero-day F1, 35% overall F1
                # This ensures the IDS works well for BOTH known and unknown attacks
                zdr_weight = 0.30  # Zero-day detection (important but not dominant)
                non_zero_day_f1_weight = 0.35  # Known attack detection (equally important)
                overall_f1_weight = 0.35  # Overall performance (includes both)
                
                # Extract non-zero-day F1 score
                non_zero_day_f1 = ttt_non_zero_day_f1
                
                # Normalize all metrics to 0-1 range (they already are, but explicit for clarity)
                zdr_score = ttt_zdr  # Already 0.0-1.0
                non_zero_day_f1_score = non_zero_day_f1  # Already 0.0-1.0
                overall_f1_score = ttt_f1  # Already 0.0-1.0
                
                # Combined balanced multi-objective score
                # This rewards systems that detect BOTH zero-day AND known attacks well
                metric_value = (
                    zdr_weight * zdr_score +
                    non_zero_day_f1_weight * non_zero_day_f1_score +
                    overall_f1_weight * overall_f1_score
                )
                
                # Log balanced multi-objective breakdown for debugging
                logger.debug(f"  Balanced multi-objective breakdown:")
                logger.debug(f"    Zero-day ZDR: {zdr_score:.4f} (weight: {zdr_weight:.2f}) → {zdr_weight * zdr_score:.4f}")
                logger.debug(f"    Non-zero-day F1: {non_zero_day_f1_score:.4f} (weight: {non_zero_day_f1_weight:.2f}) → {non_zero_day_f1_weight * non_zero_day_f1_score:.4f}")
                logger.debug(f"    Overall F1: {overall_f1_score:.4f} (weight: {overall_f1_weight:.2f}) → {overall_f1_weight * overall_f1_score:.4f}")
                logger.debug(f"    Combined: {metric_value:.4f}")
                
                # Store balanced multi-objective components in trial attributes
                trial.set_user_attr("balanced_multi_objective_score", metric_value)
                trial.set_user_attr("balanced_zdr_component", zdr_weight * zdr_score)
                trial.set_user_attr("balanced_non_zero_day_f1_component", non_zero_day_f1_weight * non_zero_day_f1_score)
                trial.set_user_attr("balanced_overall_f1_component", overall_f1_weight * overall_f1_score)
                trial.set_user_attr("balanced_zdr_weight", zdr_weight)
                trial.set_user_attr("balanced_non_zero_day_f1_weight", non_zero_day_f1_weight)
                trial.set_user_attr("balanced_overall_f1_weight", overall_f1_weight)
            elif self.metric == "ttt_auc_pr":
                metric_value = ttt_auc_pr
            elif self.metric == "ttt_f1_score":
                metric_value = ttt_f1
            elif self.metric == "ttt_accuracy":
                metric_value = ttt_accuracy
            elif self.metric == "ttt_zero_day_detection_rate":
                metric_value = ttt_zdr
            else:
                # Fallback to balanced_base_ttt if unknown metric
                logger.warning(f"⚠️ Unknown metric '{self.metric}'. Falling back to 'balanced_base_ttt'.")
                base_f1_weight = 0.40
                ttt_zdr_weight = 0.30
                ttt_f1_weight = 0.30
                metric_value = (
                    base_f1_weight * base_f1 +
                    ttt_zdr_weight * ttt_zdr +
                    ttt_f1_weight * ttt_f1
                )
            
            # Log metrics and hyperparameters to Wandb
            wandb.log({
                # Hyperparameters for this trial (logged as metrics, not config)
                **{f"hyperparam_{k}": v for k, v in hyperparams.items()},
                
                # Base model metrics
                "base_accuracy": base_accuracy,
                "base_f1_score": base_f1,
                "base_auc_pr": base_auc_pr,
                "base_zero_day_detection_rate": base_zdr,
                "base_non_zero_day_accuracy": base_non_zero_day_acc,
                "base_non_zero_day_f1": base_non_zero_day_f1,
                
                # TTT model metrics
                "ttt_accuracy": ttt_accuracy,
                "ttt_f1_score": ttt_f1,
                "ttt_auc_pr": ttt_auc_pr,
                "ttt_zero_day_detection_rate": ttt_zdr,
                "ttt_non_zero_day_accuracy": ttt_non_zero_day_acc,
                "ttt_non_zero_day_f1": ttt_non_zero_day_f1,
                
                # Improvements
                "accuracy_improvement": accuracy_improvement,
                "f1_improvement": f1_improvement,
                "auc_pr_improvement": auc_pr_improvement,
                "zero_day_detection_improvement": zdr_improvement,
                "non_zero_day_accuracy_improvement": non_zero_day_acc_improvement,
                "non_zero_day_f1_improvement": non_zero_day_f1_improvement,
                
                # Balanced base + TTT metrics (if applicable)
                **({
                    "balanced_base_ttt_score": metric_value,
                    "balanced_base_f1_component": 0.40 * base_f1,
                    "balanced_ttt_zdr_component": 0.30 * ttt_zdr,
                    "balanced_ttt_f1_component": 0.30 * ttt_f1,
                } if self.metric == "balanced_base_ttt" else {}),
                # Balanced multi-objective metrics (if applicable)
                **({
                    "balanced_multi_objective_score": metric_value,
                    "balanced_zdr_component": 0.30 * ttt_zdr,
                    "balanced_non_zero_day_f1_component": 0.35 * ttt_non_zero_day_f1,
                    "balanced_overall_f1_component": 0.35 * ttt_f1,
                } if self.metric == "multi_objective" else {}),
                
                # Trial info
                "trial_number": trial.number,
                "trial_state": "COMPLETE"
            })
            
            # Report to Optuna (metric_value already calculated above)
            
            trial.set_user_attr("base_accuracy", base_accuracy)
            trial.set_user_attr("base_f1", base_f1)
            trial.set_user_attr("base_auc_pr", base_auc_pr)
            trial.set_user_attr("base_zdr", base_zdr)
            trial.set_user_attr("base_non_zero_day_acc", base_non_zero_day_acc)
            trial.set_user_attr("base_non_zero_day_f1", base_non_zero_day_f1)
            trial.set_user_attr("ttt_accuracy", ttt_accuracy)
            trial.set_user_attr("ttt_f1", ttt_f1)
            trial.set_user_attr("ttt_auc_pr", ttt_auc_pr)
            trial.set_user_attr("ttt_zdr", ttt_zdr)
            trial.set_user_attr("ttt_non_zero_day_acc", ttt_non_zero_day_acc)
            trial.set_user_attr("ttt_non_zero_day_f1", ttt_non_zero_day_f1)
            trial.set_user_attr("accuracy_improvement", accuracy_improvement)
            trial.set_user_attr("f1_improvement", f1_improvement)
            trial.set_user_attr("auc_pr_improvement", auc_pr_improvement)
            trial.set_user_attr("zdr_improvement", zdr_improvement)
            trial.set_user_attr("non_zero_day_acc_improvement", non_zero_day_acc_improvement)
            trial.set_user_attr("non_zero_day_f1_improvement", non_zero_day_f1_improvement)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {trial.number + 1} Results:")
            logger.info(f"  Base Model: Accuracy={base_accuracy:.4f}, F1={base_f1:.4f}, AUC-PR={base_auc_pr:.4f}, ZDR={base_zdr:.4f}")
            logger.info(f"    Non-Zero-Day: Acc={base_non_zero_day_acc:.4f}, F1={base_non_zero_day_f1:.4f}")
            logger.info(f"  TTT Model: Accuracy={ttt_accuracy:.4f}, F1={ttt_f1:.4f}, AUC-PR={ttt_auc_pr:.4f}, ZDR={ttt_zdr:.4f}")
            logger.info(f"    Non-Zero-Day: Acc={ttt_non_zero_day_acc:.4f}, F1={ttt_non_zero_day_f1:.4f}")
            logger.info(f"  Improvements:")
            logger.info(f"    Overall: Acc={accuracy_improvement:+.4f}, F1={f1_improvement:+.4f}, AUC-PR={auc_pr_improvement:+.4f}, ZDR={zdr_improvement:+.4f}")
            logger.info(f"    Non-Zero-Day: Acc={non_zero_day_acc_improvement:+.4f}, F1={non_zero_day_f1_improvement:+.4f}")
            
            # Enhanced logging for balanced base + TTT optimization
            if self.metric == "balanced_base_ttt":
                base_f1_weight = 0.40
                ttt_zdr_weight = 0.30
                ttt_f1_weight = 0.30
                logger.info(f"  🎯 Balanced Base + TTT Score ({self.metric}):")
                logger.info(f"    Components (optimizes for BOTH strong base model AND excellent TTT performance):")
                logger.info(f"      Base F1: {base_f1:.4f} × {base_f1_weight:.2f} = {base_f1_weight * base_f1:.4f}")
                logger.info(f"      TTT ZDR: {ttt_zdr:.4f} × {ttt_zdr_weight:.2f} = {ttt_zdr_weight * ttt_zdr:.4f}")
                logger.info(f"      TTT F1: {ttt_f1:.4f} × {ttt_f1_weight:.2f} = {ttt_f1_weight * ttt_f1:.4f}")
                logger.info(f"    Combined Score: {metric_value:.4f}")
                logger.info(f"    📊 Balance: 40% base model + 30% TTT zero-day + 30% TTT overall = 100%")
            # Enhanced logging for balanced multi-objective optimization
            elif self.metric == "multi_objective":
                zdr_weight = 0.30
                non_zero_day_f1_weight = 0.35
                overall_f1_weight = 0.35
                logger.info(f"  🎯 Balanced Multi-Objective Score ({self.metric}):")
                logger.info(f"    Components (balanced for both zero-day AND known attacks):")
                logger.info(f"      Zero-day ZDR: {ttt_zdr:.4f} × {zdr_weight:.2f} = {zdr_weight * ttt_zdr:.4f}")
                logger.info(f"      Non-zero-day F1: {ttt_non_zero_day_f1:.4f} × {non_zero_day_f1_weight:.2f} = {non_zero_day_f1_weight * ttt_non_zero_day_f1:.4f}")
                logger.info(f"      Overall F1: {ttt_f1:.4f} × {overall_f1_weight:.2f} = {overall_f1_weight * ttt_f1:.4f}")
                logger.info(f"    Combined Score: {metric_value:.4f}")
                logger.info(f"    📊 Balance: 30% zero-day + 35% known attacks + 35% overall = 100%")
            else:
                logger.info(f"  Optimizing metric ({self.metric}): {metric_value:.4f}")
            logger.info(f"{'='*80}\n")
            
            # Cleanup
            del system
            torch.cuda.empty_cache()
            
            return metric_value
            
        except Exception as e:
            logger.error(f"❌ Trial {trial.number + 1} failed: {str(e)}")
            wandb.log({
                "trial_number": trial.number,
                "trial_state": "FAILED",
                "error": str(e)
            })
            return float('-inf') if self.direction == "maximize" else float('inf')
    
    def optimize(self):
        """Run the optimization study"""
        logger.info(f"🚀 Starting Optuna optimization study: {self.study_name}")
        logger.info(f"   Trials: {self.n_trials}")
        logger.info(f"   Direction: {self.direction}")
        logger.info(f"   Metric: {self.metric}")
        logger.info(f"   Wandb Project: zero-day-detection-optimization\n")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Log best trial to Wandb
        best_trial = self.study.best_trial
        wandb.log({
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_user_attrs": best_trial.user_attrs
        })
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Optimization Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Best Trial: {best_trial.number}")
        logger.info(f"Best Value ({self.metric}): {best_trial.value:.4f}")
        logger.info(f"\nBest Hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nBest Trial Metrics:")
        for key, value in best_trial.user_attrs.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"{'='*80}\n")
        
        # Save best hyperparameters
        best_params_path = Path("best_hyperparameters.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                "best_trial_number": best_trial.number,
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_user_attrs": best_trial.user_attrs
            }, f, indent=2)
        
        logger.info(f"💾 Best hyperparameters saved to: {best_params_path}")
        
        # Finish Wandb run
        wandb.finish()
        
        return best_trial


def main():
    """Main entry point for hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters using Optuna and Wandb")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--study_name", type=str, default="zero_day_detection_optimization", 
                       help="Name for Optuna study")
    parser.add_argument("--metric", type=str, default="balanced_base_ttt",
                       choices=["balanced_base_ttt", "ttt_zero_day_detection_rate", "ttt_auc_pr", "ttt_f1_score", "ttt_accuracy", "multi_objective"],
                       help="Primary metric to optimize. 'balanced_base_ttt' (default) optimizes for both base model (40% base F1) and TTT performance (30% ZDR, 30% TTT F1). 'multi_objective' balances TTT-only metrics.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"],
                       help="Optimization direction")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name=args.study_name,
        n_trials=args.n_trials,
        direction=args.direction,
        metric=args.metric
    )
    
    # Run optimization
    best_trial = optimizer.optimize()
    
    return best_trial


if __name__ == "__main__":
    main()

