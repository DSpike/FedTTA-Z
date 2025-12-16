"""
Hyperparameter Optimization using Optuna and Wandb for CICIDS2017 Dataset
Optimizes both meta-learning and TTT parameters for zero-day detection
This is a specialized version for CICIDS2017 dataset
"""

import optuna
from optuna.exceptions import TrialPruned
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


class HyperparameterOptimizerCICIDS:
    """Optimizes hyperparameters using Optuna with Wandb integration for CICIDS2017 dataset"""
    
    def __init__(self, 
                 study_name: str = "cicids_zero_day_detection_optimization",
                 n_trials: int = 20,
                 direction: str = "maximize",
                 metric: str = "balanced_base_ttt",
                 zero_day_attack: str = "PortScan"):
        """
        Initialize the optimizer for CICIDS2017 dataset
        
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
                - "improved_multi_objective": OPTIMAL - Base F1 (25%), TTT ZDR (25%), TTT Non-zero-day F1 (25%), TTT Overall F1 (25%)
                  Balanced metric that includes base model performance for fair comparison
            zero_day_attack: Zero-day attack type for CICIDS2017 (default: "PortScan")
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric
        self.zero_day_attack = zero_day_attack
        
        # Initialize Wandb for the entire study
        # Use offline mode to prevent hanging on network issues
        try:
            wandb.init(
                project="cicids-zero-day-detection-optimization",
                name=study_name,
                config={
                    "n_trials": n_trials,
                    "optimization_metric": metric,
                    "direction": direction,
                    "dataset": "CICIDS2017",
                    "zero_day_attack": zero_day_attack
                },
                mode="offline",  # Use offline mode to prevent hanging
                reinit=True  # Allow reinitialization if script is run multiple times
            )
            logger.info("✅ Wandb initialized successfully (offline mode) for CICIDS2017")
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
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial (same search space as UNSW version)
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # === FEDERATED LEARNING HYPERPARAMETERS ===
        # CONSTRAINTS: Limit num_clients to ensure sample availability
        # With fewer clients, each gets more samples (better chance to meet requirements)
        num_clients = trial.suggest_int("num_clients", 3, 5)  # Reduced max from 10 to 5
        num_rounds = trial.suggest_int("num_rounds", 5, 20)
        # Constrain dirichlet_alpha: Lower alpha = more heterogeneity = some clients get very few samples
        # Higher alpha = more uniform = better sample distribution
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 2.0, 5.0, log=False)  # Constrained to moderate values
        
        # === META-LEARNING HYPERPARAMETERS ===
        meta_lr = trial.suggest_float("meta_learning_rate", 1e-4, 1e-2, log=True)
        meta_epochs = trial.suggest_int("meta_epochs", 20, 35)  # Optimal range: 20-30 epochs for best convergence without overfitting
        
        # CONSTRAINTS: Limit k_shot to ensure clients can participate
        # FEW-SHOT LEARNING: Samples can be REUSED across tasks (we don't need k_shot × num_meta_tasks unique samples!)
        # Minimum requirement: k_shot * n_way + n_query (for ONE task)
        # Practical minimum: 3 * k_shot * n_way + n_query (for diverse tasks with sample reuse)
        # With CICIDS dataset and num_clients=5, alpha=3.0: average client gets ~20k samples
        # So practical min = 3 * k_shot * 2 + n_query ≈ 6 * k_shot + 15
        # Conservative: 6 * k_shot < 3,000 → k_shot < 500 (but we limit to 100 for diversity)
        max_k_shot = 100  # Reasonable maximum for diverse few-shot tasks
        max_num_meta_tasks = 50  # Can be higher since samples are reused
        
        k_shot = trial.suggest_int("k_shot", 30, max_k_shot)
        n_query = trial.suggest_int("n_query", 10, 20)
        num_meta_tasks = trial.suggest_int("num_meta_tasks", 20, max_num_meta_tasks)  # Number of meta-tasks per client per round
        
        # CONSTRAINT: Ensure clients have enough samples for diverse tasks
        # With sample reuse, practical minimum is ~3 * k_shot * 2 + n_query
        practical_min_samples = 3 * k_shot * 2 + n_query  # ~6 * k_shot + n_query
        min_samples_per_client_estimate = 3000  # Conservative estimate for CICIDS with 5 clients
        
        if practical_min_samples > min_samples_per_client_estimate:
            # k_shot too high - prune this trial early
            logger.warning(f"⚠️ k_shot={k_shot} requires {practical_min_samples} samples per client (too high), pruning trial")
            raise TrialPruned(f"k_shot={k_shot} requires too many samples ({practical_min_samples} > {min_samples_per_client_estimate})")
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
        # FIXED: Equal distribution is now a fixed configuration (not optimized)
        # Multi-attack support sets (3-5 attacks per task) with balanced distribution
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
        ttt_steps = trial.suggest_int("ttt_base_steps", 150, 300)  # Updated range: 150-300 (optimal around 200)
        ttt_batch_size = trial.suggest_categorical("ttt_batch_size", [4, 8, 16, 32, 64, 128])
        ttt_adaptation_query_size = trial.suggest_int("ttt_adaptation_query_size", 1000, 2000)
        
        # === NEW TTT PARAMETERS (L2 Regularization & Confidence Rejection) ===
        ttt_l2_reg_weight = trial.suggest_float("ttt_l2_reg_weight", 0.001, 0.1, log=True)  # L2 regularization weight
        confidence_rejection_threshold = trial.suggest_float("confidence_rejection_threshold", 0.5, 0.9)  # Confidence threshold for rejection
        
        # === TRANSDUCTIVE OPTIMIZATION (within-task) ===
        transductive_steps = trial.suggest_int("transductive_steps", 10, 20)  # Prototype refinement steps per task
        
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
            "ttt_l2_reg_weight": ttt_l2_reg_weight,
            "confidence_rejection_threshold": confidence_rejection_threshold,
            
            # Transductive Optimization
            "transductive_steps": transductive_steps,
            
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
        Update SystemConfig with suggested hyperparameters and CICIDS2017 dataset settings
        
        Args:
            config: Original SystemConfig
            hyperparams: Dictionary of hyperparameters to apply
            
        Returns:
            Updated SystemConfig for CICIDS2017
        """
        # Set dataset to CICIDS2017 (ensure CICIDS2017 is being used)
        config.data_path = "CICIDS2017_train.csv"
        config.test_path = "CICIDS2017_test.csv"
        config.zero_day_attack = self.zero_day_attack
        
        # IMPORTANT: Validation and Test Set Composition
        # Validation: 60% Normal, 40% Known attacks (equal per attack type)
        # Test: 60% Normal, 30% Known attacks, 10% Zero-day
        # This composition is handled automatically in main.py's preprocess_data() method
        logger.info(f"📊 Dataset: CICIDS2017")
        logger.info(f"   Validation set target: 60% Normal, 40% Known attacks (balanced)")
        logger.info(f"   Test set target: 60% Normal, 30% Known attacks, 10% Zero-day")
        
        # Ensure CICIDS2017 attack types are used (not UNSW-NB15)
        # The attack_types dictionary in config should already be set for CICIDS2017
        
        # Update input_dim for CICIDS2017 (43 features after feature selection)
        config.input_dim = 43
        
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
        config.ttt_l2_reg_weight = hyperparams.get("ttt_l2_reg_weight", 0.01)
        config.confidence_rejection_threshold = hyperparams.get("confidence_rejection_threshold", 0.7)
        
        # Update transductive optimization parameters
        config.transductive_steps = hyperparams["transductive_steps"]
        
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
                'dataset': 'CICIDS2017'
            }
            
            # Save test set
            test_set_path = test_set_dir / f"cicids_test_set_trial_{trial_number}.pkl"
            with open(test_set_path, 'wb') as f:
                pickle.dump(test_set_data, f)
            
            logger.info(f"💾 CICIDS2017 test set saved to: {test_set_path}")
            
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
            logger.info(f"CICIDS2017 Optimization - Trial {trial.number + 1}/{self.n_trials}")
            logger.info(f"Zero-Day Attack: {self.zero_day_attack}")
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
            
            # PROGRESSIVE EPOCHS: Increase max_meta_epochs based on trial number
            # Early trials use fewer meta-epochs, later trials use more for better convergence
            trial_number = trial.number
            if trial_number < 30:
                max_meta_epochs = 50
            elif trial_number < 70:
                max_meta_epochs = 100
            else:
                max_meta_epochs = 200
            
            # Apply progressive epochs to meta_epochs (FIXED: was incorrectly limiting local_epochs)
            original_meta_epochs = config.meta_epochs
            config.meta_epochs = min(config.meta_epochs, max_meta_epochs)
            
            logger.info(f"📊 Progressive Training Strategy:")
            logger.info(f"   Trial {trial_number + 1}: Original meta_epochs={original_meta_epochs}, Limited to {config.meta_epochs}")
            logger.info(f"   Early stopping: Enabled (patience=10 rounds, min_rounds=5)")
            
            # Run federated training rounds with early stopping and pruning
            system.training_history = []
            min_client_participation = 0.4  # Require at least 40% of clients to participate
            rounds_with_low_participation = 0
            max_low_participation_rounds = 2  # Allow up to 2 rounds with low participation
            
            # Early stopping variables (FIXED: adaptive threshold based on num_rounds)
            best_validation_metric = float('-inf') if self.direction == "maximize" else float('inf')
            no_improvement_counter = 0
            early_stopping_patience = 10
            min_rounds_for_early_stopping = max(5, config.num_rounds // 3)  # Adaptive: 33% of total rounds or minimum 5
            
            for round_num in range(1, config.num_rounds + 1):
                logger.info(f"🔄 Federated Round {round_num}/{config.num_rounds}")
                
                # Progressive epochs already applied to config.meta_epochs above
                # Use standard local_epochs for federated rounds
                round_results = system.coordinator.run_federated_round(
                    epochs=config.local_epochs
                )
                if not round_results:
                    logger.error(f"❌ Federated round {round_num} failed")
                    return float('-inf') if self.direction == "maximize" else float('inf')
                
                # PRUNING: Report intermediate value and check if trial should be pruned
                # Use validation accuracy from round results if available, otherwise use training metrics
                if 'validation_accuracy' in round_results:
                    intermediate_value = round_results['validation_accuracy']
                elif 'accuracy' in round_results:
                    intermediate_value = round_results['accuracy']
                elif len(system.training_history) > 0:
                    # Use latest training accuracy as proxy
                    intermediate_value = system.training_history[-1].get('accuracy', 0.0)
                else:
                    intermediate_value = 0.0
                
                # Report to Optuna for pruning decision
                trial.report(intermediate_value, step=round_num)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    logger.warning(f"✂️  Trial {trial_number + 1} pruned at round {round_num} (intermediate_value={intermediate_value:.4f})")
                    raise optuna.TrialPruned()
                
                # VALIDATION: Check client participation
                if 'client_updates' in round_results:
                    client_updates = round_results.get('client_updates', [])
                    if isinstance(client_updates, (list, tuple)):
                        active_clients = sum(1 for update in client_updates if getattr(update, 'sample_count', 0) > 0)
                        total_clients = len(client_updates)
                        participation_ratio = active_clients / total_clients if total_clients > 0 else 0.0
                        
                        if participation_ratio < min_client_participation:
                            rounds_with_low_participation += 1
                            logger.warning(
                                f"⚠️ Round {round_num}: Low client participation ({active_clients}/{total_clients} = {participation_ratio:.1%})"
                            )
                            
                            if rounds_with_low_participation > max_low_participation_rounds:
                                logger.error(
                                    f"❌ Too many rounds with low client participation ({rounds_with_low_participation} > {max_low_participation_rounds}). "
                                    f"This trial's hyperparameters are incompatible with the dataset."
                                )
                                return float('-inf') if self.direction == "maximize" else float('inf')
                
                # EARLY STOPPING: Check if validation metric improved (FIXED: use rounds, not epochs)
                if round_num >= min_rounds_for_early_stopping:
                    # Get validation metric for early stopping check
                    current_metric = intermediate_value
                    
                    if self.direction == "maximize":
                        improved = current_metric > best_validation_metric
                    else:
                        improved = current_metric < best_validation_metric
                    
                    if improved:
                        best_validation_metric = current_metric
                        no_improvement_counter = 0
                        logger.info(f"   ✅ Round {round_num}: Improvement detected (metric={current_metric:.4f})")
                    else:
                        no_improvement_counter += 1
                        logger.debug(f"   ⏳ Round {round_num}: No improvement ({no_improvement_counter}/{early_stopping_patience})")
                        
                        # Early stopping: break if no improvement for patience rounds (FIXED: rounds, not epochs)
                        if no_improvement_counter >= early_stopping_patience:
                            logger.info(f"   🛑 Early stopping triggered at round {round_num}: No improvement for {early_stopping_patience} rounds")
                            logger.info(f"      Best metric: {best_validation_metric:.4f} at round {round_num - early_stopping_patience}")
                            break
            
            # Evaluate base model (use validation set for intermediate pruning check)
            base_results = system.evaluate_base_model_only()
            system.base_evaluation_results = base_results
            
            # Final pruning check after base model evaluation
            base_metric = base_results.get('accuracy', 0.0)
            trial.report(base_metric, step=config.num_rounds + 1)
            if trial.should_prune():
                logger.warning(f"✂️  Trial {trial_number + 1} pruned after base model evaluation (metric={base_metric:.4f})")
                raise optuna.TrialPruned()
            
            # Run TTT adaptation
            adapted_model = system.perform_coordinator_side_ttt_adaptation()
            if adapted_model is None:
                logger.error("❌ TTT adaptation failed")
                return float('-inf') if self.direction == "maximize" else float('inf')
            
            # Evaluate adapted model
            adapted_results = system.evaluate_adapted_model(adapted_model)
            system.adapted_evaluation_results = adapted_results
            
            # VALIDATION: Check test set size (lowered threshold for CICIDS - smaller test sets after sequence filtering)
            test_set_size = len(system.preprocessed_data.get('X_test', []))
            if test_set_size < 20:  # Lowered from 100 to 20 - CICIDS produces smaller test sets after sequence filtering
                logger.error(f"❌ Test set too small ({test_set_size} samples) - insufficient for reliable evaluation")
                return float('-inf') if self.direction == "maximize" else float('inf')
            elif test_set_size < 50:
                logger.warning(f"⚠️  Test set is small ({test_set_size} samples) - metrics may be less reliable")
            
            # Extract metrics (same as UNSW version)
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
            
            # Calculate metric_value for optimization (same logic as UNSW version)
            metric_value = 0.0
            
            if self.metric == "balanced_base_ttt":
                base_f1_weight = 0.40
                ttt_zdr_weight = 0.30
                ttt_f1_weight = 0.30
                
                base_f1_score = base_f1
                ttt_zdr_score = ttt_zdr
                ttt_f1_score = ttt_f1
                
                metric_value = (
                    base_f1_weight * base_f1_score +
                    ttt_zdr_weight * ttt_zdr_score +
                    ttt_f1_weight * ttt_f1_score
                )
                
                logger.debug(f"  Balanced Base + TTT objective breakdown:")
                logger.debug(f"    Base F1: {base_f1_score:.4f} (weight: {base_f1_weight:.2f}) → {base_f1_weight * base_f1_score:.4f}")
                logger.debug(f"    TTT ZDR: {ttt_zdr_score:.4f} (weight: {ttt_zdr_weight:.2f}) → {ttt_zdr_weight * ttt_zdr_score:.4f}")
                logger.debug(f"    TTT F1: {ttt_f1_score:.4f} (weight: {ttt_f1_weight:.2f}) → {ttt_f1_weight * ttt_f1_score:.4f}")
                logger.debug(f"    Combined: {metric_value:.4f}")
                
                trial.set_user_attr("balanced_base_ttt_score", metric_value)
                trial.set_user_attr("balanced_base_f1_component", base_f1_weight * base_f1_score)
                trial.set_user_attr("balanced_ttt_zdr_component", ttt_zdr_weight * ttt_zdr_score)
                trial.set_user_attr("balanced_ttt_f1_component", ttt_f1_weight * ttt_f1_score)
            elif self.metric == "multi_objective":
                zdr_weight = 0.30
                non_zero_day_f1_weight = 0.35
                overall_f1_weight = 0.35
                
                non_zero_day_f1 = ttt_non_zero_day_f1
                zdr_score = ttt_zdr
                non_zero_day_f1_score = non_zero_day_f1
                overall_f1_score = ttt_f1
                
                metric_value = (
                    zdr_weight * zdr_score +
                    non_zero_day_f1_weight * non_zero_day_f1_score +
                    overall_f1_weight * overall_f1_score
                )
                
                logger.debug(f"  Balanced multi-objective breakdown:")
                logger.debug(f"    Zero-day ZDR: {zdr_score:.4f} (weight: {zdr_weight:.2f}) → {zdr_weight * zdr_score:.4f}")
                logger.debug(f"    Non-zero-day F1: {non_zero_day_f1_score:.4f} (weight: {non_zero_day_f1_weight:.2f}) → {non_zero_day_f1_weight * non_zero_day_f1_score:.4f}")
                logger.debug(f"    Overall F1: {overall_f1_score:.4f} (weight: {overall_f1_weight:.2f}) → {overall_f1_weight * overall_f1_score:.4f}")
                logger.debug(f"    Combined: {metric_value:.4f}")
                
                trial.set_user_attr("balanced_multi_objective_score", metric_value)
                trial.set_user_attr("balanced_zdr_component", zdr_weight * zdr_score)
                trial.set_user_attr("balanced_non_zero_day_f1_component", non_zero_day_f1_weight * non_zero_day_f1_score)
                trial.set_user_attr("balanced_overall_f1_component", overall_f1_weight * overall_f1_score)
            elif self.metric == "improved_multi_objective":
                # IMPROVED MULTI-OBJECTIVE: Balanced metric including base model performance
                # Formula: 0.25 × base_f1 + 0.25 × ttt_zero_day_zdr + 0.25 × ttt_non_zero_day_f1 + 0.25 × ttt_overall_f1
                base_f1_weight = 0.25
                ttt_zdr_weight = 0.25
                ttt_non_zero_day_f1_weight = 0.25
                ttt_overall_f1_weight = 0.25
                
                # Extract components (ensure no double-counting)
                base_f1_component = base_f1  # Base model F1-score
                ttt_zdr_component = ttt_zdr  # TTT zero-day detection rate
                ttt_non_zero_day_f1_component = ttt_non_zero_day_f1  # TTT non-zero-day F1
                ttt_overall_f1_component = ttt_f1  # TTT overall F1-score
                
                # VALIDATION: Check all components are valid (0.0-1.0 range)
                components = {
                    'base_f1': base_f1_component,
                    'ttt_zdr': ttt_zdr_component,
                    'ttt_non_zero_day_f1': ttt_non_zero_day_f1_component,
                    'ttt_overall_f1': ttt_overall_f1_component
                }
                
                invalid_components = {k: v for k, v in components.items() if not (0.0 <= v <= 1.0)}
                if invalid_components:
                    logger.warning(f"⚠️ Invalid component values (outside 0.0-1.0): {invalid_components}")
                
                # VALIDATION: Check weights sum to 1.0
                total_weight = base_f1_weight + ttt_zdr_weight + ttt_non_zero_day_f1_weight + ttt_overall_f1_weight
                if abs(total_weight - 1.0) > 1e-6:
                    logger.error(f"❌ Weight sum = {total_weight} (should be 1.0). Metric calculation may be incorrect!")
                
                # Calculate improved multi-objective score
                metric_value = (
                    base_f1_weight * base_f1_component +
                    ttt_zdr_weight * ttt_zdr_component +
                    ttt_non_zero_day_f1_weight * ttt_non_zero_day_f1_component +
                    ttt_overall_f1_weight * ttt_overall_f1_component
                )
                
                logger.debug(f"  Improved multi-objective breakdown:")
                logger.debug(f"    Base F1: {base_f1_component:.4f} (weight: {base_f1_weight:.2f}) → {base_f1_weight * base_f1_component:.4f}")
                logger.debug(f"    TTT Zero-day ZDR: {ttt_zdr_component:.4f} (weight: {ttt_zdr_weight:.2f}) → {ttt_zdr_weight * ttt_zdr_component:.4f}")
                logger.debug(f"    TTT Non-zero-day F1: {ttt_non_zero_day_f1_component:.4f} (weight: {ttt_non_zero_day_f1_weight:.2f}) → {ttt_non_zero_day_f1_weight * ttt_non_zero_day_f1_component:.4f}")
                logger.debug(f"    TTT Overall F1: {ttt_overall_f1_component:.4f} (weight: {ttt_overall_f1_weight:.2f}) → {ttt_overall_f1_weight * ttt_overall_f1_component:.4f}")
                logger.debug(f"    Combined: {metric_value:.4f}")
                
                # Store all components separately in trial attributes for analysis
                trial.set_user_attr("improved_multi_objective_score", metric_value)
                trial.set_user_attr("improved_base_f1_component", base_f1_weight * base_f1_component)
                trial.set_user_attr("improved_ttt_zdr_component", ttt_zdr_weight * ttt_zdr_component)
                trial.set_user_attr("improved_ttt_non_zero_day_f1_component", ttt_non_zero_day_f1_weight * ttt_non_zero_day_f1_component)
                trial.set_user_attr("improved_ttt_overall_f1_component", ttt_overall_f1_weight * ttt_overall_f1_component)
                
                # Store raw component values for validation
                trial.set_user_attr("raw_base_f1", base_f1_component)
                trial.set_user_attr("raw_ttt_zdr", ttt_zdr_component)
                trial.set_user_attr("raw_ttt_non_zero_day_f1", ttt_non_zero_day_f1_component)
                trial.set_user_attr("raw_ttt_overall_f1", ttt_overall_f1_component)
            elif self.metric == "ttt_auc_pr":
                metric_value = ttt_auc_pr
            elif self.metric == "ttt_f1_score":
                metric_value = ttt_f1
            elif self.metric == "ttt_accuracy":
                metric_value = ttt_accuracy
            elif self.metric == "ttt_zero_day_detection_rate":
                metric_value = ttt_zdr
            else:
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
                # Hyperparameters for this trial
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
                # Improved multi-objective metrics (if applicable)
                **({
                    "improved_multi_objective_score": metric_value,
                    "improved_base_f1_component": 0.25 * base_f1,
                    "improved_ttt_zdr_component": 0.25 * ttt_zdr,
                    "improved_ttt_non_zero_day_f1_component": 0.25 * ttt_non_zero_day_f1,
                    "improved_ttt_overall_f1_component": 0.25 * ttt_f1,
                } if self.metric == "improved_multi_objective" else {}),
                
                # Balanced multi-objective metrics (if applicable)
                **({
                    "balanced_multi_objective_score": metric_value,
                    "balanced_zdr_component": 0.30 * ttt_zdr,
                    "balanced_non_zero_day_f1_component": 0.35 * ttt_non_zero_day_f1,
                    "balanced_overall_f1_component": 0.35 * ttt_f1,
                } if self.metric == "multi_objective" else {}),
                
                # Trial info
                "trial_number": trial.number,
                "trial_state": "COMPLETE",
                "dataset": "CICIDS2017",
                "zero_day_attack": self.zero_day_attack
            })
            
            # Report to Optuna
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
            trial.set_user_attr("dataset", "CICIDS2017")
            trial.set_user_attr("zero_day_attack", self.zero_day_attack)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"CICIDS2017 Trial {trial.number + 1} Results:")
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
            elif self.metric == "improved_multi_objective":
                base_f1_weight = 0.25
                ttt_zdr_weight = 0.25
                ttt_non_zero_day_f1_weight = 0.25
                ttt_overall_f1_weight = 0.25
                logger.info(f"  🎯 Improved Multi-Objective Score ({self.metric}):")
                logger.info(f"    Components (balanced for base model, zero-day, and known attacks):")
                logger.info(f"      Base F1: {base_f1:.4f} × {base_f1_weight:.2f} = {base_f1_weight * base_f1:.4f}")
                logger.info(f"      TTT Zero-day ZDR: {ttt_zdr:.4f} × {ttt_zdr_weight:.2f} = {ttt_zdr_weight * ttt_zdr:.4f}")
                logger.info(f"      TTT Non-zero-day F1: {ttt_non_zero_day_f1:.4f} × {ttt_non_zero_day_f1_weight:.2f} = {ttt_non_zero_day_f1_weight * ttt_non_zero_day_f1:.4f}")
                logger.info(f"      TTT Overall F1: {ttt_f1:.4f} × {ttt_overall_f1_weight:.2f} = {ttt_overall_f1_weight * ttt_f1:.4f}")
                logger.info(f"    Combined Score: {metric_value:.4f}")
                logger.info(f"    📊 Balance: 25% base + 25% zero-day + 25% non-zero-day + 25% overall = 100%")
                logger.info(f"    ✅ Includes base model for fair comparison")
            else:
                logger.info(f"  Optimizing metric ({self.metric}): {metric_value:.4f}")
            logger.info(f"{'='*80}\n")
            
            # Cleanup
            del system
            torch.cuda.empty_cache()
            
            return metric_value
            
        except optuna.TrialPruned:
            # Trial was pruned - this is expected behavior, not an error
            logger.info(f"✂️  Trial {trial.number + 1} was pruned (stopped early to save time)")
            wandb.log({
                "trial_number": trial.number,
                "trial_state": "PRUNED",
                "dataset": "CICIDS2017"
            })
            raise  # Re-raise to let Optuna know the trial was pruned
        except Exception as e:
            logger.error(f"❌ CICIDS2017 Trial {trial.number + 1} failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            wandb.log({
                "trial_number": trial.number,
                "trial_state": "FAILED",
                "error": str(e),
                "dataset": "CICIDS2017"
            })
            return float('-inf') if self.direction == "maximize" else float('inf')
    
    def optimize(self):
        """Run the optimization study"""
        logger.info(f"🚀 Starting CICIDS2017 Optuna optimization study: {self.study_name}")
        logger.info(f"   Dataset: CICIDS2017")
        logger.info(f"   Zero-Day Attack: {self.zero_day_attack}")
        logger.info(f"   Trials: {self.n_trials}")
        logger.info(f"   Direction: {self.direction}")
        logger.info(f"   Metric: {self.metric}")
        logger.info(f"   Wandb Project: cicids-zero-day-detection-optimization\n")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Log best trial to Wandb
        best_trial = self.study.best_trial
        wandb.log({
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_user_attrs": best_trial.user_attrs,
            "dataset": "CICIDS2017",
            "zero_day_attack": self.zero_day_attack
        })
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ CICIDS2017 Optimization Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: CICIDS2017")
        logger.info(f"Zero-Day Attack: {self.zero_day_attack}")
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
        best_params_path = Path("best_hyperparameters_cicids.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                "dataset": "CICIDS2017",
                "zero_day_attack": self.zero_day_attack,
                "best_trial_number": best_trial.number,
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_user_attrs": best_trial.user_attrs
            }, f, indent=2)
        
        logger.info(f"💾 Best CICIDS2017 hyperparameters saved to: {best_params_path}")
        
        # Finish Wandb run
        wandb.finish()
        
        return best_trial


def main():
    """Main entry point for CICIDS2017 hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for CICIDS2017 dataset using Optuna and Wandb")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--study_name", type=str, default="cicids_zero_day_detection_optimization", 
                       help="Name for Optuna study")
    parser.add_argument("--metric", type=str, default="balanced_base_ttt",
                       choices=["balanced_base_ttt", "ttt_zero_day_detection_rate", "ttt_auc_pr", "ttt_f1_score", "ttt_accuracy", "multi_objective", "improved_multi_objective"],
                       help="Primary metric to optimize. 'balanced_base_ttt' (default) optimizes for both base model (40% base F1) and TTT performance (30% ZDR, 30% TTT F1). 'multi_objective' balances TTT-only metrics.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"],
                       help="Optimization direction")
    parser.add_argument("--zero_day_attack", type=str, default="PortScan",
                       help="Zero-day attack type for CICIDS2017 (default: PortScan). Options: PortScan, DDoS, DoS Hulk, etc.")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = HyperparameterOptimizerCICIDS(
        study_name=args.study_name,
        n_trials=args.n_trials,
        direction=args.direction,
        metric=args.metric,
        zero_day_attack=args.zero_day_attack
    )
    
    # Run optimization
    best_trial = optimizer.optimize()
    
    return best_trial


if __name__ == "__main__":
    main()

