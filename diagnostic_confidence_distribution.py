#!/usr/bin/env python3
"""
Diagnostic Test: Confidence Distribution Analysis
Compares confidence distributions before and after TTT adaptation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import BlockchainFederatedIncentiveSystem
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_confidence_distribution_analysis():
    """
    Test 1: Confidence Distribution Analysis
    Compares confidence distributions before and after TTT adaptation
    """
    logger.info("=" * 80)
    logger.info("🔍 DIAGNOSTIC TEST 1: CONFIDENCE DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)
    
    try:
        # Initialize system
        logger.info("📦 Initializing system...")
        config = get_config()
        system = BlockchainFederatedIncentiveSystem(config)
        
        # Initialize system components (preprocessor, model, coordinator)
        logger.info("📦 Initializing system components...")
        if not system.initialize_system():
            logger.error("❌ System initialization failed")
            return False
        
        # Load preprocessed data
        logger.info("📥 Loading preprocessed data...")
        if not system.preprocess_data():
            logger.error("❌ Data preprocessing failed")
            return False
        
        # Check if preprocessed data exists
        if not hasattr(system, 'preprocessed_data') or system.preprocessed_data is None:
            logger.error("❌ Preprocessed data not found after preprocessing")
            return False
        
        # Setup federated learning structure (doesn't train, just sets up)
        logger.info("📦 Setting up federated learning structure...")
        if not system.setup_federated_learning():
            logger.error("❌ Federated learning setup failed")
            return False
        
        # Check if we have a model
        if system.coordinator.model is None:
            logger.error("❌ No model found after setup")
            return False
        
        # Check if model has been trained (has non-random weights)
        # This is a simple check - if model hasn't been trained, weights will be near initialization
        logger.info("🔍 Checking if model has been trained...")
        model_trained = False
        try:
            # Check a few weight parameters to see if they've been trained
            # Random initialization typically has weights near 0, trained models have learned weights
            sample_param = next(system.coordinator.model.parameters())
            param_magnitude = torch.abs(sample_param.data).mean().item()
            
            # If weights are very small (near 0), model might not be trained
            # But this is heuristic - better to check if training_history exists
            if hasattr(system, 'training_history') and len(system.training_history) > 0:
                model_trained = True
                logger.info(f"✅ Model appears to be trained (training history found)")
            elif param_magnitude > 0.01:  # Heuristic: trained models have larger weights
                model_trained = True
                logger.info(f"✅ Model appears to be trained (param magnitude: {param_magnitude:.4f})")
            else:
                logger.warning(f"⚠️  Model may not be trained (param magnitude: {param_magnitude:.4f})")
                logger.warning("⚠️  Results may not be meaningful without a trained model")
                logger.warning("⚠️  Consider running main.py first to train the model")
                # Continue anyway - user can still see diagnostic
                model_trained = False
        except Exception as e:
            logger.warning(f"⚠️  Could not verify model training status: {e}")
            logger.warning("⚠️  Continuing with diagnostic (results may not be meaningful)")
        
        logger.info("✅ System initialized with preprocessed data")
        
        # Get test data
        X_test = torch.FloatTensor(system.preprocessed_data['X_test']).to(system.device)
        y_test = torch.LongTensor(system.preprocessed_data['y_test']).to(system.device)
        
        logger.info(f"📊 Test data shape: {X_test.shape}")
        logger.info(f"📊 Test labels shape: {y_test.shape}")
        logger.info(f"📊 Number of test samples: {len(X_test)}")
        
        # Use a subset for faster analysis (optional - can use full dataset)
        max_samples = min(1000, len(X_test))
        if len(X_test) > max_samples:
            logger.info(f"📊 Using subset of {max_samples} samples for analysis")
            indices = torch.randperm(len(X_test))[:max_samples]
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        # Get base model (before TTT)
        logger.info("🔍 Getting base model predictions (before TTT)...")
        base_model = system.coordinator.model
        base_model.eval()
        
        with torch.no_grad():
            logits_before = base_model(X_test)
            probs_before = torch.softmax(logits_before, dim=1)
            confidence_before = torch.max(probs_before, dim=1)[0]
            
            # For binary classification: get attack probability (class 1)
            # Assuming multiclass: class 0 = Normal, classes 1-9 = Attacks
            # Convert to binary: Normal=0, Attack=1
            num_classes = probs_before.shape[1]
            if num_classes > 2:
                # Attack probability = 1 - Normal probability
                attack_prob_before = 1.0 - probs_before[:, 0]
            else:
                # Binary classification
                attack_prob_before = probs_before[:, 1] if num_classes == 2 else probs_before[:, 0]
        
        logger.info(f"✅ Base model predictions complete")
        logger.info(f"   Mean confidence (before): {confidence_before.mean().item():.4f}")
        logger.info(f"   Std confidence (before): {confidence_before.std().item():.4f}")
        logger.info(f"   Mean attack prob (before): {attack_prob_before.mean().item():.4f}")
        
        # Perform TTT adaptation
        logger.info("🔄 Performing TTT adaptation...")
        
        # Use a subset of test data for adaptation (to match evaluation setup)
        adaptation_size = min(200, len(X_test))
        adaptation_indices = torch.randperm(len(X_test))[:adaptation_size]
        X_adapt = X_test[adaptation_indices]
        
        # Perform unsupervised TTT adaptation
        adapted_model = system.coordinator._perform_advanced_ttt_adaptation(
            X_adapt, system.config
        )
        adapted_model.eval()
        
        logger.info("🔍 Getting adapted model predictions (after TTT)...")
        with torch.no_grad():
            logits_after = adapted_model(X_test)
            probs_after = torch.softmax(logits_after, dim=1)
            confidence_after = torch.max(probs_after, dim=1)[0]
            
            # Get attack probability
            if num_classes > 2:
                attack_prob_after = 1.0 - probs_after[:, 0]
            else:
                attack_prob_after = probs_after[:, 1] if num_classes == 2 else probs_after[:, 0]
        
        logger.info(f"✅ Adapted model predictions complete")
        logger.info(f"   Mean confidence (after): {confidence_after.mean().item():.4f}")
        logger.info(f"   Std confidence (after): {confidence_after.std().item():.4f}")
        logger.info(f"   Mean attack prob (after): {attack_prob_after.mean().item():.4f}")
        
        # Calculate statistics
        confidence_before_np = confidence_before.cpu().numpy()
        confidence_after_np = confidence_after.cpu().numpy()
        attack_prob_before_np = attack_prob_before.cpu().numpy()
        attack_prob_after_np = attack_prob_after.cpu().numpy()
        
        confidence_mean_change = np.mean(confidence_after_np) - np.mean(confidence_before_np)
        confidence_std_change = np.std(confidence_after_np) - np.std(confidence_before_np)
        attack_prob_mean_change = np.mean(attack_prob_after_np) - np.mean(attack_prob_before_np)
        
        logger.info("\n📊 STATISTICAL SUMMARY:")
        logger.info(f"   Confidence mean change: {confidence_mean_change:+.4f}")
        logger.info(f"   Confidence std change: {confidence_std_change:+.4f}")
        logger.info(f"   Attack prob mean change: {attack_prob_mean_change:+.4f}")
        
        # Create plots
        logger.info("📈 Generating plots...")
        output_dir = Path("performance_plots")
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Confidence Distribution Analysis: Before vs After TTT', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Confidence Distribution
        ax1 = axes[0]
        ax1.hist(confidence_before_np, bins=50, alpha=0.7, label='Before TTT', 
                color='blue', edgecolor='black', linewidth=0.5)
        ax1.hist(confidence_after_np, bins=50, alpha=0.7, label='After TTT', 
                color='orange', edgecolor='black', linewidth=0.5)
        ax1.axvline(x=confidence_before_np.mean(), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean Before: {confidence_before_np.mean():.3f}')
        ax1.axvline(x=confidence_after_np.mean(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Mean After: {confidence_after_np.mean():.3f}')
        ax1.set_xlabel('Prediction Confidence', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Confidence Distribution Shift', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attack Probability Distribution
        ax2 = axes[1]
        ax2.hist(attack_prob_before_np, bins=50, alpha=0.7, label='Before TTT', 
                color='blue', edgecolor='black', linewidth=0.5)
        ax2.hist(attack_prob_after_np, bins=50, alpha=0.7, label='After TTT', 
                color='orange', edgecolor='black', linewidth=0.5)
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                   label='Decision Threshold (0.5)')
        ax2.axvline(x=attack_prob_before_np.mean(), color='blue', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        ax2.axvline(x=attack_prob_after_np.mean(), color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('P(Attack)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Attack Probability Distribution', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "diagnostic_confidence_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved to: {plot_path}")
        
        # Also save as PDF
        plot_path_pdf = output_dir / "diagnostic_confidence_distribution.pdf"
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved to: {plot_path_pdf}")
        
        plt.close()
        
        # Additional analysis: High-confidence predictions
        logger.info("\n🔍 ADDITIONAL ANALYSIS:")
        high_conf_threshold = 0.9
        high_conf_before = np.sum(confidence_before_np >= high_conf_threshold)
        high_conf_after = np.sum(confidence_after_np >= high_conf_threshold)
        high_conf_pct_before = (high_conf_before / len(confidence_before_np)) * 100
        high_conf_pct_after = (high_conf_after / len(confidence_after_np)) * 100
        
        logger.info(f"   High-confidence predictions (≥{high_conf_threshold}):")
        logger.info(f"     Before TTT: {high_conf_before} ({high_conf_pct_before:.2f}%)")
        logger.info(f"     After TTT: {high_conf_after} ({high_conf_pct_after:.2f}%)")
        logger.info(f"     Change: {high_conf_after - high_conf_before:+d} ({high_conf_pct_after - high_conf_pct_before:+.2f}%)")
        
        # Low-confidence predictions
        low_conf_threshold = 0.5
        low_conf_before = np.sum(confidence_before_np < low_conf_threshold)
        low_conf_after = np.sum(confidence_after_np < low_conf_threshold)
        low_conf_pct_before = (low_conf_before / len(confidence_before_np)) * 100
        low_conf_pct_after = (low_conf_after / len(confidence_after_np)) * 100
        
        logger.info(f"   Low-confidence predictions (<{low_conf_threshold}):")
        logger.info(f"     Before TTT: {low_conf_before} ({low_conf_pct_before:.2f}%)")
        logger.info(f"     After TTT: {low_conf_after} ({low_conf_pct_after:.2f}%)")
        logger.info(f"     Change: {low_conf_after - low_conf_before:+d} ({low_conf_pct_after - low_conf_pct_before:+.2f}%)")
        
        # Save statistics to JSON
        stats = {
            "confidence_before": {
                "mean": float(np.mean(confidence_before_np)),
                "std": float(np.std(confidence_before_np)),
                "min": float(np.min(confidence_before_np)),
                "max": float(np.max(confidence_before_np)),
                "median": float(np.median(confidence_before_np))
            },
            "confidence_after": {
                "mean": float(np.mean(confidence_after_np)),
                "std": float(np.std(confidence_after_np)),
                "min": float(np.min(confidence_after_np)),
                "max": float(np.max(confidence_after_np)),
                "median": float(np.median(confidence_after_np))
            },
            "attack_prob_before": {
                "mean": float(np.mean(attack_prob_before_np)),
                "std": float(np.std(attack_prob_before_np)),
                "min": float(np.min(attack_prob_before_np)),
                "max": float(np.max(attack_prob_before_np)),
                "median": float(np.median(attack_prob_before_np))
            },
            "attack_prob_after": {
                "mean": float(np.mean(attack_prob_after_np)),
                "std": float(np.std(attack_prob_after_np)),
                "min": float(np.min(attack_prob_after_np)),
                "max": float(np.max(attack_prob_after_np)),
                "median": float(np.median(attack_prob_after_np))
            },
            "changes": {
                "confidence_mean_change": float(confidence_mean_change),
                "confidence_std_change": float(confidence_std_change),
                "attack_prob_mean_change": float(attack_prob_mean_change),
                "high_conf_before_pct": float(high_conf_pct_before),
                "high_conf_after_pct": float(high_conf_pct_after),
                "low_conf_before_pct": float(low_conf_pct_before),
                "low_conf_after_pct": float(low_conf_pct_after)
            }
        }
        
        stats_path = output_dir / "diagnostic_confidence_statistics.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✅ Statistics saved to: {stats_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DIAGNOSTIC TEST 1 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in confidence distribution analysis: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_confidence_distribution_analysis()
    sys.exit(0 if success else 1)

