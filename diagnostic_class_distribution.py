#!/usr/bin/env python3
"""
Diagnostic Test: Class Distribution Shift Analysis
Analyzes how TTT adaptation changes prediction class distribution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

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

def run_class_distribution_analysis():
    """
    Test 2: Class Distribution Shift Analysis
    Analyzes how TTT adaptation changes prediction class distribution
    """
    logger.info("=" * 80)
    logger.info("🔍 DIAGNOSTIC TEST 2: CLASS DISTRIBUTION SHIFT ANALYSIS")
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
        
        logger.info("✅ System initialized with preprocessed data")
        
        # Get test data
        X_test = torch.FloatTensor(system.preprocessed_data['X_test']).to(system.device)
        y_test = torch.LongTensor(system.preprocessed_data['y_test']).to(system.device)
        
        # Convert multiclass to binary for analysis (Normal=0, Attack=1)
        y_test_binary = (y_test != 0).long() if y_test.max() > 1 else y_test
        
        logger.info(f"📊 Test data shape: {X_test.shape}")
        logger.info(f"📊 Test labels shape: {y_test_binary.shape}")
        logger.info(f"📊 Number of test samples: {len(X_test)}")
        
        # Use a subset for faster analysis (optional - can use full dataset)
        max_samples = min(1000, len(X_test))
        if len(X_test) > max_samples:
            logger.info(f"📊 Using subset of {max_samples} samples for analysis")
            indices = torch.randperm(len(X_test))[:max_samples]
            X_test = X_test[indices]
            y_test_binary = y_test_binary[indices]
        
        # Get base model (before TTT)
        logger.info("🔍 Getting base model predictions (before TTT)...")
        base_model = system.coordinator.model
        base_model.eval()
        
        with torch.no_grad():
            logits_before = base_model(X_test)
            probs_before = torch.softmax(logits_before, dim=1)
            preds_before = torch.argmax(probs_before, dim=1)
            
            # For binary classification: get attack probability (class 1)
            num_classes = probs_before.shape[1]
            if num_classes > 2:
                # Attack probability = 1 - Normal probability
                attack_prob_before = 1.0 - probs_before[:, 0]
                # Convert multiclass predictions to binary (0=Normal, 1=Attack)
                preds_before_binary = (preds_before != 0).long()
            else:
                # Binary classification
                attack_prob_before = probs_before[:, 1] if num_classes == 2 else probs_before[:, 0]
                preds_before_binary = preds_before
        
        # Calculate attack rate before TTT
        preds_before_np = preds_before_binary.cpu().numpy()
        attack_rate_before = (preds_before_np == 1).sum() / len(preds_before_np)
        
        logger.info(f"✅ Base model predictions complete")
        logger.info(f"   Attack prediction rate (before): {attack_rate_before:.2%}")
        logger.info(f"   Normal prediction rate (before): {1 - attack_rate_before:.2%}")
        
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
            preds_after = torch.argmax(probs_after, dim=1)
            
            # Get attack probability
            if num_classes > 2:
                attack_prob_after = 1.0 - probs_after[:, 0]
                preds_after_binary = (preds_after != 0).long()
            else:
                attack_prob_after = probs_after[:, 1] if num_classes == 2 else probs_after[:, 0]
                preds_after_binary = preds_after
        
        # Calculate attack rate after TTT
        preds_after_np = preds_after_binary.cpu().numpy()
        attack_rate_after = (preds_after_np == 1).sum() / len(preds_after_np)
        
        # True distribution
        y_test_binary_np = y_test_binary.cpu().numpy()
        true_attack_rate = (y_test_binary_np == 1).sum() / len(y_test_binary_np)
        
        logger.info(f"✅ Adapted model predictions complete")
        logger.info(f"   Attack prediction rate (after): {attack_rate_after:.2%}")
        logger.info(f"   Normal prediction rate (after): {1 - attack_rate_after:.2%}")
        
        # Calculate changes
        attack_rate_change = attack_rate_after - attack_rate_before
        attack_rate_change_pct = (attack_rate_after - attack_rate_before) / attack_rate_before * 100 if attack_rate_before > 0 else 0
        
        # Calculate bias from true distribution
        bias_before = attack_rate_before - true_attack_rate
        bias_after = attack_rate_after - true_attack_rate
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 CLASS DISTRIBUTION ANALYSIS RESULTS")
        logger.info("=" * 80)
        logger.info(f"Attack Prediction Rate:")
        logger.info(f"  Before TTT: {attack_rate_before:.2%}")
        logger.info(f"  After TTT:  {attack_rate_after:.2%}")
        logger.info(f"  True Rate:  {true_attack_rate:.2%}")
        logger.info(f"\nChange: {attack_rate_change:+.2%} absolute ({attack_rate_change_pct:+.1f}% relative)")
        logger.info(f"\nBias from True Distribution:")
        logger.info(f"  Before TTT: {bias_before:+.2%} ({'over-predicting' if bias_before > 0 else 'under-predicting' if bias_before < 0 else 'perfect'} attacks)")
        logger.info(f"  After TTT:  {bias_after:+.2%} ({'over-predicting' if bias_after > 0 else 'under-predicting' if bias_after < 0 else 'perfect'} attacks)")
        logger.info(f"  Bias change: {bias_after - bias_before:+.2%} ({'improved' if abs(bias_after) < abs(bias_before) else 'worsened' if abs(bias_after) > abs(bias_before) else 'unchanged'})")
        
        # Confusion matrices
        cm_before = confusion_matrix(y_test_binary_np, preds_before_np)
        cm_after = confusion_matrix(y_test_binary_np, preds_after_np)
        
        logger.info("\n📊 CONFUSION MATRIX ANALYSIS:")
        logger.info("\nBefore TTT:")
        logger.info(f"  True Negatives (TN): {cm_before[0, 0]}")
        logger.info(f"  False Positives (FP): {cm_before[0, 1]}")
        logger.info(f"  False Negatives (FN): {cm_before[1, 0]}")
        logger.info(f"  True Positives (TP): {cm_before[1, 1]}")
        
        logger.info("\nAfter TTT:")
        logger.info(f"  True Negatives (TN): {cm_after[0, 0]}")
        logger.info(f"  False Positives (FP): {cm_after[0, 1]}")
        logger.info(f"  False Negatives (FN): {cm_after[1, 0]}")
        logger.info(f"  True Positives (TP): {cm_after[1, 1]}")
        
        # Calculate changes in confusion matrix
        tn_change = cm_after[0, 0] - cm_before[0, 0]
        fp_change = cm_after[0, 1] - cm_before[0, 1]
        fn_change = cm_after[1, 0] - cm_before[1, 0]
        tp_change = cm_after[1, 1] - cm_before[1, 1]
        
        logger.info("\nChanges:")
        logger.info(f"  TN: {tn_change:+d}, FP: {fp_change:+d}, FN: {fn_change:+d}, TP: {tp_change:+d}")
        
        # Create plots
        logger.info("📈 Generating plots...")
        output_dir = Path("performance_plots")
        output_dir.mkdir(exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Class Distribution Shift Analysis: Before vs After TTT', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Prediction Rate Comparison
        ax1 = plt.subplot(2, 3, 1)
        categories = ['Before TTT', 'After TTT', 'True Rate']
        rates = [attack_rate_before, attack_rate_after, true_attack_rate]
        colors = ['blue', 'orange', 'green']
        bars = ax1.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Attack Rate', fontsize=11)
        ax1.set_title('Attack Prediction Rate Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, max(rates) * 1.2])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Class Distribution Pie Charts
        ax2 = plt.subplot(2, 3, 2)
        sizes_before = [1 - attack_rate_before, attack_rate_before]
        labels = ['Normal', 'Attack']
        colors_pie = ['lightblue', 'lightcoral']
        ax2.pie(sizes_before, labels=labels, autopct='%1.1f%%', startangle=90, 
               colors=colors_pie, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax2.set_title('Before TTT\nPrediction Distribution', fontsize=12, fontweight='bold')
        
        ax3 = plt.subplot(2, 3, 3)
        sizes_after = [1 - attack_rate_after, attack_rate_after]
        ax3.pie(sizes_after, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors_pie, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax3.set_title('After TTT\nPrediction Distribution', fontsize=12, fontweight='bold')
        
        # Plot 4: Confusion Matrix Before TTT
        ax4 = plt.subplot(2, 3, 4)
        im4 = ax4.imshow(cm_before, interpolation='nearest', cmap=plt.cm.Blues)
        ax4.figure.colorbar(im4, ax=ax4)
        ax4.set(xticks=np.arange(cm_before.shape[1]),
               yticks=np.arange(cm_before.shape[0]),
               xticklabels=['Normal', 'Attack'],
               yticklabels=['Normal', 'Attack'],
               title='Confusion Matrix: Before TTT',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm_before.max() / 2.
        for i in range(cm_before.shape[0]):
            for j in range(cm_before.shape[1]):
                ax4.text(j, i, format(cm_before[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm_before[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
        
        # Plot 5: Confusion Matrix After TTT
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.imshow(cm_after, interpolation='nearest', cmap=plt.cm.Oranges)
        ax5.figure.colorbar(im5, ax=ax5)
        ax5.set(xticks=np.arange(cm_after.shape[1]),
               yticks=np.arange(cm_after.shape[0]),
               xticklabels=['Normal', 'Attack'],
               yticklabels=['Normal', 'Attack'],
               title='Confusion Matrix: After TTT',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm_after.max() / 2.
        for i in range(cm_after.shape[0]):
            for j in range(cm_after.shape[1]):
                ax5.text(j, i, format(cm_after[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm_after[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
        
        # Plot 6: Change Analysis
        ax6 = plt.subplot(2, 3, 6)
        changes = [tn_change, fp_change, fn_change, tp_change]
        change_labels = ['TN', 'FP', 'FN', 'TP']
        colors_change = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in changes]
        bars6 = ax6.bar(change_labels, changes, color=colors_change, alpha=0.7, 
                       edgecolor='black', linewidth=1.5)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.set_ylabel('Change', fontsize=11)
        ax6.set_title('Confusion Matrix Changes', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, change in zip(bars6, changes):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{change:+d}',
                    ha='center', va='bottom' if change > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "diagnostic_class_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved to: {plot_path}")
        
        # Also save as PDF
        plot_path_pdf = output_dir / "diagnostic_class_distribution.pdf"
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved to: {plot_path_pdf}")
        
        plt.close()
        
        # Save statistics to JSON
        stats = {
            "attack_rate": {
                "before_ttt": float(attack_rate_before),
                "after_ttt": float(attack_rate_after),
                "true_rate": float(true_attack_rate),
                "absolute_change": float(attack_rate_change),
                "relative_change_pct": float(attack_rate_change_pct)
            },
            "bias_analysis": {
                "bias_before": float(bias_before),
                "bias_after": float(bias_after),
                "bias_change": float(bias_after - bias_before),
                "improved": bool(abs(bias_after) < abs(bias_before))
            },
            "confusion_matrix_before": {
                "tn": int(cm_before[0, 0]),
                "fp": int(cm_before[0, 1]),
                "fn": int(cm_before[1, 0]),
                "tp": int(cm_before[1, 1])
            },
            "confusion_matrix_after": {
                "tn": int(cm_after[0, 0]),
                "fp": int(cm_after[0, 1]),
                "fn": int(cm_after[1, 0]),
                "tp": int(cm_after[1, 1])
            },
            "confusion_matrix_changes": {
                "tn_change": int(tn_change),
                "fp_change": int(fp_change),
                "fn_change": int(fn_change),
                "tp_change": int(tp_change)
            },
            "sample_size": int(len(X_test))
        }
        
        stats_path = output_dir / "diagnostic_class_distribution_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✅ Statistics saved to: {stats_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DIAGNOSTIC TEST 2 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in class distribution analysis: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_class_distribution_analysis()
    sys.exit(0 if success else 1)

