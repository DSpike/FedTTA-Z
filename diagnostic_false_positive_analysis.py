#!/usr/bin/env python3
"""
Diagnostic Test: False Positive Pattern Analysis
Analyzes false positive patterns and how TTT adaptation affects them
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

def run_false_positive_analysis():
    """
    Test 3: False Positive Pattern Analysis
    Analyzes false positive patterns and how TTT adaptation affects them
    """
    logger.info("=" * 80)
    logger.info("🔍 DIAGNOSTIC TEST 3: FALSE POSITIVE PATTERN ANALYSIS")
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
            
            # For binary classification: get attack probability (class 1)
            num_classes = probs_before.shape[1]
            if num_classes > 2:
                # Attack probability = 1 - Normal probability
                attack_prob_before = 1.0 - probs_before[:, 0]
            else:
                # Binary classification
                attack_prob_before = probs_before[:, 1] if num_classes == 2 else probs_before[:, 0]
            
            # Get binary predictions using threshold 0.5
            preds_before = (attack_prob_before > 0.5).long()
        
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
            
            # Get attack probability
            if num_classes > 2:
                attack_prob_after = 1.0 - probs_after[:, 0]
            else:
                attack_prob_after = probs_after[:, 1] if num_classes == 2 else probs_after[:, 0]
            
            # Get binary predictions using threshold 0.5
            preds_after = (attack_prob_after > 0.5).long()
        
        # Convert to numpy for analysis
        y_true_np = y_test_binary.cpu().numpy()
        preds_before_np = preds_before.cpu().numpy()
        preds_after_np = preds_after.cpu().numpy()
        attack_prob_before_np = attack_prob_before.cpu().numpy()
        attack_prob_after_np = attack_prob_after.cpu().numpy()
        
        # Confusion matrices
        cm_before = confusion_matrix(y_true_np, preds_before_np)
        cm_after = confusion_matrix(y_true_np, preds_after_np)
        
        # Analyze false positives
        fp_before = cm_before[0, 1]  # Normal predicted as Attack
        fp_after = cm_after[0, 1]
        fp_change = fp_after - fp_before
        fp_change_pct = (fp_change / fp_before * 100) if fp_before > 0 else 0
        
        # Analyze true positives (recall)
        tp_before = cm_before[1, 1]  # Attack predicted as Attack
        tp_after = cm_after[1, 1]
        tp_change = tp_after - tp_before
        tp_change_pct = (tp_change / tp_before * 100) if tp_before > 0 else float('inf') if tp_before == 0 else 0
        
        # Analyze true negatives
        tn_before = cm_before[0, 0]
        tn_after = cm_after[0, 0]
        tn_change = tn_after - tn_before
        
        # Analyze false negatives
        fn_before = cm_before[1, 0]
        fn_after = cm_after[1, 0]
        fn_change = fn_after - fn_before
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 FALSE POSITIVE PATTERN ANALYSIS RESULTS")
        logger.info("=" * 80)
        logger.info(f"False Positives (Normal → Attack):")
        logger.info(f"  Before TTT: {fp_before}")
        logger.info(f"  After TTT:  {fp_after}")
        logger.info(f"  Change: {fp_change:+d} ({fp_change_pct:+.1f}%)")
        
        logger.info(f"\nTrue Positives (Recall - Attack → Attack):")
        logger.info(f"  Before TTT: {tp_before}")
        logger.info(f"  After TTT:  {tp_after}")
        if tp_before > 0:
            logger.info(f"  Change: {tp_change:+d} ({tp_change_pct:+.1f}%)")
        else:
            logger.info(f"  Change: {tp_change:+d} (from 0 to {tp_after})")
        
        logger.info(f"\nTrue Negatives (Normal → Normal):")
        logger.info(f"  Before TTT: {tn_before}")
        logger.info(f"  After TTT:  {tn_after}")
        logger.info(f"  Change: {tn_change:+d}")
        
        logger.info(f"\nFalse Negatives (Attack → Normal):")
        logger.info(f"  Before TTT: {fn_before}")
        logger.info(f"  After TTT:  {fn_after}")
        logger.info(f"  Change: {fn_change:+d}")
        
        # Find which samples became false positives (new FPs)
        new_fp_indices = np.where((y_true_np == 0) & (preds_before_np == 0) & (preds_after_np == 1))[0]
        
        # Find which samples stopped being false positives (fixed FPs)
        fixed_fp_indices = np.where((y_true_np == 0) & (preds_before_np == 1) & (preds_after_np == 0))[0]
        
        # Find which samples became true positives (new TPs)
        new_tp_indices = np.where((y_true_np == 1) & (preds_before_np == 0) & (preds_after_np == 1))[0]
        
        # Find which samples became false negatives (new FNs)
        new_fn_indices = np.where((y_true_np == 1) & (preds_before_np == 1) & (preds_after_np == 0))[0]
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DETAILED PATTERN ANALYSIS")
        logger.info("=" * 80)
        
        # Analyze new false positives
        new_fp_confidence_before = np.array([])
        new_fp_confidence_after = np.array([])
        if len(new_fp_indices) > 0:
            new_fp_confidence_before = attack_prob_before_np[new_fp_indices]
            new_fp_confidence_after = attack_prob_after_np[new_fp_indices]
            
            logger.info(f"\n📈 New False Positives (Normal → Attack after TTT):")
            logger.info(f"  Count: {len(new_fp_indices)}")
            logger.info(f"  Confidence before TTT: {new_fp_confidence_before.mean():.3f} ± {new_fp_confidence_before.std():.3f}")
            logger.info(f"  Confidence after TTT:  {new_fp_confidence_after.mean():.3f} ± {new_fp_confidence_after.std():.3f}")
            logger.info(f"  Confidence change: {new_fp_confidence_after.mean() - new_fp_confidence_before.mean():.3f}")
            logger.info(f"  → These were UNCERTAIN cases (conf ~{new_fp_confidence_before.mean():.3f}) that became CONFIDENT (conf ~{new_fp_confidence_after.mean():.3f})")
        else:
            logger.info(f"\n✅ No new false positives created by TTT")
        
        # Analyze fixed false positives
        fixed_fp_confidence_before = np.array([])
        fixed_fp_confidence_after = np.array([])
        if len(fixed_fp_indices) > 0:
            fixed_fp_confidence_before = attack_prob_before_np[fixed_fp_indices]
            fixed_fp_confidence_after = attack_prob_after_np[fixed_fp_indices]
            
            logger.info(f"\n📉 Fixed False Positives (Attack → Normal after TTT):")
            logger.info(f"  Count: {len(fixed_fp_indices)}")
            logger.info(f"  Confidence before TTT: {fixed_fp_confidence_before.mean():.3f} ± {fixed_fp_confidence_before.std():.3f}")
            logger.info(f"  Confidence after TTT:  {fixed_fp_confidence_after.mean():.3f} ± {fixed_fp_confidence_after.std():.3f}")
            logger.info(f"  → These were incorrectly predicted as attacks, TTT corrected them")
        else:
            logger.info(f"\n📉 No false positives were fixed by TTT")
        
        # Analyze new true positives
        new_tp_confidence_before = np.array([])
        new_tp_confidence_after = np.array([])
        if len(new_tp_indices) > 0:
            new_tp_confidence_before = attack_prob_before_np[new_tp_indices]
            new_tp_confidence_after = attack_prob_after_np[new_tp_indices]
            
            logger.info(f"\n✅ New True Positives (Attack correctly detected after TTT):")
            logger.info(f"  Count: {len(new_tp_indices)}")
            logger.info(f"  Confidence before TTT: {new_tp_confidence_before.mean():.3f} ± {new_tp_confidence_before.std():.3f}")
            logger.info(f"  Confidence after TTT:  {new_tp_confidence_after.mean():.3f} ± {new_tp_confidence_after.std():.3f}")
            logger.info(f"  → These were missed attacks, TTT now correctly detects them")
        else:
            logger.info(f"\n📊 No new true positives created by TTT")
        
        # Analyze new false negatives
        new_fn_confidence_before = np.array([])
        new_fn_confidence_after = np.array([])
        if len(new_fn_indices) > 0:
            new_fn_confidence_before = attack_prob_before_np[new_fn_indices]
            new_fn_confidence_after = attack_prob_after_np[new_fn_indices]
            
            logger.info(f"\n⚠️  New False Negatives (Attack → Normal after TTT):")
            logger.info(f"  Count: {len(new_fn_indices)}")
            logger.info(f"  Confidence before TTT: {new_fn_confidence_before.mean():.3f} ± {new_fn_confidence_before.std():.3f}")
            logger.info(f"  Confidence after TTT:  {new_fn_confidence_after.mean():.3f} ± {new_fn_confidence_after.std():.3f}")
            logger.info(f"  → These were correctly detected attacks, TTT now misses them")
        else:
            logger.info(f"\n✅ No new false negatives created by TTT")
        
        # Create plots
        logger.info("📈 Generating plots...")
        output_dir = Path("performance_plots")
        output_dir.mkdir(exist_ok=True)
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('False Positive Pattern Analysis: Before vs After TTT', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Confusion Matrix Comparison
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(cm_before, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.figure.colorbar(im1, ax=ax1)
        ax1.set(xticks=np.arange(cm_before.shape[1]),
               yticks=np.arange(cm_before.shape[0]),
               xticklabels=['Normal', 'Attack'],
               yticklabels=['Normal', 'Attack'],
               title='Confusion Matrix: Before TTT',
               ylabel='True Label',
               xlabel='Predicted Label')
        thresh = cm_before.max() / 2.
        for i in range(cm_before.shape[0]):
            for j in range(cm_before.shape[1]):
                ax1.text(j, i, format(cm_before[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm_before[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
        
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(cm_after, interpolation='nearest', cmap=plt.cm.Oranges)
        ax2.figure.colorbar(im2, ax=ax2)
        ax2.set(xticks=np.arange(cm_after.shape[1]),
               yticks=np.arange(cm_after.shape[0]),
               xticklabels=['Normal', 'Attack'],
               yticklabels=['Normal', 'Attack'],
               title='Confusion Matrix: After TTT',
               ylabel='True Label',
               xlabel='Predicted Label')
        thresh = cm_after.max() / 2.
        for i in range(cm_after.shape[0]):
            for j in range(cm_after.shape[1]):
                ax2.text(j, i, format(cm_after[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm_after[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
        
        # Plot 2: Changes in Confusion Matrix
        ax3 = plt.subplot(2, 3, 3)
        changes = [tn_change, fp_change, fn_change, tp_change]
        change_labels = ['TN', 'FP', 'FN', 'TP']
        colors_change = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in changes]
        bars3 = ax3.bar(change_labels, changes, color=colors_change, alpha=0.7, 
                       edgecolor='black', linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('Change', fontsize=11)
        ax3.set_title('Confusion Matrix Changes', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, change in zip(bars3, changes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{change:+d}',
                    ha='center', va='bottom' if change > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        # Plot 3: Confidence Distribution for New False Positives
        ax4 = plt.subplot(2, 3, 4)
        if len(new_fp_indices) > 0:
            ax4.hist(new_fp_confidence_before, bins=20, alpha=0.7, label='Before TTT', 
                    color='blue', edgecolor='black', linewidth=0.5)
            ax4.hist(new_fp_confidence_after, bins=20, alpha=0.7, label='After TTT', 
                    color='orange', edgecolor='black', linewidth=0.5)
            ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
            ax4.set_xlabel('Attack Probability', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title(f'New False Positives (n={len(new_fp_indices)})\nConfidence Distribution', 
                         fontsize=12, fontweight='bold')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No new false positives', ha='center', va='center',
                    fontsize=14, transform=ax4.transAxes)
            ax4.set_title('New False Positives', fontsize=12, fontweight='bold')
        
        # Plot 4: Confidence Distribution for New True Positives
        ax5 = plt.subplot(2, 3, 5)
        if len(new_tp_indices) > 0:
            ax5.hist(new_tp_confidence_before, bins=20, alpha=0.7, label='Before TTT', 
                    color='blue', edgecolor='black', linewidth=0.5)
            ax5.hist(new_tp_confidence_after, bins=20, alpha=0.7, label='After TTT', 
                    color='green', edgecolor='black', linewidth=0.5)
            ax5.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
            ax5.set_xlabel('Attack Probability', fontsize=11)
            ax5.set_ylabel('Frequency', fontsize=11)
            ax5.set_title(f'New True Positives (n={len(new_tp_indices)})\nConfidence Distribution', 
                         fontsize=12, fontweight='bold')
            ax5.legend(loc='best', fontsize=9)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No new true positives', ha='center', va='center',
                    fontsize=14, transform=ax5.transAxes)
            ax5.set_title('New True Positives', fontsize=12, fontweight='bold')
        
        # Plot 5: Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        categories = ['FP', 'TP', 'FN', 'TN']
        before_values = [fp_before, tp_before, fn_before, tn_before]
        after_values = [fp_after, tp_after, fn_after, tn_after]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, before_values, width, label='Before TTT', 
                       color='blue', alpha=0.7, edgecolor='black', linewidth=1)
        bars2 = ax6.bar(x + width/2, after_values, width, label='After TTT', 
                       color='orange', alpha=0.7, edgecolor='black', linewidth=1)
        
        ax6.set_ylabel('Count', fontsize=11)
        ax6.set_title('Confusion Matrix Summary', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "diagnostic_false_positive_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved to: {plot_path}")
        
        # Also save as PDF
        plot_path_pdf = output_dir / "diagnostic_false_positive_analysis.pdf"
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved to: {plot_path_pdf}")
        
        plt.close()
        
        # Save statistics to JSON
        stats = {
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
            "changes": {
                "tn_change": int(tn_change),
                "fp_change": int(fp_change),
                "fn_change": int(fn_change),
                "tp_change": int(tp_change),
                "fp_change_pct": float(fp_change_pct),
                "tp_change_pct": float(tp_change_pct) if tp_before > 0 else None
            },
            "new_false_positives": {
                "count": int(len(new_fp_indices)),
                "confidence_before_mean": float(new_fp_confidence_before.mean()) if len(new_fp_indices) > 0 else None,
                "confidence_before_std": float(new_fp_confidence_before.std()) if len(new_fp_indices) > 0 else None,
                "confidence_after_mean": float(new_fp_confidence_after.mean()) if len(new_fp_indices) > 0 else None,
                "confidence_after_std": float(new_fp_confidence_after.std()) if len(new_fp_indices) > 0 else None
            },
            "fixed_false_positives": {
                "count": int(len(fixed_fp_indices)),
                "confidence_before_mean": float(fixed_fp_confidence_before.mean()) if len(fixed_fp_indices) > 0 else None,
                "confidence_after_mean": float(fixed_fp_confidence_after.mean()) if len(fixed_fp_indices) > 0 else None
            },
            "new_true_positives": {
                "count": int(len(new_tp_indices)),
                "confidence_before_mean": float(new_tp_confidence_before.mean()) if len(new_tp_indices) > 0 else None,
                "confidence_after_mean": float(new_tp_confidence_after.mean()) if len(new_tp_indices) > 0 else None
            },
            "new_false_negatives": {
                "count": int(len(new_fn_indices)),
                "confidence_before_mean": float(new_fn_confidence_before.mean()) if len(new_fn_indices) > 0 else None,
                "confidence_after_mean": float(new_fn_confidence_after.mean()) if len(new_fn_indices) > 0 else None
            },
            "sample_size": int(len(X_test))
        }
        
        stats_path = output_dir / "diagnostic_false_positive_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"✅ Statistics saved to: {stats_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DIAGNOSTIC TEST 3 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in false positive analysis: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_false_positive_analysis()
    sys.exit(0 if success else 1)

