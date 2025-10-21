#!/usr/bin/env python3
"""
Fix indentation for the find_optimal_threshold function
"""

import re

def fix_function_indentation():
    """Fix indentation for the find_optimal_threshold function"""
    
    print("🔧 Fixing function indentation...")
    
    # Read the file
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the find_optimal_threshold function specifically
    pattern = r'def find_optimal_threshold\(y_true, y_scores, method=\'balanced\'\):.*?(?=\n@dataclass|\n\n@dataclass|\Z)'
    
    replacement = '''def find_optimal_threshold(y_true, y_scores, method='balanced'):
    """
    Robust threshold optimization that prevents extreme values and ensures valid predictions
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probability scores
        method: Optimization method ('balanced', 'youden', 'precision', 'f1')
    
    Returns:
        optimal_threshold: Best threshold value (clamped between 0.01 and 0.99)
        roc_auc: Area under ROC curve
        fpr, tpr, thresholds: ROC curve data
    """
    # Ensure we have valid probability scores
    y_scores = np.clip(y_scores, 1e-7, 1 - 1e-7)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Remove extreme thresholds to prevent infinite values
    valid_mask = (thresholds > 0.01) & (thresholds < 0.99)
    
    if not np.any(valid_mask):
        # If no valid thresholds, use default
        logger.warning("No valid thresholds found, using default threshold 0.5")
        return 0.5, roc_auc, fpr, tpr, thresholds

    valid_thresholds = thresholds[valid_mask]
    valid_fpr = fpr[valid_mask]
    valid_tpr = tpr[valid_mask]
    
    if method == 'balanced':
        # Use Youden's J statistic as a memory-efficient proxy for F1-score
        youden_j = valid_tpr - valid_fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = valid_thresholds[optimal_idx]
        return optimal_threshold, roc_auc, fpr, tpr, thresholds
    elif method == 'youden':
        # Youden's J statistic: maximize (sensitivity + specificity - 1)
        youden_j = valid_tpr - valid_fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = valid_thresholds[optimal_idx]
        return optimal_threshold, roc_auc, fpr, tpr, thresholds
    elif method == 'precision':
        # Use TPR as a memory-efficient proxy for precision
        optimal_idx = np.argmax(valid_tpr)
        optimal_threshold = valid_thresholds[optimal_idx]
        return optimal_threshold, roc_auc, fpr, tpr, thresholds
    elif method == 'f1':
        # Use Youden's J statistic as a memory-efficient proxy for F1-score
        youden_j = valid_tpr - valid_fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = valid_thresholds[optimal_idx]
        return optimal_threshold, roc_auc, fpr, tpr, thresholds
    else:
        # Default to balanced method
        optimal_threshold = 0.5
    
    # Final safety clamp to prevent extreme values
    optimal_threshold = np.clip(optimal_threshold, 0.01, 0.99)
    
    logger.info(f"Memory-efficient optimal threshold found: {optimal_threshold:.4f} (method: {method}, ROC-AUC: {roc_auc:.4f})")
    
    return optimal_threshold, roc_auc, fpr, tpr, thresholds
'''
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the fixed content
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed function indentation!")

if __name__ == "__main__":
    fix_function_indentation()









