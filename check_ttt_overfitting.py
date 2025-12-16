#!/usr/bin/env python3
"""
TTT Overfitting Diagnostic
Checks if TTT adaptation is overfitting to test data
"""

import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def check_ttt_overfitting(
    base_results: Dict[str, Any],
    ttt_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    zero_day_mask: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Check if TTT is overfitting by comparing base and TTT performance
    
    Args:
        base_results: Base model evaluation results
        ttt_results: TTT model evaluation results
        X_test: Test features
        y_test: Test labels
        zero_day_mask: Boolean mask for zero-day samples
        threshold: Performance drop threshold (default 0.05 = 5%)
        
    Returns:
        Dictionary with overfitting analysis
    """
    try:
        flags = []
        severity = "LOW"
        status = "healthy"
        
        # Extract metrics
        base_accuracy = base_results.get('accuracy', 0.0)
        ttt_accuracy = ttt_results.get('accuracy', 0.0)
        
        base_f1 = base_results.get('f1_score', 0.0)
        ttt_f1 = ttt_results.get('f1_score', 0.0)
        
        base_zdr = base_results.get('zero_day_detection_rate', 0.0)
        ttt_zdr = ttt_results.get('zero_day_detection_rate', 0.0)
        
        base_far = base_results.get('false_alarm_rate', 0.0)
        ttt_far = ttt_results.get('false_alarm_rate', 0.0)
        
        # Get zero-day specific metrics
        base_zero_day_accuracy = base_results.get('zero_day_accuracy', 0.0)
        ttt_zero_day_accuracy = ttt_results.get('zero_day_accuracy', 0.0)
        
        # Check 1: Overall Performance Drop
        accuracy_drop = base_accuracy - ttt_accuracy
        f1_drop = base_f1 - ttt_f1
        
        if accuracy_drop > threshold:
            flags.append("overall_accuracy_drop")
            severity = "MEDIUM"
            status = "overfitting"
        
        if f1_drop > threshold:
            flags.append("f1_score_drop")
            if severity == "LOW":
                severity = "MEDIUM"
            status = "overfitting"
        
        # Check 2: Zero-Day Discrepancy
        # If zero-day accuracy is much higher than overall accuracy, TTT may be overfitting to zero-day
        if ttt_zero_day_accuracy > ttt_accuracy + 0.10:  # 10% threshold
            flags.append("zero_day_discrepancy")
            if severity == "LOW":
                severity = "MEDIUM"
            status = "overfitting"
        
        # Check 3: False Alarm Rate Increase
        far_increase = ttt_far - base_far
        if far_increase > 0.02:  # 2% increase
            flags.append("false_alarm_increase")
            if severity == "LOW":
                severity = "LOW"
            # Don't change status to overfitting for FAR alone
        
        # Check 4: Zero-Day Performance Drop (unexpected)
        zdr_drop = base_zdr - ttt_zdr
        if zdr_drop > threshold:
            flags.append("zero_day_detection_drop")
            if severity == "LOW":
                severity = "MEDIUM"
            status = "overfitting"
        
        # Check 5: Extreme Zero-Day Improvement (may indicate overfitting)
        zdr_improvement = ttt_zdr - base_zdr
        if zdr_improvement > 0.30 and ttt_accuracy < base_accuracy - 0.05:
            # Large ZDR improvement but overall accuracy drop
            flags.append("extreme_zero_day_improvement")
            if severity == "LOW" or severity == "MEDIUM":
                severity = "HIGH"
            status = "overfitting"
        
        # Determine final severity
        if len(flags) >= 3:
            severity = "HIGH"
        elif len(flags) >= 2:
            if severity != "HIGH":
                severity = "MEDIUM"
        
        # Normal performance metrics
        normal_performance = {
            "base_accuracy": float(base_accuracy),
            "ttt_accuracy": float(ttt_accuracy),
            "base_f1": float(base_f1),
            "ttt_f1": float(ttt_f1),
            "base_zdr": float(base_zdr),
            "ttt_zdr": float(ttt_zdr),
            "base_far": float(base_far),
            "ttt_far": float(ttt_far),
            "base_zero_day_accuracy": float(base_zero_day_accuracy),
            "ttt_zero_day_accuracy": float(ttt_zero_day_accuracy)
        }
        
        # Performance changes
        performance_changes = {
            "accuracy_change": float(ttt_accuracy - base_accuracy),
            "f1_change": float(ttt_f1 - base_f1),
            "zdr_change": float(ttt_zdr - base_zdr),
            "far_change": float(ttt_far - base_far),
            "zero_day_accuracy_change": float(ttt_zero_day_accuracy - base_zero_day_accuracy)
        }
        
        return {
            "status": status,
            "severity": severity,
            "flags": flags,
            "normal_performance": normal_performance,
            "performance_changes": performance_changes,
            "threshold_used": float(threshold)
        }
        
    except Exception as e:
        logger.error(f"TTT overfitting check failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "status": "error",
            "severity": "UNKNOWN",
            "flags": [],
            "error": str(e)
        }


def print_overfitting_report(overfitting_analysis: Dict[str, Any]) -> None:
    """
    Print a formatted overfitting diagnostic report
    
    Args:
        overfitting_analysis: Results from check_ttt_overfitting
    """
    status = overfitting_analysis.get('status', 'unknown')
    severity = overfitting_analysis.get('severity', 'UNKNOWN')
    flags = overfitting_analysis.get('flags', [])
    
    status_symbol = "⚠️" if status == 'overfitting' else "✅"
    logger.info(f"{status_symbol} Status: {status.upper()}")
    logger.info(f"   Severity: {severity.upper()}")
    
    if flags:
        logger.warning(f"⚠️ Overfitting Flags Detected:")
        for flag in flags:
            logger.warning(f"   - {flag.replace('_', ' ').title()}")
    
    # Print performance comparison
    normal_perf = overfitting_analysis.get('normal_performance', {})
    changes = overfitting_analysis.get('performance_changes', {})
    
    if normal_perf:
        logger.info("\n📊 Performance Comparison:")
        logger.info(f"   Base Accuracy: {normal_perf.get('base_accuracy', 0):.4f}")
        logger.info(f"   TTT Accuracy:  {normal_perf.get('ttt_accuracy', 0):.4f} ({changes.get('accuracy_change', 0):+.4f})")
        logger.info(f"   Base F1:       {normal_perf.get('base_f1', 0):.4f}")
        logger.info(f"   TTT F1:       {normal_perf.get('ttt_f1', 0):.4f} ({changes.get('f1_change', 0):+.4f})")
        logger.info(f"   Base ZDR:     {normal_perf.get('base_zdr', 0):.4f}")
        logger.info(f"   TTT ZDR:      {normal_perf.get('ttt_zdr', 0):.4f} ({changes.get('zdr_change', 0):+.4f})")
