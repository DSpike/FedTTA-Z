#!/usr/bin/env python3
"""
Extract and compare Base Model vs TTT Model performance from log file
"""
import re
import sys

log_file = "run_with_fixes_log.txt"

def extract_metrics(log_content):
    """Extract performance metrics from log"""

    # Patterns to find
    patterns = {
        # Base Model
        'base_accuracy': r'Base Model.*?Accuracy[:\s]+([\d\.]+)',
        'base_f1': r'Base Model.*?F1-Score[:\s]+([\d\.]+)',
        'base_precision': r'Base Model.*?Precision[:\s]+([\d\.]+)',
        'base_recall': r'Base Model.*?Recall[:\s]+([\d\.]+)',
        'base_auc_pr': r'Base Model.*?AUC-PR[:\s]+([\d\.]+)',
        'base_roc_auc': r'Base Model.*?ROC AUC[:\s]+([\d\.]+)',
        'base_zdr': r'Base.*Zero-Day Detection Rate[:\s]+([\d\.]+)',
        'base_far': r'Base.*FAR[:\s]+([\d\.]+)',

        # TTT/Adapted Model
        'ttt_accuracy': r'(?:TTT|Adapted Model).*?Accuracy[:\s]+([\d\.]+)',
        'ttt_f1': r'(?:TTT|Adapted Model).*?F1-Score[:\s]+([\d\.]+)',
        'ttt_precision': r'(?:TTT|Adapted Model).*?Precision[:\s]+([\d\.]+)',
        'ttt_recall': r'(?:TTT|Adapted Model).*?Recall[:\s]+([\d\.]+)',
        'ttt_auc_pr': r'(?:TTT|Adapted Model).*?AUC-PR[:\s]+([\d\.]+)',
        'ttt_roc_auc': r'(?:TTT|Adapted Model).*?ROC AUC[:\s]+([\d\.]+)',
        'ttt_zdr': r'(?:TTT|Adapted).*Zero-Day Detection Rate[:\s]+([\d\.]+)',
        'ttt_far': r'(?:TTT|Adapted).*FAR[:\s]+([\d\.]+)',

        # TTT Adaptation Status
        'ttt_failed': r'TTT Adaptation FAILED',
        'param_change': r'Parameter change[:\s]+([\d\.e\-\+]+)',
        'pred_diff': r'Prediction difference[:\s]+([\d\.]+)%',
        'zero_day_samples': r'Zero-day samples[:\s]+(\d+)',
    }

    results = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        if matches:
            if key == 'ttt_failed':
                results[key] = True
            else:
                # Take last occurrence
                results[key] = matches[-1]

    return results

def print_comparison(results):
    """Print formatted comparison"""

    print("=" * 80)
    print("BASE MODEL vs TTT MODEL PERFORMANCE COMPARISON")
    print("=" * 80)

    # Check if TTT failed
    if results.get('ttt_failed'):
        print("\n❌ TTT ADAPTATION FAILED!")
        print("   TTT did not run successfully.")
        return

    # Adaptation status
    if 'param_change' in results:
        param_change = float(results['param_change'])
        print(f"\n🔧 TTT Adaptation Status:")
        if param_change > 0.001:
            print(f"   ✅ Parameter change: {param_change:.6f} (GOOD - model adapted)")
        else:
            print(f"   ⚠️  Parameter change: {param_change:.6f} (too small - minimal adaptation)")

    if 'pred_diff' in results:
        pred_diff = float(results['pred_diff'])
        print(f"   {'✅' if pred_diff > 10 else '⚠️ '} Prediction difference: {pred_diff:.1f}%")

    if 'zero_day_samples' in results:
        zd_samples = int(results['zero_day_samples'])
        print(f"   {'✅' if zd_samples > 0 else '❌'} Zero-day samples: {zd_samples}")

    # Performance metrics
    print("\n📊 OVERALL PERFORMANCE:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Base Model':<15} {'TTT Model':<15} {'Improvement':<15}")
    print("-" * 80)

    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc_pr', 'roc_auc']
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-PR', 'ROC AUC']

    for metric, name in zip(metrics, metric_names):
        base_key = f'base_{metric}'
        ttt_key = f'ttt_{metric}'

        if base_key in results and ttt_key in results:
            base_val = float(results[base_key])
            ttt_val = float(results[ttt_key])
            diff = ttt_val - base_val
            diff_pct = (diff / base_val * 100) if base_val > 0 else 0

            symbol = '✅' if diff > 0 else ('❌' if diff < -0.01 else '➖')
            print(f"{name:<20} {base_val:<15.4f} {ttt_val:<15.4f} {symbol} {diff:+.4f} ({diff_pct:+.1f}%)")

    # Zero-day detection
    print("\n🎯 ZERO-DAY DETECTION:")
    print("-" * 80)

    if 'base_zdr' in results and 'ttt_zdr' in results:
        base_zdr = float(results['base_zdr'])
        ttt_zdr = float(results['ttt_zdr'])
        diff = ttt_zdr - base_zdr
        diff_pct = (diff / base_zdr * 100) if base_zdr > 0 else 0

        print(f"{'Zero-Day Detection Rate':<20} {base_zdr:<15.4f} {ttt_zdr:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)")

    if 'base_far' in results and 'ttt_far' in results:
        base_far = float(results['base_far'])
        ttt_far = float(results['ttt_far'])
        diff = ttt_far - base_far

        symbol = '✅' if diff < 0 else ('❌' if diff > 0.01 else '➖')
        print(f"{'False Alarm Rate':<20} {base_far:<15.4f} {ttt_far:<15.4f} {symbol} {diff:+.4f}")

    print("=" * 80)

    # Summary
    print("\n📋 SUMMARY:")
    improvements = 0
    degradations = 0

    for metric in metrics:
        base_key = f'base_{metric}'
        ttt_key = f'ttt_{metric}'
        if base_key in results and ttt_key in results:
            diff = float(results[ttt_key]) - float(results[base_key])
            if diff > 0.001:
                improvements += 1
            elif diff < -0.001:
                degradations += 1

    print(f"   Metrics improved: {improvements}/{len(metrics)}")
    print(f"   Metrics degraded: {degradations}/{len(metrics)}")

    if improvements > degradations:
        print("\n   ✅ TTT MODEL OUTPERFORMS BASE MODEL!")
    elif improvements == 0 and degradations == 0:
        print("\n   ➖ TTT MODEL SAME AS BASE MODEL (no adaptation)")
    else:
        print("\n   ⚠️  TTT MODEL UNDERPERFORMS BASE MODEL")

    print("=" * 80)

if __name__ == '__main__':
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        results = extract_metrics(log_content)
        print_comparison(results)

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
