"""
Check if the prototype reuse fix is working correctly.
This script examines the latest run log to verify:
1. TTT prototypes are being stored
2. Evaluation is using stored TTT prototypes
3. Performance has improved
"""

import re
import json
from pathlib import Path

print("=" * 80)
print("PROTOTYPE REUSE FIX VERIFICATION")
print("=" * 80)

# 1. Check latest log file for TTT prototype storage
log_file = Path("latest_run.log")
if not log_file.exists():
    print("\n❌ latest_run.log not found - run hasn't completed yet")
    exit(1)

with open(log_file, 'r') as f:
    log_content = f.read()

# 2. Check if TTT prototypes were stored
print("\n1. CHECKING TTT PROTOTYPE STORAGE:")
if "Stored TTT prototypes" in log_content:
    # Extract the shape
    match = re.search(r"Stored TTT prototypes \(shape: ([^)]+)\)", log_content)
    if match:
        shape = match.group(1)
        print(f"   ✅ TTT prototypes stored with shape: {shape}")
    else:
        print(f"   ✅ TTT prototypes stored (shape not found in log)")
else:
    print(f"   ❌ TTT prototypes NOT stored")

# 3. Check if evaluation used stored prototypes
print("\n2. CHECKING EVALUATION PROTOTYPE USAGE:")
if "Using stored TTT prototypes for consistent evaluation" in log_content:
    print(f"   ✅ Evaluation used stored TTT prototypes")
elif "TTT prototypes not found - recomputing from support set" in log_content:
    print(f"   ❌ Evaluation recomputed prototypes (MISMATCH!)")
else:
    print(f"   ⚠️  Could not determine prototype usage in evaluation")

# 4. Extract performance metrics
print("\n3. CHECKING PERFORMANCE RESULTS:")
try:
    metrics_file = Path("performance_plots/performance_metrics_.json")
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        if 'evaluation_results' in data:
            results = data['evaluation_results']

            base_acc = results.get('base_model', {}).get('accuracy', 0)
            ttt_acc = results.get('adapted_model', {}).get('accuracy', 0)

            print(f"   Base Model Accuracy:    {base_acc:.4f} ({base_acc*100:.2f}%)")
            print(f"   TTT Model Accuracy:     {ttt_acc:.4f} ({ttt_acc*100:.2f}%)")

            improvement = ttt_acc - base_acc
            print(f"\n   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

            if improvement < -0.01:
                print(f"   ❌ TTT IS STILL DEGRADING PERFORMANCE!")
                print(f"      → The fix may not be working correctly")
            elif abs(improvement) < 0.001:
                print(f"   ⚠️  TTT SHOWS NO CHANGE (identical performance)")
                print(f"      → Prototypes may still be mismatched")
            elif improvement > 0:
                print(f"   ✅ TTT IS IMPROVING PERFORMANCE!")
                print(f"      → The fix is working correctly!")
        else:
            print(f"   ⚠️  evaluation_results not found in metrics file")
    else:
        print(f"   ⚠️  performance_metrics_.json not found")
except Exception as e:
    print(f"   ❌ Error reading metrics: {e}")

# 5. Check for specific log patterns
print("\n4. DETAILED LOG ANALYSIS:")

# Check TTT adaptation logs
ttt_adaptation_lines = [line for line in log_content.split('\n') if 'Starting TTT adaptation' in line]
if ttt_adaptation_lines:
    print(f"   ✅ TTT adaptation started ({len(ttt_adaptation_lines)} time(s))")
else:
    print(f"   ❌ TTT adaptation not found in logs")

# Check for prototype updates during TTT
prototype_update_lines = [line for line in log_content.split('\n') if 'Prototypes updated at step' in line]
if prototype_update_lines:
    print(f"   ✅ Dynamic prototype updates occurred ({len(prototype_update_lines)} updates)")
else:
    print(f"   ⚠️  No dynamic prototype updates found")

# Check for evaluation section
eval_lines = [line for line in log_content.split('\n') if 'Evaluating adapted model' in line]
if eval_lines:
    print(f"   ✅ Adapted model evaluation found")
else:
    print(f"   ❌ Adapted model evaluation not found")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
