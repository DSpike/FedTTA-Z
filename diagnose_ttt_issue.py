"""
Diagnostic script to understand why TTT is not improving the base model.
"""

import torch
import json
from config import config

print("="*80)
print("TTT CONFIGURATION DIAGNOSIS")
print("="*80)

# 1. Check TTT configuration
print("\n1. TTT CONFIGURATION PARAMETERS:")
print(f"   ttt_base_steps: {config.ttt_base_steps}")
print(f"   ttt_lr: {config.ttt_lr}")
print(f"   ttt_l2_reg_weight: {config.ttt_l2_reg_weight}")
print(f"   pseudo_threshold: {config.pseudo_threshold}")
print(f"   use_pseudo_labels: {config.use_pseudo_labels}")
print(f"   entropy_weight: {config.entropy_weight}")
print(f"   pseudo_weight: {config.pseudo_weight}")

# 2. Check if fixes are correct
print("\n2. VERIFYING OUR FIXES:")
if config.ttt_base_steps == 200:
    print(f"   ✅ TTT steps increased: {config.ttt_base_steps}")
else:
    print(f"   ❌ TTT steps NOT increased: {config.ttt_base_steps} (expected 200)")

if config.ttt_l2_reg_weight == 0.0001:
    print(f"   ✅ L2 reg reduced: {config.ttt_l2_reg_weight}")
else:
    print(f"   ❌ L2 reg NOT reduced: {config.ttt_l2_reg_weight} (expected 0.0001)")

if config.pseudo_threshold == 0.75:
    print(f"   ✅ Pseudo threshold lowered: {config.pseudo_threshold}")
else:
    print(f"   ❌ Pseudo threshold NOT lowered: {config.pseudo_threshold} (expected 0.75)")

# 3. Check performance results
print("\n3. LATEST PERFORMANCE RESULTS:")
try:
    with open('performance_plots/performance_metrics_.json') as f:
        data = json.load(f)

    if 'evaluation_results' in data:
        results = data['evaluation_results']

        if 'base_model' in results:
            base = results['base_model']
            print(f"   Base Model Accuracy: {base.get('accuracy', 'N/A'):.4f}")
            print(f"   Base Model ZDR: {base.get('zero_day_detection_rate', 'N/A'):.4f}")

        if 'adapted_model' in results:
            adapted = results['adapted_model']
            print(f"   TTT Model Accuracy: {adapted.get('accuracy', 'N/A'):.4f}")
            print(f"   TTT Model ZDR: {adapted.get('zero_day_detection_rate', 'N/A'):.4f}")

        if 'base_model' in results and 'adapted_model' in results:
            base_acc = results['base_model'].get('accuracy', 0)
            ttt_acc = results['adapted_model'].get('accuracy', 0)
            improvement = ttt_acc - base_acc
            print(f"\n   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

            if improvement < 0:
                print(f"   ❌ TTT IS DEGRADING PERFORMANCE!")
            elif improvement < 0.01:
                print(f"   ⚠️  TTT IS NOT IMPROVING SIGNIFICANTLY")
            else:
                print(f"   ✅ TTT IS IMPROVING PERFORMANCE")

except Exception as e:
    print(f"   ❌ Could not load results: {e}")

# 4. Check model architecture
print("\n4. MODEL ARCHITECTURE CHECK:")
from models.transductive_fewshot_model import TransductiveLearner

try:
    model = TransductiveLearner(config)
    print(f"   ✅ Model has forward_with_prototypes: {hasattr(model, 'forward_with_prototypes')}")
    print(f"   ✅ Model has compute_prototypes: {hasattr(model, 'compute_prototypes')}")

    # Test forward output
    test_input = torch.randn(10, 22, 43)  # (batch, seq_len, features)
    output = model(test_input)
    print(f"   forward() output shape: {output.shape}")

    if output.shape[1] == 2:
        print(f"   ⚠️  WARNING: forward() returns logits, not embeddings!")
        print(f"   This means the model is NOT prototype-based as expected")
    elif output.shape[1] > 10:
        print(f"   ✅ forward() returns embeddings (shape: {output.shape})")

except Exception as e:
    print(f"   ❌ Model check failed: {e}")

# 5. Check coordinator code
print("\n5. COORDINATOR CODE CHECK:")
import inspect
from coordinators.centralized_coordinator import CentralizedCoordinator

try:
    # Check if adapt_to_test_data has prototype support
    source = inspect.getsource(CentralizedCoordinator.adapt_to_test_data)

    if 'forward_with_prototypes' in source:
        print(f"   ✅ adapt_to_test_data has prototype support")
    else:
        print(f"   ❌ adapt_to_test_data does NOT have prototype support")

    if 'prototype_update_interval' in source:
        print(f"   ✅ adapt_to_test_data has dynamic prototype updating")
    else:
        print(f"   ❌ adapt_to_test_data does NOT have dynamic prototype updating")

    if 'momentum = 0.8' in source or 'module.momentum = 0.8' in source:
        print(f"   ✅ BatchNorm momentum is set to 0.8")
    elif 'momentum = 0.1' in source or 'module.momentum = 0.1' in source:
        print(f"   ❌ BatchNorm momentum is still 0.1 (should be 0.8)")
    else:
        print(f"   ⚠️  Could not find BatchNorm momentum setting")

except Exception as e:
    print(f"   ❌ Coordinator check failed: {e}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
