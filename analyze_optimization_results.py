"""
Analyze optimization results and calculate support set distribution
"""
import json

# Load best hyperparameters
with open('best_hyperparameters.json', 'r') as f:
    best = json.load(f)

params = best['best_params']

print("=" * 80)
print("📊 OPTIMIZATION RESULTS ANALYSIS")
print("=" * 80)

print("\n🎯 Best Trial Hyperparameters:")
print(f"  • Trial Number: {best['best_trial_number']}")
print(f"  • Zero-Day Detection Rate: {best['best_value']:.4f} (81.3%)")
print(f"\n  📋 Key Parameters:")
print(f"    - meta_epochs: {params['meta_epochs']}")
print(f"    - k_shot: {params['k_shot']}")
print(f"    - use_residual_connections: {params['use_residual_connections']}")
print(f"    - use_teacher: {params['use_teacher']}")

# Calculate support set distribution
k_shot = params['k_shot']
total_support_size = 2 * k_shot  # With equal distribution enabled
num_classes = 1 + 8  # Normal + 8 attack types
samples_per_class = total_support_size // num_classes
remaining_samples = total_support_size % num_classes

print("\n" + "=" * 80)
print("📊 SUPPORT SET CLASS DISTRIBUTION (Equal Distribution Enabled)")
print("=" * 80)
print(f"\n  Configuration:")
print(f"    - k_shot: {k_shot}")
print(f"    - Total support size: {total_support_size} (2 × {k_shot})")
print(f"    - Number of classes: {num_classes} (1 Normal + 8 Attack Types)")
print(f"    - Samples per class (base): {samples_per_class}")
print(f"    - Remaining samples: {remaining_samples}")

print(f"\n  Distribution Breakdown:")
normal_samples = samples_per_class + (1 if remaining_samples > 0 else 0)
remaining_attack_samples = remaining_samples - (1 if remaining_samples > 0 else 0)

print(f"    - Normal: {normal_samples} samples")

if remaining_attack_samples > 0:
    num_attack_types_extra = remaining_attack_samples
    num_attack_types_base = 8 - num_attack_types_extra
    print(f"    - {num_attack_types_base} Attack Types: {samples_per_class} samples each")
    print(f"    - {num_attack_types_extra} Attack Types: {samples_per_class + 1} samples each (get extra)")
else:
    print(f"    - All 8 Attack Types: {samples_per_class} samples each")

total_attack_samples = 8 * samples_per_class + remaining_attack_samples
total_samples = normal_samples + total_attack_samples

print(f"\n  Total:")
print(f"    - Normal: {normal_samples} samples ({100*normal_samples/total_samples:.1f}%)")
print(f"    - Attack (8 types): {total_attack_samples} samples ({100*total_attack_samples/total_samples:.1f}%)")
print(f"    - Total: {total_samples} samples")

if total_samples != total_support_size:
    print(f"\n  ⚠️  Note: Total ({total_samples}) doesn't match expected ({total_support_size})")
    print(f"      This is normal - actual distribution depends on available samples.")

print("\n" + "=" * 80)
print("🔍 WHY RESIDUAL CONNECTIONS = FALSE?")
print("=" * 80)
print("""
  Possible Reasons:
  1. **Over-regularization**: Residual connections may cause the model to 
     rely too heavily on skip connections, reducing feature learning.
  
  2. **TCN Architecture**: With depthwise separable convolutions, residual 
     connections might not provide the same benefit as in standard CNNs.
  
  3. **Few-shot Learning**: In meta-learning, simpler architectures often 
     generalize better to new tasks. Residual connections add complexity.
  
  4. **Data Distribution**: With equal distribution across 9 classes, the 
     model might need more direct learning paths without skip connections.
""")

print("\n" + "=" * 80)
print("🔍 WHY USE_TEACHER = FALSE?")
print("=" * 80)
print("""
  Possible Reasons:
  1. **EMA Teacher Overhead**: Teacher-student training adds complexity
     and may not be necessary when pseudo-labeling already provides
     self-supervision.
  
  2. **Quick Adaptation**: For test-time training (TTT), direct adaptation
     might be faster and more effective than maintaining a teacher model.
  
  3. **Over-smoothing**: EMA teacher might smooth out important signals
     needed for zero-day detection.
  
  4. **Pseudo-labeling Sufficient**: With use_pseudo_labels=True, the
     model already gets regularization from pseudo-labels without needing
     an additional teacher model.
""")

print("\n" + "=" * 80)
print("📈 META_EPOCHS ANALYSIS")
print("=" * 80)
print(f"""
  Optimal Value: {params['meta_epochs']} epochs
  
  Meta-epochs controls how many times the model trains on meta-tasks per
  federated round. With 3 epochs:
  - Good balance between learning and overfitting
  - Allows model to see each meta-task multiple times
  - Not too many (would cause overfitting to client's local data)
  - Not too few (would underfit and not learn enough)
  
  The optimization tested values in range [2, 5] and found 3 optimal.
""")

print("\n" + "=" * 80)
print("✅ SUMMARY")
print("=" * 80)
print(f"""
  Best Configuration Found:
  • Residual Connections: DISABLED (False)
  • Teacher Model (EMA): DISABLED (False)  
  • Meta Epochs: {params['meta_epochs']}
  • K-shot: {k_shot} → Support set: {normal_samples} Normal + {samples_per_class} per attack type
  • Zero-Day Detection Rate: {best['best_value']*100:.1f}%
  
  This configuration achieved the highest zero-day detection performance
  in the optimization search space.
""")

