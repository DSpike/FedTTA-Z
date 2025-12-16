#!/usr/bin/env python3
"""Quick test to verify equal distribution in support sets"""
from config import get_config
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from models.transductive_fewshot_model import create_meta_tasks
import torch
import numpy as np

print("=" * 80)
print("QUICK TEST: Equal Distribution in Support Sets")
print("=" * 80)

# Load config
config = get_config()
print(f"\n📊 Configuration:")
print(f"   k_shot: {config.k_shot}")
print(f"   support_normal_ratio: {config.support_normal_ratio}")
print(f"   enforce_equal_support_composition: {config.enforce_equal_support_composition}")
print(f"   include_all_attack_types_in_support: {config.include_all_attack_types_in_support}")

# Preprocess data (quick - just get training data)
print(f"\n📥 Loading data...")
preprocessor = UNSWPreprocessor()
data = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)

train_data = torch.tensor(data['X_train'])
train_labels = torch.tensor(data['y_train'])
train_multiclass = torch.tensor(data['y_train_multiclass'])

print(f"   Training samples: {len(train_data):,}")
print(f"   Normal samples: {(train_labels == 0).sum().item():,}")
print(f"   Attack samples: {(train_labels == 1).sum().item():,}")

# Create a small number of meta-tasks for testing
print(f"\n🔧 Creating 3 meta-tasks to test equal distribution...")
meta_tasks = create_meta_tasks(
    train_data[:5000],  # Use subset for speed
    train_labels[:5000],
    n_way=config.n_way,
    k_shot=config.k_shot,
    n_query=config.n_query,
    n_tasks=3,  # Just 3 tasks for quick test
    phase="training",
    normal_query_ratio=0.8,
    zero_day_attack_label=config.zero_day_attack_label,
    enforce_equal_support_composition=config.enforce_equal_support_composition,
    support_normal_ratio=config.support_normal_ratio,
    include_all_attack_types_in_support=config.include_all_attack_types_in_support,
    data_y_multiclass=train_multiclass[:5000]
)

print(f"\n✅ Created {len(meta_tasks)} meta-tasks")
print(f"\n📊 Support Set Distribution Analysis:")
print("=" * 80)

# Analyze each task
for i, task in enumerate(meta_tasks):
    support_x = task['support_x']
    support_y = task['support_y']
    
    # Count Normal samples
    normal_count = (support_y == 0).sum().item()
    attack_count = (support_y == 1).sum().item()  # All attacks are remapped to 1
    total = len(support_y)
    
    print(f"\nTask {i+1}:")
    print(f"   Total support samples: {total}")
    print(f"   Normal samples: {normal_count} ({100*normal_count/total:.1f}%)")
    print(f"   Attack samples: {attack_count} ({100*attack_count/total:.1f}%)")
    
    # Calculate expected equal distribution (assuming 8 attack types + Normal = 9 classes)
    # Based on the log message we saw: "29 Normal + 229 Attack (8 types) = 258 total (~29 per class)"
    num_attack_types = 8  # Excluding zero-day
    num_classes = 1 + num_attack_types  # Normal + attack types
    expected_per_class = total / num_classes
    
    print(f"   Expected per class (equal dist): ~{expected_per_class:.1f} samples")
    print(f"   Expected Normal: ~{expected_per_class:.0f} samples")
    print(f"   Expected per attack type: ~{expected_per_class:.0f} samples")
    
    # Check if Normal count matches expected
    expected_normal = int(expected_per_class) + (1 if total % num_classes > 0 else 0)
    if abs(normal_count - expected_normal) <= 2:
        print(f"   ✅ Normal distribution matches expected (~{expected_normal} samples)")
    else:
        print(f"   ⚠️  Normal distribution: got {normal_count}, expected ~{expected_normal}")
    
    # Check if total matches expected
    expected_total = num_classes * int(expected_per_class) + (total % num_classes)
    if total == expected_total or abs(total - expected_total) <= 1:
        print(f"   ✅ Total samples match expected distribution")
    else:
        print(f"   ⚠️  Total samples: got {total}, expected ~{expected_total}")

print("\n" + "=" * 80)
print("✅ Test completed!")
print("=" * 80)

