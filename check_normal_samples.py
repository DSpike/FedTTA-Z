#!/usr/bin/env python3
"""Quick script to check Normal sample counts"""
from config import get_config
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator
import torch
import numpy as np

config = get_config()
print("=" * 80)
print("NORMAL SAMPLE COUNT CHECK")
print("=" * 80)

# Step 1: After preprocessing
print("\n1. After Preprocessing:")
preprocessor = UNSWPreprocessor()
data = preprocessor.preprocess_unsw_dataset(zero_day_attack=config.zero_day_attack)
train_data = torch.tensor(data['X_train'])
train_labels = torch.tensor(data['y_train'])

normal_count = (train_labels == 0).sum().item()
attack_count = (train_labels == 1).sum().item()
total = len(train_labels)

print(f"   Total training samples: {total:,}")
print(f"   Normal samples (Class 0): {normal_count:,} ({100*normal_count/total:.2f}%)")
print(f"   Attack samples (Class 1): {attack_count:,} ({100*attack_count/total:.2f}%)")

# Step 2: After distribution to clients
print("\n2. After Dirichlet Distribution to Clients:")
# Initialize coordinator properly
coordinator = SimpleFedAVGCoordinator(config, train_data, train_labels)
coordinator.distribute_data(train_data, train_labels, train_multiclass_labels=torch.tensor(data['y_train_multiclass']))

normal_counts = []
for client in coordinator.clients:
    if hasattr(client, 'train_labels'):
        client_normal = (client.train_labels == 0).sum().item()
        normal_counts.append(client_normal)
        client_total = len(client.train_labels)
        normal_pct = 100 * client_normal / client_total if client_total > 0 else 0
        print(f"   {client.client_id}: {client_normal:,} Normal samples ({normal_pct:.1f}% of {client_total:,} total)")

# Step 3: Summary
print("\n3. Summary:")
normal_needed_per_task = int(2 * config.k_shot * config.support_normal_ratio)
print(f"   Required Normal samples per meta-task: {normal_needed_per_task}")
print(f"   Number of meta-tasks per client: {config.num_meta_tasks}")
print(f"   Minimum Normal samples per client: {min(normal_counts):,}")
print(f"   Maximum Normal samples per client: {max(normal_counts):,}")
print(f"   Average Normal samples per client: {np.mean(normal_counts):.0f}")
print(f"   Clients with enough Normal samples: {sum(1 for n in normal_counts if n >= normal_needed_per_task)}/{len(normal_counts)}")

print("\n" + "=" * 80)

