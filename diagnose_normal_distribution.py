#!/usr/bin/env python3
"""
Diagnostic script to check Normal sample distribution across clients
"""
import sys
sys.path.append('.')

from config import get_config
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
from coordinators.simple_fedavg_coordinator import SimpleFedAVGCoordinator
import torch
import numpy as np

def diagnose_normal_distribution():
    """Check why clients have insufficient Normal samples"""
    
    print("=" * 80)
    print("DIAGNOSTIC: Normal Sample Distribution Across Clients")
    print("=" * 80)
    
    # Load config
    config = get_config()
    
    print(f"\n📊 Configuration:")
    print(f"   dirichlet_alpha: {config.dirichlet_alpha}")
    print(f"   num_clients: {config.num_clients}")
    print(f"   k_shot: {config.k_shot}")
    print(f"   support_normal_ratio: {config.support_normal_ratio}")
    print(f"   num_meta_tasks: {config.num_meta_tasks}")
    
    # Preprocess data
    print(f"\n📥 Loading and preprocessing data...")
    preprocessor = UNSWPreprocessor()
    train_data, train_labels, train_multiclass, test_data, test_labels, test_multiclass = preprocessor.preprocess_and_split(
        zero_day_attack=config.zero_day_attack
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total training samples: {len(train_data):,}")
    
    # Check Normal samples in training data
    normal_mask = train_labels == 0
    attack_mask = train_labels == 1
    normal_count = normal_mask.sum().item()
    attack_count = attack_mask.sum().item()
    
    print(f"   Normal samples (Class 0): {normal_count:,} ({100*normal_count/len(train_data):.1f}%)")
    print(f"   Attack samples (Class 1): {attack_count:,} ({100*attack_count/len(train_data):.1f}%)")
    
    # Create coordinator
    print(f"\n🔧 Creating coordinator with {config.num_clients} clients...")
    coordinator = SimpleFedAVGCoordinator(config, train_data, train_labels, train_multiclass)
    
    # Distribute data
    print(f"\n📦 Distributing data using Dirichlet (α={config.dirichlet_alpha})...")
    coordinator.distribute_data(train_data, train_labels, train_multiclass_labels=train_multiclass)
    
    # Check client distributions
    print(f"\n" + "=" * 80)
    print("CLIENT NORMAL SAMPLE DISTRIBUTION:")
    print("=" * 80)
    
    min_normal = float('inf')
    max_normal = 0
    total_normal_per_client = []
    
    for client in coordinator.clients:
        if hasattr(client, 'train_labels'):
            client_normal_mask = client.train_labels == 0
            client_normal_count = client_normal_mask.sum().item()
            client_attack_mask = client.train_labels == 1
            client_attack_count = client_attack_mask.sum().item()
            client_total = len(client.train_labels)
            
            total_normal_per_client.append(client_normal_count)
            min_normal = min(min_normal, client_normal_count)
            max_normal = max(max_normal, client_normal_count)
            
            normal_pct = 100 * client_normal_count / client_total if client_total > 0 else 0
            
            # Check if enough for meta-tasks
            normal_needed_per_task = int(2 * config.k_shot * config.support_normal_ratio)
            max_normal_needed = config.num_meta_tasks * normal_needed_per_task
            enough_samples = "✅" if client_normal_count >= normal_needed_per_task else "⚠️"
            
            print(f"\n{client.client_id}:")
            print(f"   Total samples: {client_total:,}")
            print(f"   Normal samples: {client_normal_count:,} ({normal_pct:.1f}%) {enough_samples}")
            print(f"   Attack samples: {client_attack_count:,} ({100-normal_pct:.1f}%)")
            print(f"   Needed per task: {normal_needed_per_task} Normal samples")
            print(f"   Status: {'✅ Enough' if client_normal_count >= normal_needed_per_task else '⚠️ Insufficient'} for meta-tasks")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"   Total Normal samples in dataset: {normal_count:,}")
    print(f"   Normal samples per client (min): {min_normal:,}")
    print(f"   Normal samples per client (max): {max_normal:,}")
    print(f"   Normal samples per client (avg): {np.mean(total_normal_per_client):.0f}")
    print(f"   Normal samples per client (std): {np.std(total_normal_per_client):.0f}")
    print(f"   Required per meta-task: {int(2 * config.k_shot * config.support_normal_ratio)} Normal samples")
    print(f"   Clients with enough samples: {sum(1 for n in total_normal_per_client if n >= int(2 * config.k_shot * config.support_normal_ratio))}/{len(total_normal_per_client)}")
    
    # Check Dirichlet distribution
    print(f"\n" + "=" * 80)
    print("DIRICHLET DISTRIBUTION ANALYSIS:")
    print("=" * 80)
    
    # Simulate Dirichlet distribution
    np.random.seed(42)  # Same seed as coordinator
    dirichlet_dist = np.random.dirichlet([config.dirichlet_alpha] * config.num_clients)
    
    print(f"\nDirichlet proportions (should sum to 1.0):")
    print(f"   Sum: {dirichlet_dist.sum():.6f}")
    print(f"\nExpected Normal samples per client:")
    for i, prop in enumerate(dirichlet_dist):
        expected_normal = int(prop * normal_count)
        actual_normal = total_normal_per_client[i] if i < len(total_normal_per_client) else 0
        diff = actual_normal - expected_normal
        print(f"   Client {i+1}: {prop:.3f} × {normal_count:,} = {expected_normal:,} (actual: {actual_normal:,}, diff: {diff:+d})")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if min_normal < int(2 * config.k_shot * config.support_normal_ratio):
        print(f"\n⚠️  WARNING: Some clients have insufficient Normal samples!")
        print(f"\n   Options to fix:")
        print(f"   1. Increase dirichlet_alpha (currently {config.dirichlet_alpha})")
        print(f"      → Try: dirichlet_alpha = 10.0 (more uniform distribution)")
        print(f"\n   2. Decrease k_shot (currently {config.k_shot})")
        print(f"      → Try: k_shot = 50 (fewer samples per meta-task)")
        print(f"\n   3. Use more Normal samples in dataset")
        print(f"      → Current: {normal_count:,} Normal samples")
        print(f"      → Minimum needed: {config.num_clients * int(2 * config.k_shot * config.support_normal_ratio):,} samples")
        print(f"\n   4. Accept current behavior (code handles it gracefully)")
    else:
        print(f"\n✅ All clients have sufficient Normal samples for meta-tasks!")
    
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    diagnose_normal_distribution()

