"""
Quick diagnostic: Check what config will actually be loaded at runtime
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("Runtime Configuration Check")
print("=" * 80)

# Simulate what main.py does
from config_loader import get_dataset_config

config = get_dataset_config()

print(f"\n📋 Configuration that WILL be used:")
print(f"   Dataset: {config.data_path}")
print(f"   n_way: {config.n_way}")
print(f"   k_shot: {config.k_shot}")
print(f"   n_query: {config.n_query}")
print(f"   num_meta_tasks: {config.num_meta_tasks}")

# Calculate expected episode structure
support_total = 100 + config.k_shot  # ~100 Normal + k_shot Attack
query_total = config.n_query * config.n_way
total_per_episode = support_total + query_total
episodes_per_epoch = 50000 // total_per_episode

print(f"\n📊 Expected Training Characteristics:")
print(f"   Support samples: ~{support_total}")
print(f"   Query samples: {query_total}")
print(f"   Total per episode: ~{total_per_episode}")
print(f"   Episodes per epoch: ~{episodes_per_epoch}")

print(f"\n✅ Verdict:")
if config.n_query == 304:
    print(f"   n_query=304 WILL be used ✅")
    print(f"   Expected episodes per epoch: ~{episodes_per_epoch}")
    print(f"   Support:Query ratio: {support_total/query_total:.2f}:1 (balanced)")
elif config.n_query == 20:
    print(f"   n_query=20 will be used ❌")
    print(f"   Expected episodes per epoch: ~{episodes_per_epoch}")
    print(f"   Support:Query ratio: {support_total/query_total:.2f}:1 (imbalanced)")
else:
    print(f"   n_query={config.n_query}")
    print(f"   Expected episodes per epoch: ~{episodes_per_epoch}")

print("=" * 80)
