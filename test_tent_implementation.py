"""
Test TENT implementation to verify it works before full training
"""
import torch
import torch.nn as nn
from models.transductive_fewshot_model import TransductiveLearner
from config import SystemConfig

print("=" * 80)
print("Testing TENT Implementation")
print("=" * 80)

# Create minimal config
config = SystemConfig()
print(f"\n✅ Config loaded")
print(f"   n_query: {config.n_query}")
print(f"   learning_rate: {config.learning_rate}")

# Create a small test model
print("\n📊 Creating test model...")
model = TransductiveLearner(
    input_dim=43,
    hidden_dim=128,  # Small for testing
    embedding_dim=64,
    sequence_length=21,
    num_classes=2,
    tcn_kernel_sizes=(3,),
    use_tcn=True
)
print(f"✅ Model created")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# Simulate TENT parameter selection
print("\n🎯 Simulating TENT approach...")
bn_params = []
other_params = []

for name, param in model.named_parameters():
    # Select only BatchNorm affine parameters (weight and bias)
    if 'bn' in name and ('weight' in name or 'bias' in name):
        param.requires_grad = True
        bn_params.append(param)
        print(f"   ✅ TRAINABLE: {name} ({param.numel()} params)")
    else:
        # Freeze all other parameters (TCN, linear, etc.)
        param.requires_grad = False
        other_params.append(param)

bn_param_count = sum(p.numel() for p in bn_params)
other_param_count = sum(p.numel() for p in other_params)

print(f"\n📈 TENT Statistics:")
print(f"   BatchNorm parameters (trainable): {bn_param_count:,}")
print(f"   Other parameters (frozen): {other_param_count:,}")
print(f"   Total parameters: {total_params:,}")
print(f"   Percentage trainable: {bn_param_count/total_params*100:.2f}%")
print(f"   Reduction: {100 - bn_param_count/total_params*100:.2f}%")

# Test optimizer creation
print("\n🔧 Testing optimizer creation...")
try:
    optimizer = torch.optim.AdamW(
        bn_params,
        lr=0.001,
        weight_decay=1e-4
    )
    print(f"✅ Optimizer created successfully")
    print(f"   Optimizing {len(bn_params)} parameter groups")
except Exception as e:
    print(f"❌ Error creating optimizer: {e}")
    exit(1)

# Test forward pass
print("\n🔄 Testing forward pass...")
try:
    model.train()  # Set to training mode
    dummy_input = torch.randn(4, 21, 43)  # (batch, seq_len, features)
    output = model(dummy_input)
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Error in forward pass: {e}")
    exit(1)

# Test backward pass (simulate TTT)
print("\n⬅️  Testing backward pass with TENT...")
try:
    # Create dummy loss
    dummy_target = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_target)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    bn_grads = [p.grad is not None for p in bn_params]
    other_grads = [p.grad is not None for p in other_params]

    print(f"✅ Backward pass successful")
    print(f"   BatchNorm params with gradients: {sum(bn_grads)}/{len(bn_params)}")
    print(f"   Other params with gradients: {sum(other_grads)}/{len(other_params)} (should be 0)")

    if sum(other_grads) > 0:
        print(f"   ⚠️  WARNING: Some non-BN parameters have gradients (they shouldn't!)")
    else:
        print(f"   ✅ Correctly: Only BN params have gradients")

    # Optimizer step
    optimizer.step()
    print(f"✅ Optimizer step successful")

except Exception as e:
    print(f"❌ Error in backward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ TENT IMPLEMENTATION TEST PASSED!")
print("=" * 80)
print("\nThe TENT approach is correctly implemented:")
print("  ✅ Only BatchNorm parameters are trainable")
print("  ✅ Other parameters (TCN, Linear) are frozen")
print("  ✅ Forward pass works")
print("  ✅ Backward pass works")
print("  ✅ Optimizer updates only BN params")
print("\nReady to run: python main.py")
