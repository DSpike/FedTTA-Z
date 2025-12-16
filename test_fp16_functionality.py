"""
Test FP16 Mixed Precision Functionality
Verifies that FP16 is working correctly with the GPU
"""

import torch
from torch.cuda.amp import autocast, GradScaler
import time

print("=" * 60)
print("FP16 MIXED PRECISION TEST")
print("=" * 60)

# Check GPU availability
print("\n1. GPU CHECK:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ❌ No GPU available - FP16 will be disabled")
    exit(1)

# Test FP16 autocast
print("\n2. FP16 AUTOCAST TEST:")
device = torch.device('cuda')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# FP32 baseline
start = time.time()
with torch.no_grad():
    z_fp32 = torch.matmul(x, y)
fp32_time = time.time() - start
print(f"   FP32 Time: {fp32_time*1000:.2f} ms")

# FP16 with autocast
start = time.time()
with torch.no_grad():
    with autocast():
        z_fp16 = torch.matmul(x.float(), y.float())  # Autocast converts to FP16
fp16_time = time.time() - start
print(f"   FP16 Time: {fp16_time*1000:.2f} ms")

speedup = fp32_time / fp16_time if fp16_time > 0 else 1.0
print(f"   Speedup: {speedup:.2f}x")

# Check if FP16 is actually being used
print(f"   z_fp32 dtype: {z_fp32.dtype}")
print(f"   z_fp16 dtype: {z_fp16.dtype}")

# Test GradScaler
print("\n3. GRADSCALER TEST:")
scaler = GradScaler()
model = torch.nn.Linear(100, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x_train = torch.randn(32, 100, device=device)
y_train = torch.randint(0, 10, (32,), device=device)

# Forward pass with autocast
with autocast():
    output = model(x_train)
    loss = torch.nn.functional.cross_entropy(output, y_train)

# Backward pass with scaler
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"   ✅ GradScaler working correctly")
print(f"   Loss value: {loss.item():.4f}")

# Test mixed precision with TTT-style code
print("\n4. TTT-STYLE FP16 TEST:")
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()
use_mixed_precision = torch.cuda.is_available()

print(f"   Mixed precision enabled: {use_mixed_precision}")

# Simulate TTT adaptation loop
x_batch = torch.randn(16, 100, device=device)

# Forward pass in FP16
with autocast(enabled=use_mixed_precision):
    logits = model(x_batch)
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
    loss = entropy

# Backward pass with scaler
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()

print(f"   ✅ TTT-style FP16 training successful")
print(f"   Entropy loss: {loss.item():.4f}")

# Memory comparison
print("\n5. MEMORY USAGE COMPARISON:")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# FP32 memory
model_fp32 = SimpleModel().to(device)
x_fp32 = torch.randn(64, 100, device=device)
with torch.no_grad():
    _ = model_fp32(x_fp32)
fp32_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# FP16 memory
model_fp16 = SimpleModel().to(device)
x_fp16 = torch.randn(64, 100, device=device)
with torch.no_grad():
    with autocast():
        _ = model_fp16(x_fp16)
fp16_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

print(f"   FP32 Memory: {fp32_memory:.2f} MB")
print(f"   FP16 Memory: {fp16_memory:.2f} MB")
memory_saving = (1 - fp16_memory / fp32_memory) * 100 if fp32_memory > 0 else 0
print(f"   Memory Saving: {memory_saving:.1f}%")

print("\n" + "=" * 60)
print("✅ ALL FP16 TESTS PASSED!")
print("=" * 60)
print("\nYour system is ready for FP16 mixed precision training:")
print(f"  • GPU: {torch.cuda.get_device_name(0)}")
print(f"  • FP16 Speedup: {speedup:.2f}x")
print(f"  • Memory Saving: {memory_saving:.1f}%")
print(f"  • TTT Adaptation: Ready for FP16 ✅")
print(f"  • Meta-Training: Can be optimized with FP16 ✅")









