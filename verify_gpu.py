#!/usr/bin/env python3
"""Quick script to verify GPU setup"""
import torch
import sys

print("="*70)
print("GPU VERIFICATION SCRIPT")
print("="*70)
print(f"Python Executable: {sys.executable}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("✅ GPU IS AVAILABLE AND READY!")
else:
    print("❌ GPU NOT AVAILABLE - Using CPU")
    print("\nPossible reasons:")
    print("1. Virtual environment not activated")
    print("2. CPU-only PyTorch installed")
    print("\nTo fix:")
    print("   cd C:\\Users\\Dspike\\Documents\\PhD\\TNN\\exp1\\Tgnn")
    print("   ..\\Tgnn_gpu\\Scripts\\activate")
    print("   python verify_gpu.py")
print("="*70)
