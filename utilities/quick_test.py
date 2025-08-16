#!/usr/bin/env python3
"""
QUICK GPU TEST - Fast diagnostic without interruptions
"""

import torch

print("🚀 QUICK GPU TEST STARTING...")
print("=" * 40)

# Test 1: CUDA availability
print("1️⃣ CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("2️⃣ GPU Name:", torch.cuda.get_device_name(0))
    print("3️⃣ GPU Memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("4️⃣ CUDA Version:", torch.version.cuda)
    
    # Test 5: Quick GPU operation
    try:
        x = torch.randn(100, 100, device='cuda')
        y = x * 2
        print("5️⃣ GPU Operation: ✅ SUCCESS")
        print("6️⃣ GPU Memory Used:", f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        print("5️⃣ GPU Operation: ❌ FAILED -", e)
else:
    print("❌ CUDA NOT AVAILABLE!")

print("🎉 QUICK TEST COMPLETE!")
