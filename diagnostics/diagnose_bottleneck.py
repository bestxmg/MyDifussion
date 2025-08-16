#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT - Find the bottleneck in your Stable Diffusion setup
"""

import torch
import time
import os

def diagnose_gpu():
    print("🔍 DIAGNOSING YOUR STABLE DIFFUSION BOTTLENECK!")
    print("=" * 60)
    
    # 1. Check CUDA availability
    print("1️⃣ CUDA Status:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
    else:
        print("   ❌ CUDA NOT AVAILABLE!")
        return
    
    # 2. Test GPU memory allocation
    print("\n2️⃣ GPU Memory Test:")
    try:
        # Allocate a test tensor on GPU
        test_tensor = torch.randn(1000, 1000, device='cuda')
        print(f"   ✅ GPU Memory Test: Success")
        print(f"   GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Clear test tensor
        del test_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU Memory Test Failed: {e}")
        return
    
    # 3. Test basic PyTorch operations on GPU
    print("\n3️⃣ GPU Computation Test:")
    try:
        start_time = time.time()
        
        # Create tensors on GPU
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        # Perform matrix multiplication (GPU intensive)
        c = torch.mm(a, b)
        
        # Force GPU sync
        torch.cuda.synchronize()
        
        gpu_time = time.time() - start_time
        
        print(f"   ✅ GPU Computation Test: Success")
        print(f"   Matrix multiplication time: {gpu_time:.4f} seconds")
        print(f"   Result shape: {c.shape}")
        
        # Clear test tensors
        del a, b, c
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU Computation Test Failed: {e}")
        return
    
    # 4. Test diffusers import
    print("\n4️⃣ Diffusers Import Test:")
    try:
        from diffusers import StableDiffusionPipeline
        print(f"   ✅ Diffusers Import: Success")
    except Exception as e:
        print(f"   ❌ Diffusers Import Failed: {e}")
        return
    
    # 5. Test model loading (without generation)
    print("\n5️⃣ Model Loading Test:")
    try:
        print("   Loading model... (this may take a minute)")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to GPU
        pipe = pipe.to('cuda')
        
        print(f"   ✅ Model Loading: Success")
        print(f"   Model device: {pipe.device}")
        print(f"   GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Clear model
        del pipe
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Model Loading Failed: {e}")
        return
    
    print("\n🎉 DIAGNOSIS COMPLETE!")
    print("   If all tests passed, the issue is in the generation pipeline.")
    print("   If any test failed, that's your bottleneck!")

if __name__ == "__main__":
    diagnose_gpu()
