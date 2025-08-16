#!/usr/bin/env python3
"""
DEBUG SCRIPT - Find why images are black
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np

def debug_generation():
    print("🔍 DEBUGGING BLACK IMAGE ISSUE!")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = "cuda"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Load model
        print("\n📥 Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        
        print("✅ Model loaded")
        print(f"💾 GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Test with simple prompt
        print("\n🎨 Testing with simple prompt...")
        prompt = "a red circle on white background"
        
        # Generate with debug info
        print("   Starting generation...")
        start_time = time.time()
        
        # Force CPU fallback to test if it's GPU-specific
        print("   Testing CPU fallback first...")
        pipe_cpu = pipe.to("cpu")
        
        with torch.no_grad():
            result_cpu = pipe_cpu(
                prompt=prompt,
                num_inference_steps=5,  # Very few steps for testing
                guidance_scale=1.0,
                width=256,  # Small size for testing
                height=256
            )
        
        cpu_time = time.time() - start_time
        print(f"   CPU generation completed in {cpu_time:.2f}s")
        
        # Check CPU result
        if result_cpu.images and len(result_cpu.images) > 0:
            cpu_image = result_cpu.images[0]
            print(f"   CPU image size: {cpu_image.size}")
            print(f"   CPU image mode: {cpu_image.mode}")
            
            # Check pixel values
            cpu_array = np.array(cpu_image)
            print(f"   CPU image shape: {cpu_array.shape}")
            print(f"   CPU min/max values: {cpu_array.min()}/{cpu_array.max()}")
            print(f"   CPU mean value: {cpu_array.mean():.2f}")
            
            # Save CPU test
            cpu_image.save("debug_cpu_test.png")
            print("   ✅ CPU test image saved as debug_cpu_test.png")
        else:
            print("   ❌ CPU generation failed - no images")
        
        # Now test GPU
        print("\n🎮 Testing GPU generation...")
        pipe_gpu = pipe.to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            result_gpu = pipe_gpu(
                prompt=prompt,
                num_inference_steps=5,
                guidance_scale=1.0,
                width=256,
                height=256
            )
        
        gpu_time = time.time() - start_time
        print(f"   GPU generation completed in {gpu_time:.2f}s")
        
        # Check GPU result
        if result_gpu.images and len(result_gpu.images) > 0:
            gpu_image = result_gpu.images[0]
            print(f"   GPU image size: {gpu_image.size}")
            print(f"   GPU image mode: {gpu_image.mode}")
            
            # Check pixel values
            gpu_array = np.array(gpu_image)
            print(f"   GPU image shape: {gpu_array.shape}")
            print(f"   GPU min/max values: {gpu_array.min()}/{gpu_array.max()}")
            print(f"   GPU mean value: {gpu_array.mean():.2f}")
            
            # Save GPU test
            gpu_image.save("debug_gpu_test.png")
            print("   ✅ GPU test image saved as debug_gpu_test.png")
        else:
            print("   ❌ GPU generation failed - no images")
        
        # Compare results
        print("\n📊 COMPARISON:")
        print(f"   CPU time: {cpu_time:.2f}s")
        print(f"   GPU time: {gpu_time:.2f}s")
        
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"   GPU speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_generation()
