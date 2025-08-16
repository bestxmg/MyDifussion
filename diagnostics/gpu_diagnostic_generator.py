#!/usr/bin/env python3
"""
GPU Diagnostic Generator - Find out why we get black images
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_gpu_basics():
    """Test basic GPU functionality"""
    print("🔍 TESTING GPU BASICS")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Basic GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ Memory: {gpu_memory:.1f} GB")
        
        # Test basic tensor operations
        test_tensor = torch.randn(100, 100, device='cuda')
        result = test_tensor * 2 + 1
        print(f"✅ Basic tensor operations work")
        
        # Test memory allocation
        large_tensor = torch.randn(1000, 1000, device='cuda')
        memory_used = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"✅ GPU memory allocation works: {memory_used:.1f} MB")
        
        # Cleanup
        del test_tensor, result, large_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU basics failed: {e}")
        return False

def test_simple_image_creation():
    """Test creating a simple image on GPU"""
    print("\n🎨 TESTING SIMPLE IMAGE CREATION")
    print("=" * 40)
    
    try:
        # Create a simple test image on GPU
        test_image = torch.randn(3, 64, 64, device='cuda')
        
        # Convert to proper format
        test_image = (test_image + 1) * 127.5  # Convert from [-1,1] to [0,255]
        test_image = test_image.clamp(0, 255).byte()
        
        # Move to CPU for analysis
        test_image_cpu = test_image.cpu().permute(1, 2, 0).numpy()
        
        # Check image values
        brightness = test_image_cpu.mean()
        variation = test_image_cpu.std()
        
        print(f"✅ Test image created on GPU")
        print(f"   Brightness: {brightness:.2f}")
        print(f"   Variation: {variation:.2f}")
        
        if brightness > 50 and brightness < 200 and variation > 20:
            print("✅ Simple image generation works")
            return True
        else:
            print("⚠️  Image values seem off")
            return False
            
    except Exception as e:
        print(f"❌ Simple image creation failed: {e}")
        return False

def test_stable_diffusion_step_by_step():
    """Test Stable Diffusion step by step"""
    print("\n🔬 TESTING STABLE DIFFUSION STEP BY STEP")
    print("=" * 50)
    
    try:
        print("Step 1: Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        print("✅ Model loaded")
        
        print("Step 2: Moving to GPU...")
        pipe = pipe.to('cuda')
        print("✅ Model moved to GPU")
        
        print("Step 3: Enabling optimizations...")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print("✅ Optimizations enabled")
        
        print("Step 4: Testing minimal generation...")
        # Use very minimal settings
        test_image = pipe(
            prompt="simple test",
            num_inference_steps=5,
            guidance_scale=1.0,
            width=64,
            height=64
        ).images[0]
        
        print("✅ Minimal generation completed")
        
        # Check the result
        img_array = np.array(test_image)
        brightness = img_array.mean()
        variation = img_array.std()
        
        print(f"   Result brightness: {brightness:.2f}")
        print(f"   Result variation: {variation:.2f}")
        
        if brightness > 10:
            print("✅ Minimal generation produces non-black image")
            result = True
        else:
            print("❌ Minimal generation still produces black image")
            result = False
        
        # Cleanup
        del pipe, test_image
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ Step-by-step test failed: {e}")
        return False

def test_different_model():
    """Test with a different model"""
    print("\n🔄 TESTING DIFFERENT MODEL")
    print("=" * 40)
    
    try:
        print("Trying a different Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",  # Different model
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to('cuda')
        pipe.enable_attention_slicing()
        
        print("✅ Different model loaded")
        
        # Test generation
        test_image = pipe(
            prompt="simple test",
            num_inference_steps=5,
            guidance_scale=1.0,
            width=64,
            height=64
        ).images[0]
        
        # Check result
        img_array = np.array(test_image)
        brightness = img_array.mean()
        
        print(f"   Result brightness: {brightness:.2f}")
        
        if brightness > 10:
            print("✅ Different model works!")
            result = True
        else:
            print("❌ Different model also produces black image")
            result = False
        
        # Cleanup
        del pipe, test_image
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ Different model test failed: {e}")
        return False

def main():
    print("🔬 GPU DIAGNOSTIC GENERATOR")
    print("=" * 50)
    print("Let's find out why GPU generation produces black images")
    print()
    
    # Test 1: GPU basics
    gpu_basics_ok = test_gpu_basics()
    
    if not gpu_basics_ok:
        print("\n❌ GPU basics failed - cannot proceed")
        return
    
    # Test 2: Simple image creation
    simple_image_ok = test_simple_image_creation()
    
    # Test 3: Stable Diffusion step by step
    sd_test_ok = test_stable_diffusion_step_by_step()
    
    # Test 4: Different model
    different_model_ok = test_different_model()
    
    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 40)
    
    tests = [
        ("GPU Basics", gpu_basics_ok),
        ("Simple Image Creation", simple_image_ok),
        ("Stable Diffusion Step-by-Step", sd_test_ok),
        ("Different Model", different_model_ok)
    ]
    
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    # Analysis
    print("\n🔍 ANALYSIS")
    print("-" * 20)
    
    if gpu_basics_ok and simple_image_ok and not sd_test_ok:
        print("🎯 The issue is with Stable Diffusion specifically")
        print("   GPU works fine, but SD pipeline has problems")
        print("   Possible causes:")
        print("   - Model corruption")
        print("   - Library version issues")
        print("   - Memory handling in SD pipeline")
        
    elif gpu_basics_ok and not simple_image_ok:
        print("🎯 The issue is with GPU image processing")
        print("   Basic GPU ops work, but image creation fails")
        print("   Possible causes:")
        print("   - GPU driver issues")
        print("   - Memory corruption")
        print("   - CUDA version mismatch")
        
    elif not gpu_basics_ok:
        print("🎯 The issue is fundamental GPU problems")
        print("   Basic GPU functionality is broken")
        print("   Possible causes:")
        print("   - Driver issues")
        print("   - Hardware problems")
        print("   - CUDA installation issues")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    if not sd_test_ok:
        print("1. Try reinstalling diffusers library")
        print("2. Clear model cache: rm -rf ~/.cache/huggingface")
        print("3. Try different Stable Diffusion model")
        print("4. Check for library version conflicts")
    
    if not simple_image_ok:
        print("1. Update GPU drivers")
        print("2. Reinstall CUDA toolkit")
        print("3. Check GPU memory for corruption")
    
    print("\n5. For now, use CPU generation (it works!)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
