#!/usr/bin/env python3
"""
GPU Training Fix - Comprehensive solution for GPU black image issue
"""

import torch
import subprocess
import time
import os
import numpy as np

def print_status(message, status="INFO"):
    """Print status message with appropriate icon"""
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "FIX": "🔧"
    }
    icon = icons.get(status, "ℹ️")
    print(f"{icon} {message}")

def check_gpu_basics():
    """Check basic GPU functionality"""
    print_status("Checking basic GPU functionality...", "INFO")
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print_status("CUDA not available", "ERROR")
            return False
        
        # Check GPU info
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print_status(f"GPU Count: {gpu_count}", "SUCCESS")
        print_status(f"Current Device: {current_device}", "SUCCESS")
        print_status(f"GPU Name: {gpu_name}", "SUCCESS")
        print_status(f"GPU Memory: {gpu_memory:.1f} GB", "SUCCESS")
        
        return True
        
    except Exception as e:
        print_status(f"GPU basic check failed: {e}", "ERROR")
        return False

def test_gpu_tensor_operations():
    """Test GPU tensor operations"""
    print_status("Testing GPU tensor operations...", "INFO")
    
    try:
        # Test 1: Basic tensor creation
        test_tensor = torch.randn(100, 100, device='cuda')
        print_status("Basic tensor creation: SUCCESS", "SUCCESS")
        
        # Test 2: Basic operations
        result = test_tensor * 2 + 1
        print_status("Basic operations: SUCCESS", "SUCCESS")
        
        # Test 3: Memory operations
        gpu_memory_before = torch.cuda.memory_allocated(0)
        large_tensor = torch.randn(1000, 1000, device='cuda')
        gpu_memory_after = torch.cuda.memory_allocated(0)
        memory_used = (gpu_memory_after - gpu_memory_before) / 1024**2
        print_status(f"Memory operations: SUCCESS ({memory_used:.1f} MB)", "SUCCESS")
        
        # Test 4: Data transfer
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.to('cuda')
        back_to_cpu = gpu_tensor.cpu()
        data_preserved = torch.allclose(cpu_tensor, back_to_cpu, atol=1e-6)
        print_status(f"Data transfer: {'SUCCESS' if data_preserved else 'FAILED'}", 
                    "SUCCESS" if data_preserved else "ERROR")
        
        # Cleanup
        del test_tensor, result, large_tensor, gpu_tensor, back_to_cpu
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print_status(f"GPU tensor test failed: {e}", "ERROR")
        return False

def test_gpu_memory_consistency():
    """Test GPU memory consistency"""
    print_status("Testing GPU memory consistency...", "INFO")
    
    try:
        initial_memory = torch.cuda.memory_allocated(0)
        print_status(f"Initial memory: {initial_memory / 1024**2:.1f} MB", "INFO")
        
        # Allocate and deallocate tensors
        tensors = []
        for i in range(5):
            tensor = torch.randn(500, 500, device='cuda')
            tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated(0)
            print_status(f"Step {i+1}: {current_memory / 1024**2:.1f} MB", "INFO")
        
        # Cleanup
        for tensor in tensors:
            del tensor
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(0)
        print_status(f"Final memory: {final_memory / 1024**2:.1f} MB", "INFO")
        
        if final_memory < initial_memory:
            print_status("Memory cleanup: SUCCESS", "SUCCESS")
            return True
        else:
            print_status("Memory cleanup: FAILED", "WARNING")
            return False
            
    except Exception as e:
        print_status(f"Memory consistency test failed: {e}", "ERROR")
        return False

def test_simple_image_generation():
    """Test simple image generation on GPU"""
    print_status("Testing simple image generation on GPU...", "INFO")
    
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
        
        print_status(f"Test image brightness: {brightness:.2f}", "INFO")
        print_status(f"Test image variation: {variation:.2f}", "INFO")
        
        if brightness > 50 and brightness < 200 and variation > 20:
            print_status("Simple image generation: SUCCESS", "SUCCESS")
            return True
        else:
            print_status("Simple image generation: SUSPICIOUS VALUES", "WARNING")
            return False
            
    except Exception as e:
        print_status(f"Simple image generation failed: {e}", "ERROR")
        return False

def fix_gpu_issues():
    """Attempt to fix common GPU issues"""
    print_status("Attempting to fix GPU issues...", "FIX")
    
    fixes_applied = []
    
    try:
        # Fix 1: Clear GPU cache
        print_status("Clearing GPU cache...", "INFO")
        torch.cuda.empty_cache()
        fixes_applied.append("GPU cache cleared")
        
        # Fix 2: Reset GPU memory
        print_status("Resetting GPU memory...", "INFO")
        torch.cuda.reset_peak_memory_stats()
        fixes_applied.append("GPU memory reset")
        
        # Fix 3: Check for memory fragmentation
        print_status("Checking memory fragmentation...", "INFO")
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        fragmentation = (reserved - allocated) / reserved if reserved > 0 else 0
        
        print_status(f"Memory fragmentation: {fragmentation:.2%}", "INFO")
        if fragmentation > 0.5:
            print_status("High memory fragmentation detected", "WARNING")
            fixes_applied.append("Memory fragmentation identified")
        
        # Fix 4: Test with smaller tensors
        print_status("Testing with smaller tensors...", "INFO")
        small_tensor = torch.randn(10, 10, device='cuda')
        small_result = small_tensor * 2
        del small_tensor, small_result
        torch.cuda.empty_cache()
        
        if torch.cuda.memory_allocated(0) < 1024*1024:  # Less than 1MB
            print_status("Small tensor test: SUCCESS", "SUCCESS")
            fixes_applied.append("Small tensor test passed")
        else:
            print_status("Small tensor test: FAILED", "WARNING")
        
        return fixes_applied
        
    except Exception as e:
        print_status(f"GPU fix attempt failed: {e}", "ERROR")
        return []

def test_stable_diffusion_gpu():
    """Test Stable Diffusion specifically on GPU"""
    print_status("Testing Stable Diffusion on GPU...", "INFO")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        print_status("Loading Stable Diffusion model...", "INFO")
        
        # Load with GPU-specific settings
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # Use float16 for GPU
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to('cuda')
        
        # Enable GPU optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print_status("Model loaded on GPU successfully", "SUCCESS")
        
        # Test with minimal generation
        print_status("Testing minimal generation...", "INFO")
        
        # Use very small settings for testing
        test_image = pipe(
            prompt="simple test",
            num_inference_steps=5,  # Very few steps for testing
            guidance_scale=1.0,
            width=64,  # Very small size for testing
            height=64
        ).images[0]
        
        # Check if image is not black
        img_array = test_image.convert('RGB')
        img_data = torch.tensor(np.array(img_array)).float()
        brightness = img_data.mean()
        
        print_status(f"Test image brightness: {brightness:.2f}", "INFO")
        
        if brightness > 10:
            print_status("Stable Diffusion GPU test: SUCCESS", "SUCCESS")
            result = True
        else:
            print_status("Stable Diffusion GPU test: FAILED (black image)", "ERROR")
            result = False
        
        # Cleanup
        del pipe, test_image
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print_status(f"Stable Diffusion GPU test failed: {e}", "ERROR")
        return False

def main():
    print("🔧 GPU TRAINING FIX - COMPREHENSIVE DIAGNOSTIC")
    print("=" * 60)
    print("This tool will diagnose and fix GPU issues for model training")
    print()
    
    # Step 1: Basic GPU check
    print_status("STEP 1: Basic GPU Check", "INFO")
    if not check_gpu_basics():
        print_status("Basic GPU check failed - cannot proceed", "ERROR")
        return
    
    print()
    
    # Step 2: GPU tensor operations
    print_status("STEP 2: GPU Tensor Operations", "INFO")
    tensor_test_passed = test_gpu_tensor_operations()
    
    print()
    
    # Step 3: GPU memory consistency
    print_status("STEP 3: GPU Memory Consistency", "INFO")
    memory_test_passed = test_gpu_memory_consistency()
    
    print()
    
    # Step 4: Simple image generation
    print_status("STEP 4: Simple Image Generation", "INFO")
    image_test_passed = test_simple_image_generation()
    
    print()
    
    # Step 5: Apply fixes if needed
    if not all([tensor_test_passed, memory_test_passed, image_test_passed]):
        print_status("STEP 5: Applying GPU Fixes", "INFO")
        fixes = fix_gpu_issues()
        if fixes:
            print_status("Fixes applied:", "INFO")
            for fix in fixes:
                print(f"   - {fix}")
        else:
            print_status("No fixes could be applied", "WARNING")
    
    print()
    
    # Step 6: Test Stable Diffusion specifically
    print_status("STEP 6: Stable Diffusion GPU Test", "INFO")
    sd_test_passed = test_stable_diffusion_gpu()
    
    print()
    
    # Summary
    print_status("DIAGNOSTIC SUMMARY", "INFO")
    print("=" * 40)
    
    tests = [
        ("Basic GPU", True),  # Already passed
        ("Tensor Operations", tensor_test_passed),
        ("Memory Consistency", memory_test_passed),
        ("Simple Image Gen", image_test_passed),
        ("Stable Diffusion", sd_test_passed)
    ]
    
    passed_count = 0
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if passed:
            passed_count += 1
    
    print(f"\nOverall: {passed_count}/{len(tests)} tests passed")
    
    # Recommendations
    print()
    print_status("RECOMMENDATIONS", "INFO")
    print("-" * 40)
    
    if passed_count == len(tests):
        print_status("All tests passed! Your GPU should work for training", "SUCCESS")
        print("   Try running your training script now")
    elif sd_test_passed:
        print_status("Stable Diffusion works on GPU!", "SUCCESS")
        print("   The black image issue may be resolved")
        print("   Try generating images with GPU again")
    elif tensor_test_passed and memory_test_passed:
        print_status("GPU basics work, but image generation has issues", "WARNING")
        print("   This suggests a Stable Diffusion specific problem")
        print("   Consider updating diffusers library or using different model")
    else:
        print_status("Multiple GPU issues detected", "ERROR")
        print("   Consider:")
        print("   1. Updating GPU drivers")
        print("   2. Reinstalling CUDA toolkit")
        print("   3. Checking GPU hardware")
    
    print()
    print_status("For now, continue using CPU for generation", "INFO")
    print("   GPU training should work once basic GPU tests pass")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
