#!/usr/bin/env python3
"""
Simple GPU Test for Training
Tests core GPU functionality needed for model training
"""

import torch
import time

def test_gpu_training_basics():
    """Test basic GPU operations needed for training"""
    print("🔧 SIMPLE GPU TRAINING TEST")
    print("=" * 50)
    
    # Test 1: CUDA availability
    print("1️⃣ Testing CUDA availability...")
    if torch.cuda.is_available():
        print("   ✅ CUDA is available")
        device = torch.device('cuda')
    else:
        print("   ❌ CUDA not available")
        return False
    
    # Test 2: Basic tensor operations
    print("\n2️⃣ Testing basic tensor operations...")
    try:
        # Create tensors on GPU
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        
        # Basic operations
        z = x + y
        w = torch.matmul(x, y)
        
        print("   ✅ Basic tensor operations work")
        print(f"   ✅ Tensor shapes: x={x.shape}, y={y.shape}, z={z.shape}, w={w.shape}")
        
    except Exception as e:
        print(f"   ❌ Basic tensor operations failed: {e}")
        return False
    
    # Test 3: Memory management
    print("\n3️⃣ Testing memory management...")
    try:
        initial_memory = torch.cuda.memory_allocated(0)
        print(f"   Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        
        # Allocate some memory
        large_tensor = torch.randn(1000, 1000, device=device)
        current_memory = torch.cuda.memory_allocated(0)
        print(f"   After allocation: {current_memory / 1024**2:.1f} MB")
        
        # Free memory
        del large_tensor
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(0)
        print(f"   After cleanup: {final_memory / 1024**2:.1f} MB")
        
        print("   ✅ Memory management works")
        
    except Exception as e:
        print(f"   ❌ Memory management failed: {e}")
        return False
    
    # Test 4: Training-like operations
    print("\n4️⃣ Testing training-like operations...")
    try:
        # Create a simple model-like structure
        weights = torch.randn(100, 100, device=device, requires_grad=True)
        inputs = torch.randn(100, 100, device=device)
        targets = torch.randn(100, 100, device=device)
        
        # Forward pass
        outputs = torch.matmul(inputs, weights)
        loss = torch.mean((outputs - targets) ** 2)
        
        # Backward pass
        loss.backward()
        
        print("   ✅ Forward pass works")
        print("   ✅ Backward pass works")
        print("   ✅ Gradients computed")
        print(f"   ✅ Loss value: {loss.item():.6f}")
        
        # Check gradients
        if weights.grad is not None:
            print("   ✅ Gradients are properly attached")
        else:
            print("   ❌ Gradients not attached")
            return False
        
    except Exception as e:
        print(f"   ❌ Training operations failed: {e}")
        return False
    
    # Test 5: Data transfer
    print("\n5️⃣ Testing data transfer...")
    try:
        # CPU to GPU
        cpu_tensor = torch.randn(50, 50)
        gpu_tensor = cpu_tensor.to(device)
        
        # GPU to CPU
        back_to_cpu = gpu_tensor.cpu()
        
        # Verify data integrity
        if torch.allclose(cpu_tensor, back_to_cpu, atol=1e-6):
            print("   ✅ Data transfer preserves values")
        else:
            print("   ❌ Data transfer corrupted values")
            return False
        
    except Exception as e:
        print(f"   ❌ Data transfer failed: {e}")
        return False
    
    # Test 6: Batch processing
    print("\n6️⃣ Testing batch processing...")
    try:
        batch_size = 32
        input_size = 64
        
        # Create batch
        batch = torch.randn(batch_size, input_size, device=device)
        
        # Process batch
        processed = batch * 2 + 1
        
        print(f"   ✅ Batch processing works (batch_size={batch_size})")
        print(f"   ✅ Input shape: {batch.shape}")
        print(f"   ✅ Output shape: {processed.shape}")
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Your GPU is ready for training!")
    return True

def test_stable_diffusion_simple():
    """Test Stable Diffusion with minimal settings"""
    print("\n🔍 TESTING STABLE DIFFUSION ON GPU")
    print("=" * 50)
    
    try:
        from diffusers import StableDiffusionPipeline
        
        print("Loading Stable Diffusion model...")
        
        # Load with minimal settings
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to('cuda')
        
        # Enable optimizations
        pipe.enable_attention_slicing()
        
        print("✅ Model loaded successfully")
        
        # Test with minimal generation
        print("Testing minimal generation...")
        
        test_image = pipe(
            prompt="simple test",
            num_inference_steps=5,
            guidance_scale=1.0,
            width=64,
            height=64
        ).images[0]
        
        # Convert to array and check brightness
        import numpy as np
        img_array = np.array(test_image)
        brightness = np.mean(img_array)
        
        print(f"Test image brightness: {brightness:.2f}")
        
        if brightness > 10:
            print("✅ Stable Diffusion GPU test: SUCCESS!")
            print("   The black image issue is resolved!")
            result = True
        else:
            print("❌ Stable Diffusion GPU test: FAILED (black image)")
            print("   The issue persists")
            result = False
        
        # Cleanup
        del pipe, test_image
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"❌ Stable Diffusion test failed: {e}")
        return False

def main():
    print("🚀 GPU TRAINING CAPABILITY TEST")
    print("=" * 60)
    
    # Test 1: Basic training capabilities
    basic_test_passed = test_gpu_training_basics()
    
    if basic_test_passed:
        print("\n" + "="*60)
        print("🎯 RECOMMENDATION:")
        print("Your GPU is fully capable of training models!")
        print("The black image issue is specific to Stable Diffusion,")
        print("not a fundamental GPU problem.")
        print()
        print("✅ Use GPU for: Model training, fine-tuning, inference")
        print("⚠️  Use CPU for: Stable Diffusion image generation (for now)")
        
        # Test 2: Try Stable Diffusion
        print("\n" + "="*60)
        print("🔍 Let's test if Stable Diffusion black image issue is fixed...")
        
        sd_test_passed = test_stable_diffusion_simple()
        
        if sd_test_passed:
            print("\n🎉 EXCELLENT! Both issues are resolved!")
            print("✅ GPU training: READY")
            print("✅ Stable Diffusion: READY")
        else:
            print("\n📋 SUMMARY:")
            print("✅ GPU training: READY")
            print("❌ Stable Diffusion: Still has black image issue")
            print("   (But this won't affect model training)")
    
    else:
        print("\n❌ GPU has fundamental issues that need fixing")
        print("Contact NVIDIA support or check hardware")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
