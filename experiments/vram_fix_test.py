#!/usr/bin/env python3
"""
VRAM Fix Test - Solve the 4GB VRAM limitation
Implements all the recommended fixes for GTX 1650
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def check_vram_usage():
    """Check current VRAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"📊 VRAM Status:")
        print(f"   Total: {total:.1f} GB")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
        print(f"   Free: {total - allocated:.2f} GB")
        return allocated, total
    return 0, 0

def test_vram_optimized():
    """Test with all VRAM optimizations"""
    print("🔧 TESTING VRAM-OPTIMIZED GENERATION")
    print("=" * 50)
    print("Implementing all fixes for 4GB VRAM limitation:")
    print("1. Lower resolution (256x256)")
    print("2. Force float32 instead of fp16")
    print("3. Enable memory-efficient options")
    print("4. Monitor VRAM usage")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Check initial VRAM
        print("📊 Initial VRAM status:")
        initial_allocated, total_vram = check_vram_usage()
        print()
        
        print("📥 Loading model with float32 (VRAM friendly)...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # Force float32 instead of fp16
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("🚀 Moving to GPU...")
        pipe = pipe.to('cuda')
        
        # Enable ALL memory-saving features
        print("⚙️  Enabling memory optimizations...")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        # Check VRAM after loading
        print("\n📊 VRAM after model loading:")
        loaded_allocated, _ = check_vram_usage()
        print(f"   Model uses: {loaded_allocated - initial_allocated:.2f} GB")
        print()
        
        # Simple prompt
        prompt = "flower"
        negative_prompt = ""
        
        print(f"🎨 Generating: '{prompt}'")
        print("   Resolution: 256x256 (VRAM friendly)")
        print("   Data type: float32 (stable)")
        print("   Memory: Optimized")
        print()
        
        # Generate with minimal VRAM usage
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=10,     # Reasonable steps
            guidance_scale=3.0,         # Moderate guidance
            width=256,                  # Low resolution
            height=256,                 # Low resolution
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        
        # Check final VRAM
        print("\n📊 VRAM after generation:")
        final_allocated, _ = check_vram_usage()
        print(f"   Peak usage: {final_allocated:.2f} GB")
        print()
        
        print(f"✅ Generation completed in {generation_time:.2f} seconds")
        print()
        
        # Save image
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/vram_optimized_flower_{timestamp}.png"
        
        image.save(filename)
        print(f"💾 Image saved: {filename}")
        
        # Check image quality
        img_array = np.array(image)
        brightness = np.mean(img_array)
        std_dev = np.std(img_array)
        
        print(f"\n📊 Image Analysis:")
        print(f"   Brightness: {brightness:.2f}")
        print(f"   Variation: {std_dev:.2f}")
        print(f"   Size: {image.size[0]}x{image.size[1]}")
        
        if brightness > 10:
            print("\n🎉 SUCCESS! VRAM optimization worked!")
            print("✅ GPU generation successful with optimizations")
            print("🔍 The issue was VRAM limitation, not GPU failure")
            print("💡 Solution: Use float32 + low resolution + memory optimizations")
        else:
            print("\n❌ FAILED! Still a black image")
            print("🔍 Even VRAM optimization didn't help")
            print("   This suggests a deeper GPU issue")
        
        # Try to open
        try:
            os.startfile(filename)
        except:
            print(f"\n   Please open manually: {filename}")
        
        # Cleanup
        del pipe
        torch.cuda.empty_cache()
        
        return brightness > 10
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔍 This suggests GPU has fundamental issues")
        return False

def test_progressive_resolution():
    """Test progressively higher resolutions to find the limit"""
    print("\n📏 TESTING PROGRESSIVE RESOLUTION")
    print("=" * 50)
    print("Finding the maximum resolution your GPU can handle")
    print()
    
    resolutions = [
        (128, 128, "Tiny"),
        (256, 256, "Small"), 
        (384, 384, "Medium"),
        (448, 448, "Large"),
        (512, 512, "Standard")
    ]
    
    working_resolution = None
    
    for width, height, name in resolutions:
        print(f"🎨 Testing {name} resolution: {width}x{height}")
        
        try:
            # Load model
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            pipe = pipe.to('cuda')
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            
            # Check VRAM before generation
            allocated_before, total = check_vram_usage()
            
            # Generate
            image = pipe(
                prompt="flower",
                num_inference_steps=5,
                guidance_scale=1.0,
                width=width,
                height=height
            ).images[0]
            
            # Check result
            img_array = np.array(image)
            brightness = np.mean(img_array)
            
            if brightness > 10:
                print(f"   ✅ {name} resolution works!")
                working_resolution = (width, height)
                
                # Save working image
                timestamp = int(time.time())
                filename = f"generated_images/working_{width}x{height}_{timestamp}.png"
                image.save(filename)
                print(f"   💾 Saved: {filename}")
            else:
                print(f"   ❌ {name} resolution fails (black image)")
                print(f"   🔍 VRAM limit reached at {width}x{height}")
                break
            
            # Cleanup
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error at {width}x{height}: {e}")
            break
    
    if working_resolution:
        print(f"\n🎯 Maximum working resolution: {working_resolution[0]}x{working_resolution[1]}")
        print("💡 Use this resolution for reliable GPU generation")
    else:
        print("\n❌ No resolution works - fundamental GPU issue")
    
    return working_resolution

def main():
    print("🔧 VRAM FIX TEST - SOLVING 4GB LIMITATION")
    print("=" * 60)
    print("Your GTX 1650 has 4GB VRAM - this is likely the issue!")
    print("Let's implement all the recommended fixes...")
    print()
    
    # Test 1: VRAM optimized generation
    vram_ok = test_vram_optimized()
    
    if vram_ok:
        print("\n🎉 VRAM optimization solved the problem!")
        print("✅ Your GPU works fine - it was just VRAM limited")
        print("💡 Use these settings for future generations:")
        print("   - Resolution: 256x256 or lower")
        print("   - Data type: float32")
        print("   - Enable attention_slicing and vae_slicing")
        
        # Test 2: Find maximum working resolution
        print("\n🔍 Now let's find your maximum working resolution...")
        max_resolution = test_progressive_resolution()
        
    else:
        print("\n❌ Even VRAM optimization didn't help")
        print("🔍 This suggests a deeper GPU issue")
        print("💡 Your GPU might have fundamental problems")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
