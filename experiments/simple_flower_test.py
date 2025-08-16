#!/usr/bin/env python3
"""
Ultra-Simple Flower Test - Minimal workload to test GPU
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🌸 ULTRA-SIMPLE FLOWER TEST")
    print("=" * 40)
    print("Testing with minimal workload: just 'flower'")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    try:
        print("📥 Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("🚀 Moving to GPU...")
        pipe = pipe.to('cuda')
        
        # Enable minimal memory optimizations
        pipe.enable_attention_slicing()
        
        print("✅ Model ready")
        print()
        
        # ULTRA simple prompt - just one word
        prompt = "flower"
        negative_prompt = ""
        
        print(f"🎨 Generating: '{prompt}'")
        print("   (This is the simplest possible prompt)")
        print()
        
        # Minimal settings - as simple as possible
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=5,      # Minimal steps
            guidance_scale=1.0,         # Minimal guidance
            width=256,                  # Small size
            height=256,                 # Small size
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        
        print(f"✅ Generation completed in {generation_time:.2f} seconds")
        print()
        
        # Save image
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/simple_flower_{timestamp}.png"
        
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
            print("\n🎉 SUCCESS! Simple flower generated!")
            print("✅ GPU works with minimal workload")
            print("🔍 The issue was complex prompts, not GPU")
        else:
            print("\n❌ FAILED! Still a black image")
            print("🔍 The issue is fundamental GPU problem")
            print("   Even minimal workload fails")
        
        # Try to open
        try:
            os.startfile(filename)
        except:
            print(f"\n   Please open manually: {filename}")
        
        # Cleanup
        del pipe
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔍 This tells us the GPU has fundamental issues")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
