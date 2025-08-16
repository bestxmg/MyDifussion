#!/usr/bin/env python3
"""
GPU-Optimized Stable Diffusion Generator
Now uses your NVIDIA GTX 1650 for fast generation!
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time

def main():
    print("🚀 GPU-ACCELERATED STABLE DIFFUSION v1.5!")
    print("=" * 60)
    
    # Check GPU status
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    
    # Step 1: Load model on GPU
    print("\n📥 Loading Stable Diffusion v1.5 on GPU...")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # Use half precision for GPU memory efficiency
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to GPU
        device = "cuda"
        pipe = pipe.to(device)
        
        # Enable GPU optimizations
        pipe.enable_attention_slicing()  # Reduce memory usage
        pipe.enable_vae_slicing()       # Additional memory optimization
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"💾 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ GPU loading failed: {e}")
        return
    
    # Step 2: Generate image with GPU
    print("\n🎨 Generating your character on GPU...")
    prompt = "danganronpa style, red hair, gentle high school boy, detailed face, sharp features, anime style, high quality, masterpiece, best quality"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph"
    
    print(f"   Prompt: {prompt}")
    print(f"   Device: {device}")
    print(f"   Generating... (should be much faster now!)")
    
    try:
        start_time = time.time()
        
        # Clear GPU cache before generation
        torch.cuda.empty_cache()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=768,
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Image generated in {generation_time:.2f} seconds!")
        print(f"⚡ Speed improvement: ~10-20x faster than CPU!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Step 3: Save image
    print("\n💾 Saving your character...")
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/gpu_danganronpa_character_{timestamp}.png"
        
        image.save(filename)
        print(f"✅ Image saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    # Step 4: Display results
    print("\n🎉 SUCCESS! GPU-accelerated generation complete!")
    print(f"📁 File: {filename}")
    print(f"🎭 Style: Danganronpa with red hair and gentle expression")
    print(f"📱 Size: {image.size}")
    print(f"⚡ Generation time: {generation_time:.2f} seconds (GPU accelerated!)")
    print(f"🎮 GPU used: {torch.cuda.get_device_name(0)}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    print(f"🧹 GPU memory cleared")
    
    # Try to open the image
    try:
        print(f"\n🚀 Opening your character...")
        os.startfile(filename)
    except:
        print(f"   Image saved to: {filename}")
        print(f"   Please open it manually!")

if __name__ == "__main__":
    main()
