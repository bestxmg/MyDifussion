#!/usr/bin/env python3
"""
Working Stable Diffusion Generator - No xformers dependency
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time

def main():
    print("👨‍💻 GENERATING ULTIMATE PROGRAMMER WITH BLUE HAIR!")
    print("=" * 60)
    
    # Step 1: Load model
    print("📥 Loading Stable Diffusion v1.5...")
    print("   This will download ~4GB model file (first time only)")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # Force float32 for CPU
            safety_checker=None,
            requires_safety_checker=False,
            resume_download=True,  # Resume interrupted downloads
            local_files_only=False  # Allow network fallback
        )
        
        # Force CPU usage to avoid GPU black image issue
        device = "cpu"
        pipe = pipe.to(device)
        
        print(f"🔧 Using CPU for reliable generation (GPU has black image issue)")
        
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Step 2: Generate image
    print("\n🎨 Generating your character...")
    prompt = "ultimate programmer, with blue hair and gentle personality, professional developer, coding, computer science, anime style, high quality, masterpiece, best quality, detailed face, sharp features"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph"
    
    print(f"   Prompt: {prompt}")
    print(f"   Negative: {negative_prompt}")
    print("   Generating... (this may take 1-3 minutes)")
    
    try:
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,  # Reduced for VPN stability
            guidance_scale=7.0,      # Slightly reduced for stability
            width=512,
            height=768,
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Image generated in {generation_time:.2f} seconds!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Step 3: Save image
    print("\n💾 Saving your character...")
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/ultimate_programmer_cpu_{timestamp}.png"
        
        image.save(filename)
        print(f"✅ Image saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    # Step 4: Display results
    print("\n🎉 SUCCESS! Your ultimate programmer is ready!")
    print(f"📁 File: {filename}")
    print(f"👨‍💻 Style: Ultimate programmer with blue hair and gentle personality")
    print(f"📱 Size: {image.size}")
    print(f"⏱️  Generation time: {generation_time:.2f} seconds")
    
    # Try to open the image
    try:
        print(f"\n🚀 Opening your character...")
        os.startfile(filename)
    except:
        print(f"   Image saved to: {filename}")
        print(f"   Please open it manually to view your character!")

if __name__ == "__main__":
    main()
