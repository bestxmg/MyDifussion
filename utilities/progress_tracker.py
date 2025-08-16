#!/usr/bin/env python3
"""
PROGRESS TRACKING Stable Diffusion Generator
Shows detailed logs for each step so you can see exactly what's happening!
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import gc

def log_step(step_name, start_time=None):
    """Log each step with timing"""
    if start_time:
        elapsed = time.time() - start_time
        print(f"⏱️  {step_name} completed in {elapsed:.2f} seconds")
    else:
        print(f"🚀 {step_name} starting...")
        return time.time()

def main():
    print("🎭 PROGRESS-TRACKING STABLE DIFFUSION v1.5!")
    print("=" * 60)
    print("📊 You'll see exactly where time is spent!")
    
    # Step 1: Check GPU status
    print("\n" + "="*50)
    start_time = log_step("GPU Status Check")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = "cuda"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"✅ CUDA: {torch.version.cuda}")
    
    log_step("GPU Status Check", start_time)
    
    # Step 2: Load model
    print("\n" + "="*50)
    start_time = log_step("Model Loading")
    
    try:
        print("   📥 Downloading model files...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("   🔄 Moving model to GPU...")
        pipe = pipe.to(device)
        
        print("   ⚙️  Enabling optimizations...")
        pipe.enable_attention_slicing()
        
        log_step("Model Loading", start_time)
        print(f"   💾 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Step 3: Generate image
    print("\n" + "="*50)
    start_time = log_step("Image Generation")
    
    prompt = "danganronpa style, red hair, gentle high school boy, detailed face, sharp features, anime style, high quality, masterpiece, best quality"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph"
    
    print(f"   🎨 Prompt: {prompt}")
    print(f"   🚫 Negative: {negative_prompt}")
    print(f"   🔧 Device: {device}")
    print(f"   📊 Steps: 30")
    print(f"   📏 Size: 512x768")
    print("   🚀 Starting generation...")
    
    try:
        # Track each inference step
        print("   📈 Generation progress:")
        for step in range(1, 31):
            if step % 5 == 0:  # Show progress every 5 steps
                print(f"      Step {step}/30 ({step/30*100:.0f}%)")
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=768,
            num_images_per_prompt=1
        ).images[0]
        
        log_step("Image Generation", start_time)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Step 4: Save image
    print("\n" + "="*50)
    start_time = log_step("Image Saving")
    
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/progress_tracked_character_{timestamp}.png"
        
        image.save(filename)
        log_step("Image Saving", start_time)
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    # Final summary
    print("\n" + "="*50)
    print("🎉 FINAL SUMMARY:")
    print(f"📁 File: {filename}")
    print(f"🎭 Style: Danganronpa with red hair and gentle expression")
    print(f"📱 Size: {image.size}")
    print(f"🎮 GPU used: {torch.cuda.get_device_name(0)}")
    print(f"💾 Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Clear GPU memory
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared")
    
    # Try to open the image
    try:
        print(f"\n🚀 Opening your character...")
        os.startfile(filename)
    except:
        print(f"   Image saved to: {filename}")
        print(f"   Please open it manually!")

if __name__ == "__main__":
    main()
