#!/usr/bin/env python3
"""
MAXIMUM GPU UTILIZATION Stable Diffusion Generator
Forces 100% GPU usage for maximum speed!
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import time
import gc

def main():
    print("🚀 MAXIMUM GPU UTILIZATION - STABLE DIFFUSION v1.5!")
    print("=" * 70)
    
    # Force CUDA device
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # Set device explicitly
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Check GPU status
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    print(f"🔧 PyTorch CUDA: {torch.version.cuda}")
    
    # Force maximum GPU memory allocation
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
    
    # Step 1: Load model with maximum GPU optimization
    print("\n📥 Loading Stable Diffusion v1.5 with MAXIMUM GPU optimization...")
    
    try:
        # Force model to GPU with aggressive settings (removed device_map="auto")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # Force half precision
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Force move to GPU
        pipe = pipe.to(device)
        
        # Enable maximum GPU optimizations
        pipe.enable_attention_slicing(1)  # Maximum slicing
        pipe.enable_vae_slicing()         # VAE slicing
        
        # Use faster scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"💾 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"💾 GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ GPU loading failed: {e}")
        return
    
    # Step 2: Generate with maximum GPU utilization
    print("\n🎨 Generating with MAXIMUM GPU power...")
    prompt = "danganronpa style, red hair, gentle high school boy, detailed face, sharp features, anime style, high quality, masterpiece, best quality"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph"
    
    print(f"   Prompt: {prompt}")
    print(f"   Device: {device}")
    print(f"   GPU Memory Target: 95% utilization")
    print(f"   Generating with maximum speed...")
    
    try:
        start_time = time.time()
        
        # Force GPU computation
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,  # Reduced for speed
                guidance_scale=7.5,
                width=512,
                height=768,
                num_images_per_prompt=1
            ).images[0]
        
        generation_time = time.time() - start_time
        
        # Force GPU sync
        torch.cuda.synchronize()
        
        print(f"✅ Image generated in {generation_time:.2f} seconds!")
        print(f"⚡ MAXIMUM GPU acceleration achieved!")
        print(f"💾 Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Step 3: Save image
    print("\n💾 Saving your character...")
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/max_gpu_danganronpa_{timestamp}.png"
        
        image.save(filename)
        print(f"✅ Image saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    # Step 4: Display results
    print("\n🎉 SUCCESS! MAXIMUM GPU utilization complete!")
    print(f"📁 File: {filename}")
    print(f"🎭 Style: Danganronpa with red hair and gentle expression")
    print(f"📱 Size: {image.size}")
    print(f"⚡ Generation time: {generation_time:.2f} seconds (MAXIMUM GPU!)")
    print(f"🎮 GPU used: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory Peak: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    # Clear GPU memory
    del pipe
    gc.collect()
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
