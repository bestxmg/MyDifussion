#!/usr/bin/env python3
"""
Simple Stable Diffusion Debugging Script
Demonstrates line-by-line debugging with pdb
"""

import pdb
import torch
from diffusers import StableDiffusionPipeline
import time

def debug_stable_diffusion():
    """Main function with debug breakpoints"""
    
    print("🐛 DEBUG MODE: Stable Diffusion Pipeline")
    print("=" * 50)
    
    # BREAKPOINT 1: Before model loading
    print("About to load model...")
    pdb.set_trace()  # Debugger will stop here
    
    try:
        # Load model
        print("📥 Loading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("✅ Model loaded successfully")
        
        # BREAKPOINT 2: After model loading
        print("Model loaded, about to move to GPU...")
        pdb.set_trace()
        
        # Move to GPU
        pipe = pipe.to('cuda')
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print("✅ Model moved to GPU with optimizations")
        
        # BREAKPOINT 3: Before generation
        print("About to generate image...")
        pdb.set_trace()
        
        # Generate image
        prompt = "a cute cat, high quality, detailed"
        negative_prompt = "low quality, blurry"
        
        print(f"🎨 Generating: {prompt}")
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.0,
            width=512,
            height=512
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Generation completed in {generation_time:.2f} seconds")
        
        # BREAKPOINT 4: After generation
        print("Image generated, about to save...")
        pdb.set_trace()
        
        # Save image
        timestamp = int(time.time())
        filename = f"debug_generated_{timestamp}.png"
        image.save(filename)
        print(f"💾 Image saved as: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        pdb.set_trace()  # Debug on error

if __name__ == "__main__":
    debug_stable_diffusion()
