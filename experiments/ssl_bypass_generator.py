#!/usr/bin/env python3
"""
SSL Bypass Generator - Handles network SSL issues
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

def main():
    print("🎭 GENERATING WITH SSL BYPASS!")
    print("=" * 60)
    
    # Set environment variables to bypass SSL issues
    os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
    os.environ['CURL_CA_BUNDLE'] = ''
    
    print("📥 Loading Stable Diffusion v1.5...")
    print("   Using SSL bypass for network issues...")
    
    try:
        # Try to load from cache first
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(cache_dir, "models--runwayml--stable-diffusion-v1-5")
        
        if os.path.exists(model_path):
            print(f"✅ Found cached model at: {model_path}")
            print("   Attempting to load from cache...")
            
            # Try loading from local cache
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None
            )
            print("✅ Model loaded from cache successfully!")
            
        else:
            print("❌ No cached model found. Network download required.")
            return
            
    except Exception as e:
        print(f"❌ Cache loading failed: {e}")
        print("   Trying alternative approach...")
        
        try:
            # Try with different download settings
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                local_files_only=False,
                resume_download=True,
                force_download=False
            )
            print("✅ Model loaded with alternative settings!")
            
        except Exception as e2:
            print(f"❌ Alternative loading failed: {e2}")
            print("\n🔧 TROUBLESHOOTING OPTIONS:")
            print("   1. Check your internet connection")
            print("   2. Try using a VPN")
            print("   3. Wait and retry later")
            print("   4. Use a different network")
            return
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    print(f"✅ Model ready on {device}")
    
    # Generate image
    print("\n🎨 Generating your Danganronpa character...")
    prompt = "danganronpa style, red hair, gentle high school boy, detailed face, sharp features, anime style, high quality, masterpiece, best quality"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph"
    
    try:
        start_time = time.time()
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=768
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Image generated in {generation_time:.2f} seconds!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Save image
    print("\n💾 Saving your character...")
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/danganronpa_character_{timestamp}.png"
        
        image.save(filename)
        print(f"✅ Image saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    # Success!
    print("\n🎉 SUCCESS! Your Danganronpa character is ready!")
    print(f"📁 File: {filename}")
    print(f"🎭 Style: Danganronpa with red hair and gentle expression")
    print(f"📱 Size: {image.size}")
    print(f"⏱️  Generation time: {generation_time:.2f} seconds")
    
    # Open image
    try:
        print(f"\n🚀 Opening your character...")
        os.startfile(filename)
    except:
        print(f"   Image saved to: {filename}")
        print(f"   Please open it manually!")

if __name__ == "__main__":
    main()
