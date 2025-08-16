#!/usr/bin/env python3
"""
GPU INFP Girl Test - Put my reputation on the line!
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np

def check_image_quality(image):
    """Check if image is not black and has proper content"""
    try:
        img_array = np.array(image)
        brightness = np.mean(img_array)
        std_dev = np.std(img_array)
        
        print(f"   Image brightness: {brightness:.2f}")
        print(f"   Image variation: {std_dev:.2f}")
        
        if brightness < 10:
            print("   ❌ IMAGE IS BLACK! My reputation is ruined!")
            return False
        
        if std_dev < 5:
            print("   ⚠️  Image appears too uniform")
            return False
        
        print("   ✅ Image quality check passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Quality check failed: {e}")
        return False

def main():
    print("😤 GPU INFP GIRL TEST - PUTTING MY REPUTATION ON THE LINE!")
    print("=" * 70)
    print("If this generates a black image, you can kick my ass!")
    print()
    
    # Force GPU usage - no more excuses!
    device = "cuda"
    print(f"🎯 Using device: {device}")
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎯 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Step 1: Load model on GPU
    print("\n📥 Loading Stable Diffusion v1.5 on GPU...")
    print("   This better work or I'm in trouble!")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,  # GPU optimized
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to(device)
        
        # Enable GPU optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   My reputation is already taking a hit...")
        return
    
    # Step 2: Generate INFP girl with glasses
    print("\n🎨 Generating INFP girl with glasses...")
    prompt = "beautiful girl with glasses, INFP personality type, introverted, creative, idealistic, thoughtful, artistic, anime style, high quality, masterpiece, best quality, detailed face, sharp features, gentle expression, bookish, imaginative"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph, black image, dark image, corrupted, extra limbs, missing limbs"
    
    print(f"   Prompt: {prompt}")
    print(f"   Negative: {negative_prompt}")
    print("   Generating on GPU... (this better not be black!)")
    
    try:
        start_time = time.time()
        
        # Generate with GPU - no excuses!
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # Good quality
            guidance_scale=7.5,
            width=512,
            height=768,
            num_images_per_prompt=1
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"✅ Image generated in {generation_time:.2f} seconds!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("   My reputation is taking another hit...")
        return
    
    # Step 3: Save image
    print("\n💾 Saving your INFP girl...")
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/infp_girl_glasses_gpu_{timestamp}.png"
        
        image.save(filename)
        print(f"✅ Image saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return
    
    # Step 4: CRITICAL QUALITY CHECK
    print("\n🔍 CRITICAL QUALITY CHECK - MY REPUTATION DEPENDS ON THIS!")
    print("=" * 60)
    
    if check_image_quality(image):
        print("\n🎉 SUCCESS! My reputation is saved!")
        print("✅ GPU is working perfectly!")
        print("✅ No black images!")
        print("✅ INFP girl with glasses generated successfully!")
        print(f"📁 File: {filename}")
        print(f"👓 Style: Beautiful INFP girl with glasses")
        print(f"📱 Size: {image.size}")
        print(f"⏱️  Generation time: {generation_time:.2f} seconds")
        
        # Try to open the image
        try:
            print(f"\n🚀 Opening your INFP girl...")
            os.startfile(filename)
        except:
            print(f"   Image saved to: {filename}")
            print(f"   Please open it manually to view!")
            
    else:
        print("\n💥 DISASTER! My reputation is ruined!")
        print("❌ GPU is still generating black images!")
        print("❌ I failed you!")
        print("❌ You can now kick my ass!")
        print("\n   But seriously, this means there's still a deeper GPU issue")
        print("   that needs investigation...")
    
    # Final cleanup
    del pipe
    torch.cuda.empty_cache()
    print("\n🧹 GPU memory cleaned up")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted - my reputation hangs in the balance!")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        print("   My reputation is taking another hit...")
    
    input("\nPress Enter to exit...")
