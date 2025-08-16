#!/usr/bin/env python3
"""
Test Programmer Image Generation - With Error Checking
Generates "ultimate programmer, with blue hair and gentle personality"
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np

def check_image_quality(image):
    """Check if image is not black and has proper content"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if image is mostly black (very low brightness)
    brightness = np.mean(img_array)
    print(f"   Image brightness: {brightness:.2f}")
    
    if brightness < 10:
        print("   ⚠️  WARNING: Image appears to be very dark/black!")
        return False
    
    # Check if image has some variation (not completely uniform)
    std_dev = np.std(img_array)
    print(f"   Image variation: {std_dev:.2f}")
    
    if std_dev < 5:
        print("   ⚠️  WARNING: Image appears to be too uniform!")
        return False
    
    # Check if image has reasonable dimensions
    if image.size[0] < 100 or image.size[1] < 100:
        print("   ⚠️  WARNING: Image dimensions seem too small!")
        return False
    
    print("   ✅ Image quality check passed!")
    return True

def main():
    print("👨‍💻 GENERATING ULTIMATE PROGRAMMER WITH BLUE HAIR!")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = "cuda"
    else:
        print("⚠️  CUDA not available, using CPU (will be much slower)")
        device = "cpu"
    
    # Step 1: Load model
    print("\n📥 Loading Stable Diffusion v1.5...")
    print("   This will download ~4GB model file (first time only)")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            print("   ✅ Memory optimizations enabled")
        
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   This might be a GPU memory issue or missing dependencies")
        return
    
    # Step 2: Generate image
    print("\n🎨 Generating your ultimate programmer...")
    prompt = "ultimate programmer, with blue hair and gentle personality, professional developer, coding, computer science, anime style, high quality, masterpiece, best quality, detailed face, sharp features"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph, black image, dark image, corrupted"
    
    print(f"   Prompt: {prompt}")
    print(f"   Negative: {negative_prompt}")
    print("   Generating... (this may take 1-3 minutes)")
    
    try:
        start_time = time.time()
        
        # Generate with multiple attempts if needed
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"   Attempt {attempt + 1}/{max_attempts}...")
            
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=512,
                height=768,
                num_images_per_prompt=1,
                seed=42 + attempt  # Different seed each attempt
            ).images[0]
            
            # Check image quality
            if check_image_quality(image):
                break
            elif attempt < max_attempts - 1:
                print("   🔄 Retrying with different seed...")
            else:
                print("   ⚠️  All attempts resulted in poor quality images")
        
        generation_time = time.time() - start_time
        print(f"✅ Image generated in {generation_time:.2f} seconds!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("   This might indicate a GPU memory issue or model problem")
        return
    
    # Step 3: Save image
    print("\n💾 Saving your programmer image...")
    try:
        os.makedirs("generated_images", exist_ok=True)
        timestamp = int(time.time())
        filename = f"generated_images/ultimate_programmer_blue_hair_{timestamp}.png"
        
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
    
    # Final quality check
    print("\n🔍 Final quality verification...")
    if check_image_quality(image):
        print("🎯 PERFECT! Image quality is excellent!")
    else:
        print("⚠️  Image quality issues detected - you may want to regenerate")
    
    # Try to open the image
    try:
        print(f"\n🚀 Opening your programmer image...")
        os.startfile(filename)
    except:
        print(f"   Image saved to: {filename}")
        print(f"   Please open it manually to view your programmer!")

if __name__ == "__main__":
    main()
