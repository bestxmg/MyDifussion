#!/usr/bin/env python3
"""
Check Generated Image - Analyze the INFP girl image
"""

import numpy as np
from PIL import Image
import os

def analyze_image(image_path):
    """Analyze the generated image"""
    try:
        # Open image
        img = Image.open(image_path)
        print(f"📁 Image: {os.path.basename(image_path)}")
        print(f"📏 Size: {img.size}")
        print(f"🎨 Mode: {img.mode}")
        print(f"💾 File size: {os.path.getsize(image_path)} bytes")
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Basic statistics
        brightness = np.mean(img_array)
        std_dev = np.std(img_array)
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        
        print(f"\n📊 Image Statistics:")
        print(f"   Brightness (mean): {brightness:.2f}")
        print(f"   Variation (std): {std_dev:.2f}")
        print(f"   Min value: {min_val}")
        print(f"   Max value: {max_val}")
        
        # Check if image is black/empty
        if brightness < 10:
            print("❌ Image appears to be black/empty")
            return False
        elif std_dev < 5:
            print("⚠️  Image has very low variation (might be too uniform)")
            return False
        else:
            print("✅ Image appears to have content")
            
        # Check color distribution
        if img.mode == 'RGB':
            r_mean = np.mean(img_array[:,:,0])
            g_mean = np.mean(img_array[:,:,1])
            b_mean = np.mean(img_array[:,:,2])
            
            print(f"\n🎨 Color Analysis:")
            print(f"   Red channel: {r_mean:.2f}")
            print(f"   Green channel: {r_mean:.2f}")
            print(f"   Blue channel: {b_mean:.2f}")
        
        # Try to open the image
        print(f"\n🚀 Opening image...")
        try:
            os.startfile(image_path)
            print("✅ Image opened successfully")
        except Exception as e:
            print(f"❌ Could not open image: {e}")
            print(f"   Please open manually: {image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return False

def main():
    print("🔍 CHECKING GENERATED INFP GIRL IMAGE")
    print("=" * 50)
    
    # Check the gentle generation result
    image_path = "generated_images/infp_girl_gentle_1755333223.png"
    
    if os.path.exists(image_path):
        print(f"Found image: {image_path}")
        analyze_image(image_path)
    else:
        print(f"Image not found: {image_path}")
    
    # Also check if there are other recent images
    print(f"\n📁 Recent generated images:")
    try:
        files = os.listdir("generated_images")
        for file in sorted(files, reverse=True)[:5]:
            file_path = os.path.join("generated_images", file)
            size = os.path.getsize(file_path)
            print(f"   {file} ({size} bytes)")
    except Exception as e:
        print(f"Error listing directory: {e}")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
