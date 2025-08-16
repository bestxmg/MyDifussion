#!/usr/bin/env python3
"""
Check Generated Image Quality
Verifies the image is not black and has proper content
"""

from PIL import Image
import numpy as np
import os

def check_image_quality(image_path):
    """Check if image is not black and has proper content"""
    try:
        # Open the image
        image = Image.open(image_path)
        print(f"📁 Image: {image_path}")
        print(f"📱 Size: {image.size}")
        print(f"🎨 Mode: {image.mode}")
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if image is mostly black (very low brightness)
        brightness = np.mean(img_array)
        print(f"💡 Image brightness: {brightness:.2f}")
        
        if brightness < 10:
            print("   ⚠️  WARNING: Image appears to be very dark/black!")
            return False
        
        # Check if image has some variation (not completely uniform)
        std_dev = np.std(img_array)
        print(f"📊 Image variation: {std_dev:.2f}")
        
        if std_dev < 5:
            print("   ⚠️  WARNING: Image appears to be too uniform!")
            return False
        
        # Check if image has reasonable dimensions
        if image.size[0] < 100 or image.size[1] < 100:
            print("   ⚠️  WARNING: Image dimensions seem too small!")
            return False
        
        print("   ✅ Image quality check passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking image: {e}")
        return False

def main():
    print("🔍 CHECKING GENERATED IMAGE QUALITY")
    print("=" * 50)
    
    # Check the most recent generated image
    generated_dir = "generated_images"
    
    if not os.path.exists(generated_dir):
        print("❌ Generated images directory not found")
        return
    
    # Find the most recent image
    image_files = [f for f in os.listdir(generated_dir) if f.endswith('.png')]
    
    if not image_files:
        print("❌ No generated images found")
        return
    
    # Sort by modification time (most recent first)
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(generated_dir, x)), reverse=True)
    
    latest_image = os.path.join(generated_dir, image_files[0])
    print(f"🔍 Checking latest image: {image_files[0]}")
    print()
    
    # Check image quality
    if check_image_quality(latest_image):
        print("\n🎉 SUCCESS! Your image looks good!")
        print("   It's not black and has proper content")
        
        # Try to open the image
        try:
            print(f"\n🚀 Opening your image...")
            os.startfile(latest_image)
        except:
            print(f"   Image saved to: {latest_image}")
            print(f"   Please open it manually to view!")
    else:
        print("\n⚠️  Image quality issues detected!")
        print("   The image may be corrupted or black")
        print("   This confirms the GPU black image issue")

if __name__ == "__main__":
    main()
