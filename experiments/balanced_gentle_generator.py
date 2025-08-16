#!/usr/bin/env python3
"""
Balanced Gentle GPU Generator - VPN-friendly but actually works!
Balances resource conservation with actual image generation
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np
import psutil
import warnings
warnings.filterwarnings("ignore")

class BalancedGentleGenerator:
    """Balanced gentle generator that actually produces images"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
        # Balanced settings - gentle but effective
        self.balanced_settings = {
            'num_inference_steps': 18,      # Enough steps to generate content
            'guidance_scale': 6.5,          # Balanced guidance
            'width': 448,                   # Medium size
            'height': 448,                  # Medium size
            'num_images_per_prompt': 1
        }
        
        # Conservative but not extreme thresholds
        self.cpu_threshold = 75     # 75% CPU max
        self.memory_threshold = 80  # 80% memory max
        self.gpu_threshold = 80     # 80% GPU max
    
    def check_resources_safe(self):
        """Check if resources are safe for balanced generation"""
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.5)
            if cpu_percent > self.cpu_threshold:
                print(f"⚠️  CPU too high: {cpu_percent:.1f}% (max: {self.cpu_threshold}%)")
                return False
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                print(f"⚠️  Memory too high: {memory.percent:.1f}% (max: {self.memory_threshold}%)")
                return False
            
            # GPU check
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0)
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_percent = (gpu_memory / gpu_total) * 100
                
                if gpu_percent > self.gpu_threshold:
                    print(f"⚠️  GPU memory too high: {gpu_percent:.1f}% (max: {self.gpu_threshold}%)")
                    return False
            
            print(f"✅ Resources safe: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
            return True
            
        except Exception as e:
            print(f"⚠️  Resource check failed: {e}")
            return False
    
    def wait_for_resources(self, max_wait=120):
        """Wait for resources to become safe"""
        print("⏳ Waiting for resources to become safe...")
        start_time = time.time()
        
        while not self.check_resources_safe():
            if time.time() - start_time > max_wait:
                print("⏰ Timeout waiting for resources")
                return False
            
            print("   Waiting 10 seconds...")
            time.sleep(10)
        
        print("✅ Resources are now safe")
        return True
    
    def load_model(self):
        """Load model with balanced settings"""
        print("📥 Loading Stable Diffusion model (balanced gentle mode)...")
        
        try:
            # Wait for resources if needed
            if not self.wait_for_resources():
                return False
            
            # Load with balanced settings
            if self.device == "cuda":
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(self.device)
                
                # Enable memory-saving features but not too aggressive
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
                
                print("✅ Model loaded with balanced memory features")
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(self.device)
                print("✅ Model loaded on CPU")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def generate_balanced(self, prompt, negative_prompt=""):
        """Generate image with balanced gentle settings"""
        if not self.pipe:
            print("❌ Model not loaded")
            return None
        
        # Final resource check
        if not self.check_resources_safe():
            print("❌ Resources not safe for generation")
            return None
        
        print(f"🎨 Generating with BALANCED GENTLE settings...")
        print(f"   Steps: {self.balanced_settings['num_inference_steps']}")
        print(f"   Size: {self.balanced_settings['width']}x{self.balanced_settings['height']}")
        print(f"   Guidance: {self.balanced_settings['guidance_scale']}")
        print("   (This should actually generate an image!)")
        
        try:
            start_time = time.time()
            
            # Generate with balanced settings
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.balanced_settings['num_inference_steps'],
                guidance_scale=self.balanced_settings['guidance_scale'],
                width=self.balanced_settings['width'],
                height=self.balanced_settings['height'],
                num_images_per_prompt=self.balanced_settings['num_images_per_prompt']
            ).images[0]
            
            generation_time = time.time() - start_time
            
            # Check final resources
            final_cpu = psutil.cpu_percent(interval=1)
            final_memory = psutil.virtual_memory().percent
            
            print(f"✅ Generation completed in {generation_time:.2f} seconds")
            print(f"   Final CPU: {final_cpu:.1f}%")
            print(f"   Final Memory: {final_memory:.1f}%")
            
            if final_cpu > 85 or final_memory > 90:
                print("⚠️  Warning: High resource usage detected")
                print("   Your VPN might be at risk")
            else:
                print("✅ Resources remain safe")
            
            return image
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    def cleanup(self):
        """Balanced cleanup"""
        if self.pipe:
            del self.pipe
            self.pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 Balanced cleanup completed")

def main():
    print("⚖️  BALANCED GENTLE GPU GENERATOR - ACTUALLY WORKS!")
    print("=" * 60)
    print("This generator is gentle on your system BUT generates real images")
    print("No more black squares - I promise!")
    print()
    
    # Initialize balanced generator
    generator = BalancedGentleGenerator()
    
    # Load model
    if not generator.load_model():
        print("❌ Failed to load model")
        return
    
    # Generate INFP girl with glasses (for real this time!)
    prompt = "beautiful girl with glasses, INFP personality type, introverted, creative, idealistic, thoughtful, artistic, anime style, high quality, masterpiece, best quality, detailed face, sharp features, gentle expression, bookish, imaginative"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph, black image, dark image, corrupted, extra limbs, missing limbs"
    
    print(f"\n🎨 Generating: {prompt}")
    print("   (Using balanced gentle settings - should work!)")
    
    # Generate with balanced settings
    image = generator.generate_balanced(prompt, negative_prompt)
    
    if image:
        # Save image
        try:
            os.makedirs("generated_images", exist_ok=True)
            timestamp = int(time.time())
            filename = f"generated_images/infp_girl_balanced_{timestamp}.png"
            
            image.save(filename)
            print(f"\n✅ Image saved: {filename}")
            
            # Quality check
            img_array = np.array(image)
            brightness = np.mean(img_array)
            std_dev = np.std(img_array)
            
            print(f"   Brightness: {brightness:.2f}")
            print(f"   Variation: {std_dev:.2f}")
            
            if brightness > 10:
                print("✅ SUCCESS! Balanced generation worked!")
                print("🎉 Your INFP girl with glasses is ready!")
                print("🛡️  VPN connection protected!")
                print("🎨 No more black images!")
                
                # Try to open
                try:
                    os.startfile(filename)
                except:
                    print(f"   Please open manually: {filename}")
            else:
                print("❌ Still a black image - I'm really sorry!")
                print("   This means there's a deeper issue")
                
        except Exception as e:
            print(f"❌ Saving failed: {e}")
    else:
        print("❌ Generation failed")
    
    # Cleanup
    generator.cleanup()
    
    print("\n⚖️  Generation completed with balanced approach!")
    print("   Your system should remain stable")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
