#!/usr/bin/env python3
"""
VPN-Friendly GPU Generator - Prevents VPN disconnections
Monitors system resources and automatically throttles generation
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np
import psutil
import threading
import warnings
warnings.filterwarnings("ignore")

class ResourceMonitor:
    """Monitor system resources to prevent VPN disconnections"""
    
    def __init__(self):
        self.vpn_threshold = 0.85  # 85% resource usage threshold
        self.gpu_threshold = 0.90  # 90% GPU memory threshold
        self.monitoring = False
        self.current_usage = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_memory_percent': 0,
            'network_io': 0
        }
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        print("🔍 Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU and Memory
                self.current_usage['cpu_percent'] = psutil.cpu_percent(interval=1)
                self.current_usage['memory_percent'] = psutil.virtual_memory().percent
                
                # GPU Memory
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory
                    self.current_usage['gpu_memory_percent'] = (gpu_memory / gpu_total) * 100
                
                # Network I/O (VPN indicator)
                net_io = psutil.net_io_counters()
                self.current_usage['network_io'] = net_io.bytes_sent + net_io.bytes_recv
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(5)
    
    def check_safe_to_generate(self):
        """Check if it's safe to generate without VPN issues"""
        cpu_ok = self.current_usage['cpu_percent'] < (self.vpn_threshold * 100)
        memory_ok = self.current_usage['memory_percent'] < (self.vpn_threshold * 100)
        gpu_ok = self.current_usage['gpu_memory_percent'] < (self.gpu_threshold * 100)
        
        if not cpu_ok:
            print(f"⚠️  CPU usage too high: {self.current_usage['cpu_percent']:.1f}%")
        if not memory_ok:
            print(f"⚠️  Memory usage too high: {self.current_usage['memory_percent']:.1f}%")
        if not gpu_ok:
            print(f"⚠️  GPU memory usage too high: {self.current_usage['gpu_memory_percent']:.1f}%")
        
        return cpu_ok and memory_ok and gpu_ok
    
    def get_resource_status(self):
        """Get current resource status"""
        return self.current_usage.copy()
    
    def wait_for_resources(self, max_wait=60):
        """Wait for resources to become available"""
        print("⏳ Waiting for resources to become available...")
        start_time = time.time()
        
        while not self.check_safe_to_generate():
            if time.time() - start_time > max_wait:
                print("⏰ Timeout waiting for resources")
                return False
            
            current = self.get_resource_status()
            print(f"   CPU: {current['cpu_percent']:.1f}%, "
                  f"Memory: {current['memory_percent']:.1f}%, "
                  f"GPU: {current['gpu_memory_percent']:.1f}%")
            
            time.sleep(5)
        
        print("✅ Resources are now available")
        return True

class VPNFriendlyGenerator:
    """VPN-friendly image generator with resource monitoring"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resource_monitor = ResourceMonitor()
        self.pipe = None
        self.generation_settings = {
            'safe': {
                'num_inference_steps': 15,
                'guidance_scale': 6.0,
                'width': 512,
                'height': 512
            },
            'balanced': {
                'num_inference_steps': 20,
                'guidance_scale': 7.0,
                'width': 512,
                'height': 768
            },
            'quality': {
                'num_inference_steps': 25,
                'guidance_scale': 7.5,
                'width': 512,
                'height': 768
            }
        }
    
    def load_model(self):
        """Load Stable Diffusion model with resource awareness"""
        print("📥 Loading Stable Diffusion model...")
        
        try:
            # Check resources before loading
            if not self.resource_monitor.check_safe_to_generate():
                print("⚠️  Resources too high, waiting...")
                if not self.resource_monitor.wait_for_resources():
                    return False
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def select_generation_settings(self):
        """Select generation settings based on current resources"""
        current = self.resource_monitor.get_resource_status()
        
        if current['cpu_percent'] < 50 and current['memory_percent'] < 60:
            print("🚀 Using QUALITY settings (resources available)")
            return self.generation_settings['quality']
        elif current['cpu_percent'] < 70 and current['memory_percent'] < 75:
            print("⚖️  Using BALANCED settings (moderate resources)")
            return self.generation_settings['balanced']
        else:
            print("🛡️  Using SAFE settings (conservative resources)")
            return self.generation_settings['safe']
    
    def generate_image(self, prompt, negative_prompt="", quality_level="auto"):
        """Generate image with VPN protection"""
        if not self.pipe:
            print("❌ Model not loaded")
            return None
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Wait for resources if needed
            if not self.resource_monitor.wait_for_resources():
                print("❌ Cannot generate - resources too high")
                return None
            
            # Select appropriate settings
            if quality_level == "auto":
                settings = self.select_generation_settings()
            else:
                settings = self.generation_settings.get(quality_level, self.generation_settings['safe'])
            
            print(f"🎨 Generating with {quality_level} settings...")
            print(f"   Steps: {settings['num_inference_steps']}, "
                  f"Guidance: {settings['guidance_scale']}, "
                  f"Size: {settings['width']}x{settings['height']}")
            
            # Generate with resource monitoring
            start_time = time.time()
            
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=settings['num_inference_steps'],
                guidance_scale=settings['guidance_scale'],
                width=settings['width'],
                height=settings['height'],
                num_images_per_prompt=1
            ).images[0]
            
            generation_time = time.time() - start_time
            
            # Check final resource status
            final_status = self.resource_monitor.get_resource_status()
            print(f"✅ Generation completed in {generation_time:.2f} seconds")
            print(f"   Final CPU: {final_status['cpu_percent']:.1f}%")
            print(f"   Final Memory: {final_status['memory_percent']:.1f}%")
            print(f"   Final GPU: {final_status['gpu_memory_percent']:.1f}%")
            
            return image
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
        
        finally:
            self.resource_monitor.stop_monitoring()
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipe:
            del self.pipe
            self.pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 Resources cleaned up")

def main():
    print("🛡️  VPN-FRIENDLY GPU GENERATOR")
    print("=" * 50)
    print("This generator protects your VPN connection!")
    print("It automatically throttles itself to prevent disconnections.")
    print()
    
    # Initialize generator
    generator = VPNFriendlyGenerator()
    
    # Load model
    if not generator.load_model():
        print("❌ Failed to load model")
        return
    
    # Generate INFP girl with glasses
    prompt = "beautiful girl with glasses, INFP personality type, introverted, creative, idealistic, thoughtful, artistic, anime style, high quality, masterpiece, best quality, detailed face, sharp features, gentle expression, bookish, imaginative"
    negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, signature, realistic, photograph, black image, dark image, corrupted, extra limbs, missing limbs"
    
    print(f"\n🎨 Generating: {prompt}")
    print("   (VPN connection will be protected)")
    
    # Generate with VPN protection
    image = generator.generate_image(prompt, negative_prompt, quality_level="auto")
    
    if image:
        # Save image
        try:
            os.makedirs("generated_images", exist_ok=True)
            timestamp = int(time.time())
            filename = f"generated_images/infp_girl_vpn_safe_{timestamp}.png"
            
            image.save(filename)
            print(f"\n✅ Image saved: {filename}")
            
            # Quality check
            img_array = np.array(image)
            brightness = np.mean(img_array)
            std_dev = np.std(img_array)
            
            print(f"   Brightness: {brightness:.2f}")
            print(f"   Variation: {std_dev:.2f}")
            
            if brightness > 10:
                print("✅ SUCCESS! No black image, VPN protected!")
                print("🎉 Your INFP girl with glasses is ready!")
                
                # Try to open
                try:
                    os.startfile(filename)
                except:
                    print(f"   Please open manually: {filename}")
            else:
                print("❌ Image appears to be black")
                
        except Exception as e:
            print(f"❌ Saving failed: {e}")
    else:
        print("❌ Generation failed")
    
    # Cleanup
    generator.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
