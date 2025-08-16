#!/usr/bin/env python3
"""
GPU Memory Monitor - Check if your GPU memory usage is normal
"""

import torch
import time

def check_gpu_memory():
    """Check current GPU memory status"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    try:
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total_memory - allocated
        
        # Calculate percentages
        allocated_percent = (allocated / total_memory) * 100
        reserved_percent = (reserved / total_memory) * 100
        free_percent = (free / total_memory) * 100
        
        print("🔍 GPU MEMORY STATUS")
        print("=" * 40)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_memory:.1f} GB")
        print()
        print("📊 Current Usage:")
        print(f"   Allocated: {allocated:.2f} GB ({allocated_percent:.1f}%)")
        print(f"   Reserved:  {reserved:.2f} GB ({reserved_percent:.1f}%)")
        print(f"   Free:      {free:.2f} GB ({free_percent:.1f}%)")
        print()
        
        # Analysis
        if allocated_percent > 80:
            print("🟡 HIGH MEMORY USAGE - This is normal for Stable Diffusion!")
            print("   - Model is loaded and ready")
            print("   - GPU is working efficiently")
            print("   - You can still generate images")
        elif allocated_percent > 60:
            print("🟢 MODERATE MEMORY USAGE - Good balance")
        else:
            print("🟢 LOW MEMORY USAGE - Plenty of space available")
        
        print()
        print("💡 Expected Memory Usage:")
        print("   - Stable Diffusion model: ~3.5-4.0 GB")
        print("   - Generation buffer: ~0.5-1.0 GB")
        print("   - Total normal usage: ~4.0-4.5 GB")
        
        return allocated, total_memory
        
    except Exception as e:
        print(f"❌ Error checking GPU memory: {e}")
        return None, None

def monitor_memory_continuously(duration=30):
    """Monitor memory usage continuously"""
    print(f"\n📈 Monitoring GPU memory for {duration} seconds...")
    print("Press Ctrl+C to stop early")
    print()
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            allocated, total = check_gpu_memory()
            
            if allocated and total:
                usage_percent = (allocated / total) * 100
                print(f"⏰ {time.strftime('%H:%M:%S')} - Usage: {usage_percent:.1f}%")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    
    print("\n✅ Memory monitoring completed")

def main():
    print("🔍 GPU MEMORY MONITOR - Check if your usage is normal")
    print("=" * 60)
    
    # Single check
    allocated, total = check_gpu_memory()
    
    if allocated and total:
        print("\n" + "=" * 60)
        print("📋 SUMMARY:")
        
        if allocated > 3.0:
            print("✅ NORMAL: Your GPU memory usage is expected for Stable Diffusion")
            print("   - Model is loaded and ready")
            print("   - This is the correct behavior")
        elif allocated > 1.0:
            print("⚠️  MODERATE: Some memory usage detected")
            print("   - May need to load model")
        else:
            print("🟢 LOW: Minimal GPU memory usage")
            print("   - Model may not be loaded")
    
    print("\n" + "=" * 60)
    print("Options:")
    print("1. Press Enter to exit")
    print("2. Type 'monitor' to continuously monitor memory")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'monitor':
        monitor_memory_continuously()
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
