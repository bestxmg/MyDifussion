#!/usr/bin/env python3
"""
Quick Resource Checker - See system status before GPU generation
"""

import psutil
import torch
import time

def check_system_resources():
    """Check current system resource usage"""
    print("🔍 QUICK RESOURCE CHECK")
    print("=" * 40)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"💻 CPU: {cpu_percent:.1f}% usage ({cpu_count} cores)")
    
    # Memory
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_gb = memory.total / (1024**3)
    memory_used_gb = memory.used / (1024**3)
    print(f"🧠 Memory: {memory_percent:.1f}% usage ({memory_used_gb:.1f}/{memory_gb:.1f} GB)")
    
    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100
        
        print(f"🎮 GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory_percent:.1f}% usage")
        print(f"   Allocated: {gpu_memory_allocated:.2f} GB")
        print(f"   Reserved: {gpu_memory_reserved:.2f} GB")
        print(f"   Total: {gpu_memory_total:.1f} GB")
    else:
        print("🎮 GPU: CUDA not available")
    
    # Network
    net_io = psutil.net_io_counters()
    net_sent_mb = net_io.bytes_sent / (1024**2)
    net_recv_mb = net_io.bytes_recv / (1024**2)
    print(f"🌐 Network: Sent {net_sent_mb:.1f} MB, Received {net_recv_mb:.1f} MB")
    
    # VPN Safety Assessment
    print("\n🛡️  VPN SAFETY ASSESSMENT:")
    print("-" * 30)
    
    vpn_safe = True
    warnings = []
    
    if cpu_percent > 80:
        vpn_safe = False
        warnings.append(f"CPU usage too high: {cpu_percent:.1f}%")
    
    if memory_percent > 85:
        vpn_safe = False
        warnings.append(f"Memory usage too high: {memory_percent:.1f}%")
    
    if torch.cuda.is_available() and gpu_memory_percent > 90:
        vpn_safe = False
        warnings.append(f"GPU memory usage too high: {gpu_memory_percent:.1f}%")
    
    if vpn_safe:
        print("✅ VPN connection should be safe")
        print("   Resources are within safe limits")
    else:
        print("⚠️  VPN connection may be at risk!")
        for warning in warnings:
            print(f"   - {warning}")
        print("   Consider waiting or using conservative settings")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if cpu_percent < 50 and memory_percent < 60:
        print("🚀 Use QUALITY settings - plenty of resources")
    elif cpu_percent < 70 and memory_percent < 75:
        print("⚖️  Use BALANCED settings - moderate resources")
    else:
        print("🛡️  Use SAFE settings - conserve resources")
    
    if not vpn_safe:
        print("⏳ Wait for resources to become available")
        print("🔄 Close unnecessary applications")
        print("💤 Consider restarting if issues persist")

def monitor_resources(duration=30):
    """Monitor resources for a specified duration"""
    print(f"\n📊 MONITORING RESOURCES FOR {duration} SECONDS...")
    print("=" * 50)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0)
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            gpu_percent = (gpu_memory / gpu_total) * 100
            gpu_str = f"GPU: {gpu_percent:.1f}%"
        else:
            gpu_str = "GPU: N/A"
        
        elapsed = time.time() - start_time
        print(f"[{elapsed:5.1f}s] CPU: {cpu_percent:5.1f}% | Memory: {memory_percent:5.1f}% | {gpu_str}")
        
        # Check for VPN risk
        if cpu_percent > 80 or memory_percent > 85:
            print("   ⚠️  VPN RISK DETECTED!")
    
    print("✅ Monitoring completed")

def main():
    print("🚀 RESOURCE CHECKER - VPN PROTECTION TOOL")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Quick resource check")
        print("2. Monitor resources (30 seconds)")
        print("3. Monitor resources (custom duration)")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            check_system_resources()
        elif choice == "2":
            monitor_resources(30)
        elif choice == "3":
            try:
                duration = int(input("Enter duration in seconds: "))
                monitor_resources(duration)
            except ValueError:
                print("Invalid duration")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
