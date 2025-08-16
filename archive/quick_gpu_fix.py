#!/usr/bin/env python3
"""
QUICK GPU PERFORMANCE MONITORING FIX
Specifically targets the ".." issue in NVIDIA GeForce Experience
"""

import subprocess
import time
import os

def print_status(message, status="INFO"):
    """Print status message with appropriate icon"""
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "FIX": "🔧"
    }
    icon = icons.get(status, "ℹ️")
    print(f"{icon} {message}")

def check_admin():
    """Check if running as administrator"""
    try:
        result = subprocess.run(['net', 'session'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def restart_nvidia_services():
    """Restart key NVIDIA services"""
    print_status("Restarting NVIDIA services...", "FIX")
    
    services = ["nvsvc", "NvTelemetry"]
    
    for service in services:
        try:
            print_status(f"Stopping {service}...", "INFO")
            subprocess.run(['sc', 'stop', service], capture_output=True, timeout=10)
            time.sleep(2)
            
            print_status(f"Starting {service}...", "INFO")
            result = subprocess.run(['sc', 'start', service], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                print_status(f"Successfully restarted {service}", "SUCCESS")
            else:
                print_status(f"Failed to restart {service}", "WARNING")
                
        except Exception as e:
            print_status(f"Error with {service}: {e}", "ERROR")
    
    time.sleep(5)  # Wait for services to stabilize

def test_gpu_monitoring():
    """Test if GPU monitoring is working"""
    print_status("Testing GPU performance monitoring...", "INFO")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                             capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output and not output.startswith(".."):
                print_status("GPU monitoring is working!", "SUCCESS")
                print(f"   Raw output: {output}")
                return True
            else:
                print_status("GPU monitoring still showing '..' values", "WARNING")
                return False
        else:
            print_status("nvidia-smi command failed", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Error testing GPU monitoring: {e}", "ERROR")
        return False

def run_display_driver_reset():
    """Attempt to reset display driver"""
    print_status("Attempting display driver reset...", "FIX")
    
    try:
        # This is a Windows-specific command to reset display driver
        print_status("Running display driver reset command...", "INFO")
        result = subprocess.run(['powershell', '-Command', 
                               'Get-PnpDevice | Where-Object {$_.FriendlyName -like "*NVIDIA*"} | Restart-PnpDevice'], 
                             capture_output=True, timeout=30)
        
        if result.returncode == 0:
            print_status("Display driver reset command completed", "SUCCESS")
        else:
            print_status("Display driver reset command failed", "WARNING")
            print(f"   Error: {result.stderr.decode()}")
            
    except Exception as e:
        print_status(f"Error during display driver reset: {e}", "ERROR")

def main():
    """Main function"""
    print("=" * 60)
    print("🔧 QUICK GPU PERFORMANCE MONITORING FIX")
    print("=" * 60)
    print("This tool will attempt to fix the '..' issue in NVIDIA GeForce Experience")
    print("where GPU performance statistics are not displaying correctly.")
    print()
    
    # Check if running as admin
    if not check_admin():
        print_status("⚠️  WARNING: This script should be run as Administrator for best results", "WARNING")
        print_status("   Right-click on PowerShell/Command Prompt and select 'Run as Administrator'", "INFO")
        print()
    
    # Test current state
    print_status("Testing current GPU monitoring status...", "INFO")
    initial_status = test_gpu_monitoring()
    
    if initial_status:
        print_status("GPU monitoring is already working correctly!", "SUCCESS")
        return
    
    print()
    print_status("GPU monitoring issue detected. Starting fixes...", "INFO")
    
    # Fix 1: Restart NVIDIA services
    restart_nvidia_services()
    
    # Test after service restart
    print()
    print_status("Testing after service restart...", "INFO")
    if test_gpu_monitoring():
        print_status("Service restart fixed the issue!", "SUCCESS")
        return
    
    # Fix 2: Display driver reset
    print()
    run_display_driver_reset()
    
    # Test after driver reset
    print()
    print_status("Testing after driver reset...", "INFO")
    if test_gpu_monitoring():
        print_status("Driver reset fixed the issue!", "SUCCESS")
        return
    
    # If we get here, the quick fixes didn't work
    print()
    print_status("Quick fixes did not resolve the issue", "WARNING")
    print_status("Manual intervention required:", "INFO")
    print()
    print("📋 MANUAL FIXES TO TRY:")
    print("   1. Restart your computer")
    print("   2. Update NVIDIA drivers from nvidia.com")
    print("   3. In Device Manager, disable then enable your GTX 1650")
    print("   4. Reset NVIDIA Control Panel settings")
    print("   5. Check Windows Event Viewer for NVIDIA errors")
    print()
    print_status("After trying manual fixes, run this script again to test", "INFO")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Fix interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please try running the script again")
    
    input("\nPress Enter to exit...")
