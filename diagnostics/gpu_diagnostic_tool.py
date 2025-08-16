#!/usr/bin/env python3
"""
COMPREHENSIVE GPU DIAGNOSTIC TOOL
Diagnoses and potentially fixes NVIDIA GPU performance monitoring issues
"""

import os
import sys
import subprocess
import time
import json
import platform
from pathlib import Path

class GPUDiagnosticTool:
    def __init__(self):
        self.system_info = {}
        self.gpu_info = {}
        self.diagnostic_results = {}
        self.fixes_applied = []
        
    def print_header(self, title):
        print("\n" + "="*60)
        print(f"🔍 {title}")
        print("="*60)
    
    def print_section(self, title):
        print(f"\n📋 {title}")
        print("-" * 40)
    
    def print_status(self, message, status="INFO"):
        status_icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "FIX": "🔧"
        }
        icon = status_icons.get(status, "ℹ️")
        print(f"{icon} {message}")
    
    def get_system_info(self):
        """Gather basic system information"""
        self.print_section("System Information")
        
        self.system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        for key, value in self.system_info.items():
            self.print_status(f"{key.replace('_', ' ').title()}: {value}")
    
    def check_nvidia_drivers(self):
        """Check NVIDIA driver installation and version"""
        self.print_section("NVIDIA Driver Check")
        
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        driver_version, gpu_name = line.split(', ')
                        self.gpu_info['driver_version'] = driver_version
                        self.gpu_info['gpu_name'] = gpu_name
                        self.print_status(f"Driver Version: {driver_version}", "SUCCESS")
                        self.print_status(f"GPU Name: {gpu_name}", "SUCCESS")
            else:
                self.print_status("nvidia-smi command failed", "ERROR")
                return False
                
        except FileNotFoundError:
            self.print_status("nvidia-smi not found - NVIDIA drivers may not be installed", "ERROR")
            return False
        except subprocess.TimeoutExpired:
            self.print_status("nvidia-smi command timed out", "WARNING")
            return False
        except Exception as e:
            self.print_status(f"Error checking NVIDIA drivers: {e}", "ERROR")
            return False
        
        return True
    
    def check_nvidia_services(self):
        """Check if NVIDIA services are running properly"""
        self.print_section("NVIDIA Services Check")
        
        services_to_check = [
            "nvsvc",           # NVIDIA Display Driver Service
            "NvTelemetry",     # NVIDIA Telemetry Service
            "NVDisplay.ContainerLocalSystem", # NVIDIA Display Container
        ]
        
        all_running = True
        
        for service in services_to_check:
            try:
                result = subprocess.run(['sc', 'query', service], 
                                     capture_output=True, text=True, timeout=5)
                
                if "RUNNING" in result.stdout:
                    self.print_status(f"{service}: Running", "SUCCESS")
                else:
                    self.print_status(f"{service}: Not running or stopped", "WARNING")
                    all_running = False
                    
            except Exception as e:
                self.print_status(f"Could not check {service}: {e}", "WARNING")
                all_running = False
        
        return all_running
    
    def check_gpu_performance_monitoring(self):
        """Test GPU performance monitoring capabilities"""
        self.print_section("GPU Performance Monitoring Test")
        
        try:
            # Test basic GPU info
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_util, mem_used, mem_total, temp = parts
                            
                            # Check if values are numeric
                            try:
                                float(gpu_util)
                                float(mem_used)
                                float(mem_total)
                                float(temp)
                                self.print_status("Performance monitoring: Working", "SUCCESS")
                                self.print_status(f"GPU Utilization: {gpu_util}%", "INFO")
                                self.print_status(f"Memory Used: {mem_used} MB / {mem_total} MB", "INFO")
                                self.print_status(f"Temperature: {temp}°C", "INFO")
                                return True
                            except ValueError:
                                self.print_status("Performance monitoring: Values not numeric (similar to '..' issue)", "ERROR")
                                return False
            else:
                self.print_status("Could not query GPU performance data", "ERROR")
                return False
                
        except Exception as e:
            self.print_status(f"Error testing performance monitoring: {e}", "ERROR")
            return False
    
    def check_cuda_installation(self):
        """Check CUDA installation and compatibility"""
        self.print_section("CUDA Installation Check")
        
        try:
            # Check CUDA version
            result = subprocess.run(['nvcc', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Extract CUDA version from output
                output = result.stdout
                if "release" in output.lower():
                    cuda_version = output.split("release")[1].split(",")[0].strip()
                    self.print_status(f"CUDA Version: {cuda_version}", "SUCCESS")
                    return True
                else:
                    self.print_status("CUDA installed but version unclear", "WARNING")
                    return False
            else:
                self.print_status("CUDA not installed or not in PATH", "WARNING")
                return False
                
        except FileNotFoundError:
            self.print_status("CUDA compiler (nvcc) not found", "WARNING")
            return False
        except Exception as e:
            self.print_status(f"Error checking CUDA: {e}", "WARNING")
            return False
    
    def check_python_gpu_libraries(self):
        """Check if Python GPU libraries are available"""
        self.print_section("Python GPU Libraries Check")
        
        libraries_to_check = [
            ("torch", "PyTorch"),
            ("torch.cuda", "PyTorch CUDA"),
            ("numpy", "NumPy"),
        ]
        
        all_available = True
        
        for lib_name, display_name in libraries_to_check:
            try:
                if lib_name == "torch.cuda":
                    import torch
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        self.print_status(f"{display_name}: Available and CUDA enabled", "SUCCESS")
                    else:
                        self.print_status(f"{display_name}: Available but CUDA disabled", "WARNING")
                        all_available = False
                else:
                    __import__(lib_name)
                    self.print_status(f"{display_name}: Available", "SUCCESS")
            except ImportError:
                self.print_status(f"{display_name}: Not available", "WARNING")
                all_available = False
        
        return all_available
    
    def suggest_fixes(self):
        """Suggest fixes based on diagnostic results"""
        self.print_section("Recommended Fixes")
        
        fixes = []
        
        # Check if we have the performance monitoring issue
        if not hasattr(self, 'performance_monitoring_working') or not self.performance_monitoring_working:
            fixes.extend([
                "1. Restart NVIDIA Display Driver Service:",
                "   - Open Services (services.msc)",
                "   - Find 'NVIDIA Display Driver Service'",
                "   - Right-click → Restart",
                "",
                "2. Restart NVIDIA GeForce Experience:",
                "   - Close GeForce Experience completely",
                "   - Restart the application",
                "",
                "3. Update NVIDIA Drivers:",
                "   - Download latest drivers from nvidia.com",
                "   - Perform clean installation",
                "",
                "4. Reset NVIDIA Settings:",
                "   - Open NVIDIA Control Panel",
                "   - Go to 'Manage 3D Settings'",
                "   - Click 'Restore' button",
                "",
                "5. Check Windows Event Viewer:",
                "   - Look for NVIDIA-related errors",
                "   - Check for driver crashes",
                "",
                "6. Disable and Re-enable GPU in Device Manager:",
                "   - Open Device Manager",
                "   - Find your GTX 1650",
                "   - Right-click → Disable device",
                "   - Wait 10 seconds, then Enable device"
            ])
        
        # Check for service issues
        if not hasattr(self, 'services_running') or not self.services_running:
            fixes.extend([
                "7. Fix NVIDIA Services:",
                "   - Open Command Prompt as Administrator",
                "   - Run: sc start nvsvc",
                "   - Run: sc start NvTelemetry",
                "",
                "8. Check Windows Update:",
                "   - Ensure Windows is up to date",
                "   - Some updates can fix GPU driver issues"
            ])
        
        # Check for CUDA issues
        if not hasattr(self, 'cuda_working') or not self.cuda_working:
            fixes.extend([
                "9. Install/Update CUDA Toolkit:",
                "   - Download from nvidia.com/developers/cuda",
                "   - Choose version compatible with your driver",
                "",
                "10. Check PATH Environment:",
                "    - Ensure CUDA bin directory is in PATH",
                "    - Restart system after PATH changes"
            ])
        
        if not fixes:
            self.print_status("No specific fixes needed - GPU appears to be working correctly", "SUCCESS")
        else:
            for fix in fixes:
                print(f"   {fix}")
    
    def run_quick_fixes(self):
        """Attempt to run some quick fixes automatically"""
        self.print_section("Attempting Quick Fixes")
        
        try:
            # Try to restart NVIDIA services
            self.print_status("Attempting to restart NVIDIA services...", "FIX")
            
            services_to_restart = ["nvsvc", "NvTelemetry"]
            
            for service in services_to_restart:
                try:
                    # Stop service
                    subprocess.run(['sc', 'stop', service], 
                                 capture_output=True, timeout=5)
                    time.sleep(2)
                    
                    # Start service
                    result = subprocess.run(['sc', 'start', service], 
                                         capture_output=True, timeout=5)
                    
                    if result.returncode == 0:
                        self.print_status(f"Successfully restarted {service}", "SUCCESS")
                        self.fixes_applied.append(f"Restarted {service}")
                    else:
                        self.print_status(f"Failed to restart {service}", "WARNING")
                        
                except Exception as e:
                    self.print_status(f"Could not restart {service}: {e}", "WARNING")
            
            # Wait a moment for services to stabilize
            time.sleep(5)
            
            # Test if performance monitoring is now working
            if self.check_gpu_performance_monitoring():
                self.print_status("Quick fixes may have resolved the issue!", "SUCCESS")
                self.print_status("Try opening NVIDIA GeForce Experience again", "INFO")
            else:
                self.print_status("Quick fixes did not resolve the issue", "WARNING")
                self.print_status("Manual fixes may be required", "INFO")
                
        except Exception as e:
            self.print_status(f"Error during quick fixes: {e}", "ERROR")
    
    def run_full_diagnostic(self):
        """Run the complete diagnostic suite"""
        self.print_header("NVIDIA GPU DIAGNOSTIC TOOL")
        self.print_status("Starting comprehensive GPU diagnostic...", "INFO")
        
        # Gather system information
        self.get_system_info()
        
        # Check NVIDIA drivers
        drivers_ok = self.check_nvidia_drivers()
        
        # Check NVIDIA services
        self.services_running = self.check_nvidia_services()
        
        # Check GPU performance monitoring
        self.performance_monitoring_working = self.check_gpu_performance_monitoring()
        
        # Check CUDA installation
        self.cuda_working = self.check_cuda_installation()
        
        # Check Python GPU libraries
        self.python_libs_ok = self.check_python_gpu_libraries()
        
        # Summary
        self.print_section("Diagnostic Summary")
        
        status_items = [
            ("NVIDIA Drivers", drivers_ok),
            ("NVIDIA Services", self.services_running),
            ("Performance Monitoring", self.performance_monitoring_working),
            ("CUDA Installation", self.cuda_working),
            ("Python GPU Libraries", self.python_libs_ok)
        ]
        
        overall_status = True
        for item_name, status in status_items:
            icon = "✅" if status else "❌"
            status_text = "OK" if status else "ISSUE DETECTED"
            print(f"{icon} {item_name}: {status_text}")
            if not status:
                overall_status = False
        
        if overall_status:
            self.print_status("All systems operational! GPU should be working correctly.", "SUCCESS")
        else:
            self.print_status("Issues detected. Review the diagnostic results above.", "WARNING")
        
        # Suggest fixes
        self.suggest_fixes()
        
        # Ask if user wants to try quick fixes
        print("\n" + "="*60)
        response = input("🔧 Would you like me to attempt automatic quick fixes? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            self.run_quick_fixes()
        else:
            self.print_status("Skipping automatic fixes. Apply manual fixes as suggested above.", "INFO")
        
        # Final recommendations
        self.print_section("Next Steps")
        if not overall_status:
            self.print_status("1. Apply the suggested fixes above", "INFO")
            self.print_status("2. Restart your computer", "INFO")
            self.print_status("3. Run this diagnostic tool again", "INFO")
            self.print_status("4. If issues persist, consider contacting NVIDIA support", "INFO")
        else:
            self.print_status("Your GPU appears to be working correctly!", "SUCCESS")
            self.print_status("Try opening NVIDIA GeForce Experience again", "INFO")

def main():
    """Main function"""
    try:
        diagnostic_tool = GPUDiagnosticTool()
        diagnostic_tool.run_full_diagnostic()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during diagnostic: {e}")
        print("Please check your system and try again")

if __name__ == "__main__":
    main()
