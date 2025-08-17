#!/usr/bin/env python3
"""
Test script to verify Python installation and PyTorch CUDA support
"""

import sys
import os

print("🐍 Python Installation Test")
print("=" * 40)

# Check Python executable
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[0]}")

# Check if we can import torch
try:
    import torch
    print(f"\n✅ PyTorch imported successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available - this is the problem!")
        
except ImportError as e:
    print(f"\n❌ Failed to import PyTorch: {e}")
    print("This means VS Code is using the wrong Python installation!")

# Check environment variables
print(f"\n🔍 Environment Info:")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')[:100]}...")

print("\n" + "=" * 40)
print("Test completed!")
