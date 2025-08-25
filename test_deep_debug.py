#!/usr/bin/env python3
"""
Test script to verify deep debugging setup
"""

import sys
import os

print("🐍 Testing Deep Debugging Setup")
print("=" * 50)

# Check Python paths
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Check if we can import diffusers
try:
    import diffusers
    print(f"\n✅ Diffusers imported successfully!")
    print(f"Version: {diffusers.__version__}")
    print(f"Location: {diffusers.__file__}")
    
    # Check if it's pointing to source code
    if "diffusers/src" in diffusers.__file__:
        print("🎯 SUCCESS: Diffusers is pointing to SOURCE CODE!")
    else:
        print("⚠️  WARNING: Diffusers is pointing to installed package")
        
except ImportError as e:
    print(f"\n❌ Failed to import diffusers: {e}")

# Check if we can import StableDiffusionPipeline
try:
    from diffusers import StableDiffusionPipeline
    print(f"\n✅ StableDiffusionPipeline imported successfully!")
    print(f"Location: {StableDiffusionPipeline.__module__}")
    
    # Check the actual source file
    pipeline_file = os.path.join(os.path.dirname(diffusers.__file__), "pipelines", "stable_diffusion", "pipeline_stable_diffusion.py")
    if os.path.exists(pipeline_file):
        print(f"🎯 SUCCESS: Pipeline source file exists: {pipeline_file}")
    else:
        print(f"⚠️  WARNING: Pipeline source file not found")
        
except ImportError as e:
    print(f"\n❌ Failed to import StableDiffusionPipeline: {e}")

# Check Python path
print(f"\n🔍 Python Path:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"  {i}: {path}")

print("\n" + "=" * 50)
print("Test completed!")
