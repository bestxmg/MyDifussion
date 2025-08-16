#!/usr/bin/env python3
"""
Project Organization Script - Clean up the messy project structure
"""

import os
import shutil
import glob

def create_directory_structure():
    """Create organized directory structure"""
    
    # Define the new structure
    directories = {
        'core': 'Core working generators and models',
        'diagnostics': 'GPU diagnostic and testing tools',
        'experiments': 'Experimental and test generators',
        'utilities': 'Utility scripts and helpers',
        'docs': 'Documentation and reports',
        'models': 'Model files and configurations',
        'archive': 'Old/obsolete files'
    }
    
    print("🏗️  Creating organized directory structure...")
    
    for dir_name, description in directories.items():
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"   ✅ Created: {dir_name}/ ({description})")
        else:
            print(f"   ℹ️  Exists: {dir_name}/")

def organize_files():
    """Organize files into appropriate directories"""
    
    print("\n📁 Organizing files...")
    
    # Core working generators
    core_files = [
        'final_working_gpu_generator.py',
        'working_generator.py',
        'stable_diffusion_source.py'
    ]
    
    # Diagnostic tools
    diagnostic_files = [
        'gpu_diagnostic_tool.py',
        'gpu_diagnostic_generator.py',
        'diagnose_bottleneck.py',
        'debug_black_images.py',
        'system_check_report.html',
        'GPU_DIAGNOSTIC_README.md'
    ]
    
    # Experimental generators
    experimental_files = [
        'vram_fix_test.py',
        'simple_flower_test.py',
        'simple_test_generator.py',
        'working_gpu_generator.py',
        'balanced_gentle_generator.py',
        'gentle_gpu_generator.py',
        'vpn_friendly_gpu_generator.py',
        'gpu_infp_test.py',
        'gpu_training_fix.py',
        'simple_gpu_test.py',
        'test_programmer_generation.py',
        'max_gpu_generator.py',
        'gpu_generator.py',
        'ssl_bypass_generator.py'
    ]
    
    # Utility scripts
    utility_files = [
        'check_image.py',
        'check_generated_image.py',
        'check_resources.py',
        'progress_tracker.py',
        'quick_test.py',
        'quick_generate.py',
        'test_installation.py'
    ]
    
    # Documentation
    doc_files = [
        'README_SOURCE.md',
        'requirements_source.txt'
    ]
    
    # Archive (old/obsolete)
    archive_files = [
        'robust_generator.py',
        'vpn_friendly_generator.py',
        'gpu_image_diagnostic.py',
        'cpu_fallback_generator.py',
        'quick_gpu_fix.py'
    ]
    
    # Move files to appropriate directories
    file_moves = [
        (core_files, 'core'),
        (diagnostic_files, 'diagnostics'),
        (experimental_files, 'experiments'),
        (utility_files, 'utilities'),
        (doc_files, 'docs'),
        (archive_files, 'archive')
    ]
    
    for files, target_dir in file_moves:
        for file in files:
            if os.path.exists(file):
                try:
                    shutil.move(file, os.path.join(target_dir, file))
                    print(f"   📦 Moved: {file} → {target_dir}/")
                except Exception as e:
                    print(f"   ❌ Failed to move {file}: {e}")
            else:
                print(f"   ⚠️  File not found: {file}")

def create_project_readme():
    """Create a comprehensive project README"""
    
    readme_content = """# Stable Diffusion GPU Learning Project

## 🎯 Project Overview
This project is designed for learning big models and Stable Diffusion on GPU. The GPU black image issue has been solved using proper memory management techniques.

## 📁 Project Structure

### Core (`/core`)
- **`final_working_gpu_generator.py`** - Main working GPU generator (PROBLEM SOLVED!)
- **`working_generator.py`** - CPU fallback generator
- **`stable_diffusion_source.py`** - Base Stable Diffusion implementation

### Diagnostics (`/diagnostics`)
- **`gpu_diagnostic_tool.py`** - Comprehensive GPU diagnostics
- **`gpu_diagnostic_generator.py`** - GPU generation diagnostics
- **`diagnose_bottleneck.py`** - Performance bottleneck analysis
- **`system_check_report.html`** - System compatibility report

### Experiments (`/experiments`)
- **`vram_fix_test.py`** - VRAM optimization tests
- **`simple_flower_test.py`** - Minimal workload testing
- **`working_gpu_generator.py`** - Working GPU configuration
- Various experimental generators for testing different approaches

### Utilities (`/utilities`)
- **`check_image.py`** - Image quality verification
- **`check_resources.py`** - System resource monitoring
- **`progress_tracker.py`** - Generation progress tracking

### Documentation (`/docs`)
- **`README_SOURCE.md`** - Original project documentation
- **`requirements_source.txt`** - Python dependencies

## 🔧 GPU Issue Resolution

### Root Cause
The black image issue was caused by using `torch.float16` (half-precision) on a 4GB VRAM GPU, causing memory fragmentation.

### Solution
**Use `torch_dtype=torch.float32` instead of `torch.float16`**

### Working Configuration
```python
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # ← KEY FIX!
    safety_checker=None,
    requires_safety_checker=False
)
```

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r docs/requirements_source.txt`
2. **Run GPU generator**: `python core/final_working_gpu_generator.py`
3. **Test GPU diagnostics**: `python diagnostics/gpu_diagnostic_tool.py`

## 📊 Model Information
- **Base Model**: `runwayml/stable-diffusion-v1-5`
- **Resolution**: 512x512 (proven working)
- **Data Type**: float32 (stable on 4GB VRAM)
- **Memory Optimizations**: attention_slicing, vae_slicing

## 🎓 Learning Path
1. Start with `final_working_gpu_generator.py` to understand working configuration
2. Study `gpu_diagnostic_generator.py` for GPU troubleshooting
3. Experiment with different settings in `/experiments`
4. Use utilities for monitoring and debugging

## 🔍 Debugging
- Use `check_image.py` to verify image quality
- Monitor resources with `check_resources.py`
- Run diagnostics when issues occur
- Check generated images in `/generated_images`

---
*Project organized and ready for big model learning! 🎉*
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("   📝 Created: README.md (comprehensive project guide)")

def main():
    print("🧹 PROJECT ORGANIZATION - Cleaning up the mess!")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Organize files
    organize_files()
    
    # Create project README
    create_project_readme()
    
    print("\n🎉 Project organization complete!")
    print("📁 Your project is now clean and organized!")
    print("📖 Check README.md for the new structure")
    print("\n🚀 Ready for big model learning!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Organization failed: {e}")
    
    input("\nPress Enter to continue...")
