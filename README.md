# Stable Diffusion GPU Learning Project

## 🎯 Project Overview
This project is designed for learning big models and Stable Diffusion on GPU. The GPU black image issue has been solved using proper memory management techniques. **Now includes a user-friendly GUI for easy image generation!** 🎨✨

## 📁 Project Structure

### Core (`/core`)
- **`stable_diffusion_gui.py`** - 🆕 **User-friendly GUI for image generation** (RECOMMENDED!)
- **`final_working_gpu_generator.py`** - Command-line GPU generator (PROBLEM SOLVED!)
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
- **`gpu_memory_monitor.py`** - 🆕 **GPU memory usage monitoring**
- **`parameter_guide.py`** - 🆕 **Comprehensive guide to inference steps & guidance scale**
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

### 🎨 **Quick Start with GUI (Recommended):**
1. **Install dependencies**: `pip install -r docs/requirements_source.txt`
2. **Launch GUI**: `python core/stable_diffusion_gui.py`
3. **Start generating images immediately!** ✨

### 💻 **Command Line Usage:**
1. **Install dependencies**: `pip install -r docs/requirements_source.txt`
2. **Run GPU generator**: `python core/final_working_gpu_generator.py`
3. **Test GPU diagnostics**: `python diagnostics/gpu_diagnostic_tool.py`

## 📊 Model Information
- **Base Model**: `runwayml/stable-diffusion-v1-5`
- **Resolution**: 512x512 (proven working)
- **Data Type**: float32 (stable on 4GB VRAM)
- **Memory Optimizations**: attention_slicing, vae_slicing

## 🎨 **GUI Features**
- **User-friendly interface** for easy image generation
- **Real-time progress tracking** with animated progress bar
- **Parameter help system** explaining inference steps & guidance scale
- **Stop generation** capability for user control
- **Image preview** and save functionality
- **Optimized settings** for GTX 1650 (4GB VRAM)

### 📚 **Built-in Learning Resources**
- **❓ Help Button**: Click for comprehensive parameter explanations
- **Three Help Tabs**: 
  - 🔄 **Inference Steps**: Understanding quality vs speed trade-offs
  - 🎭 **Guidance Scale**: Balancing creativity vs accuracy
  - 📋 **Quick Reference**: Troubleshooting and recommended settings
- **Personalized Recommendations**: Specific advice for your GTX 1650 setup

## 🎓 Learning Path
1. **Start with GUI**: `python core/stable_diffusion_gui.py` for easy image generation
2. **Learn parameters**: Click the ❓ Help button to understand inference steps & guidance scale
3. **Study working code**: Examine `final_working_gpu_generator.py` for working configuration
4. **GPU troubleshooting**: Use `gpu_diagnostic_generator.py` for diagnostics
5. **Experiment**: Try different settings in `/experiments`
6. **Monitor resources**: Use utilities for monitoring and debugging

## 🔍 Debugging
- **GUI progress tracking**: Real-time generation status and progress
- **Parameter help**: Built-in explanations for all settings
- **Image quality**: Use `check_image.py` to verify generated images
- **Resource monitoring**: Monitor with `check_resources.py` and `gpu_memory_monitor.py`
- **Diagnostics**: Run diagnostics when issues occur
- **Generated images**: Check results in `/generated_images`

## 🆕 **What's New**
- **User-friendly GUI** for easy image generation
- **Built-in parameter help** system explaining all settings
- **Real-time progress tracking** with stop capability
- **GPU memory monitoring** tools
- **Comprehensive parameter guides** for learning

---

*Project organized and ready for big model learning with a beautiful GUI! 🎨✨🎉*
