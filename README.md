# 🎨 Stable Diffusion GPU Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.21+-green.svg)](https://github.com/huggingface/diffusers)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive learning project for Stable Diffusion on GPU with a focus on solving common GPU issues and providing an intuitive GUI for image generation. Perfect for beginners learning big models and AI image generation.

## ✨ Features

- 🎨 **User-friendly GUI** for easy image generation
- 🔧 **GPU issue resolution** - solved black image problems
- 📚 **Built-in learning resources** with parameter explanations
- 🚀 **Optimized for GTX 1650** (4GB VRAM) and similar cards
- 📊 **Real-time progress tracking** and monitoring
- 🎭 **LoRA support** for custom model fine-tuning
- 🛠️ **Comprehensive diagnostics** and debugging tools

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (4GB+ VRAM recommended)
- Windows 10/11

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MyDifussion.git
   cd MyDifussion
   ```

2. **Install dependencies**
   ```bash
   pip install -r docs/requirements_source.txt
   ```

3. **Launch the GUI**
   ```bash
   python core/stable_diffusion_gui.py
   ```

4. **Start generating images!** 🎉

## 📁 Project Structure

```
MyDifussion/
├── core/                          # Main application files
│   ├── stable_diffusion_gui.py   # 🆕 User-friendly GUI (RECOMMENDED!)
│   ├── final_working_gpu_generator.py  # Command-line GPU generator
│   ├── working_generator.py      # CPU fallback generator
│   └── stable_diffusion_source.py      # Base implementation
├── diagnostics/                   # GPU and system diagnostics
│   ├── gpu_diagnostic_tool.py   # Comprehensive GPU diagnostics
│   ├── gpu_diagnostic_generator.py     # GPU generation diagnostics
│   └── system_check_report.html  # System compatibility report
├── experiments/                   # Experimental configurations
│   ├── vram_fix_test.py         # VRAM optimization tests
│   ├── working_gpu_generator.py # Working GPU configurations
│   └── ...                      # Various experimental approaches
├── utilities/                     # Helper tools and monitoring
│   ├── gpu_memory_monitor.py    # GPU memory usage monitoring
│   ├── parameter_guide.py       # Parameter explanations
│   ├── check_image.py           # Image quality verification
│   └── progress_tracker.py      # Generation progress tracking
├── danganronpa_lora/            # Custom LoRA model example
├── docs/                         # Documentation and requirements
└── generated_images/             # Output directory for generated images
```

## 🎯 Key Features Explained

### 🎨 **Intuitive GUI Interface**
- **Simple prompt input** with negative prompt support
- **Real-time progress tracking** with animated progress bars
- **Parameter help system** explaining all settings
- **Image preview and save** functionality
- **LoRA model loading** for custom styles

### 🔧 **GPU Issue Resolution**
The project solves the common "black image" issue caused by memory fragmentation on 4GB VRAM GPUs.

**Root Cause**: Using `torch.float16` (half-precision) on limited VRAM
**Solution**: Use `torch_dtype=torch.float32` for stability

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # ← KEY FIX!
    safety_checker=None,
    requires_safety_checker=False
)
```

### 📚 **Built-in Learning Resources**
- **❓ Help Button**: Comprehensive parameter explanations
- **Three Help Tabs**: 
  - 🔄 **Inference Steps**: Quality vs speed trade-offs
  - 🎭 **Guidance Scale**: Creativity vs accuracy balance
  - 📋 **Quick Reference**: Troubleshooting and recommendations

## 🎓 Learning Path

1. **Start with GUI**: `python core/stable_diffusion_gui.py`
2. **Learn parameters**: Click the ❓ Help button
3. **Study working code**: Examine `final_working_gpu_generator.py`
4. **GPU troubleshooting**: Use diagnostic tools
5. **Experiment**: Try different settings in `/experiments`
6. **Monitor resources**: Use utility tools for monitoring

## 🔍 Troubleshooting

### Common Issues

**Black Images**: 
- Ensure you're using `torch.float32` data type
- Check GPU memory with `gpu_memory_monitor.py`
- Run `gpu_diagnostic_tool.py` for comprehensive analysis

**Out of Memory**:
- Reduce image resolution (start with 512x512)
- Enable attention_slicing and vae_slicing
- Close other GPU applications

**Slow Generation**:
- Use appropriate inference steps (20-50 for quality)
- Balance guidance scale (7-15 for creativity)
- Monitor GPU utilization

### Diagnostic Tools

- **`diagnostics/gpu_diagnostic_tool.py`** - Comprehensive GPU analysis
- **`utilities/gpu_memory_monitor.py`** - Real-time memory monitoring
- **`utilities/check_resources.py`** - System resource verification

## 🎭 LoRA Training

The project includes LoRA fine-tuning capabilities:

```bash
# Train custom LoRA model
python danganronpa_lora_training.py

# Use trained LoRA in GUI
# Load via the LoRA Settings section
```

See `LORA_TRAINING_GUIDE.md` for detailed instructions.

## 📊 Performance

**Optimized for GTX 1650 (4GB VRAM)**:
- **Resolution**: 512x512 (proven stable)
- **Data Type**: float32 (memory efficient)
- **Memory Optimizations**: attention_slicing, vae_slicing
- **Generation Time**: ~30-60 seconds per image

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the Stable Diffusion implementation
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The AI community for continuous improvements and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/MyDifussion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/MyDifussion/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/MyDifussion/wiki)

---

**⭐ Star this repository if you find it helpful!**

*Ready for big model learning with a beautiful, user-friendly interface! 🎨✨*
