# 🐛 Stable Diffusion Debugging Guide

## Overview
This guide explains how to debug your stable diffusion program line by line, just like using GDB. You have several debugging options available.

## 🔍 Debugging Options

### Option 1: GUI Debug Mode (Recommended)
Run the debug version of the GUI with built-in debugging features:

```bash
python core/stable_diffusion_gui_debug.py
```

**Features:**
- Real-time debug logging in GUI
- Breakpoint controls
- Model inspection tools
- Memory monitoring
- Detailed step-by-step logging

### Option 2: Command Line Debugging
Use the simple debugging script:

```bash
python debug_stable_diffusion.py
```

### Option 3: Manual pdb Integration
Add `pdb.set_trace()` to any line in your code where you want to pause execution.

## 🛑 Using Python Debugger (pdb)

### Basic Commands
When you hit a breakpoint, you'll see the `(Pdb)` prompt. Here are the key commands:

```
(Pdb) h                    # Show help
(Pdb) n                    # Next line (step over)
(Pdb) s                    # Step into function
(Pdb) c                    # Continue execution
(Pdb) l                    # List current code
(Pdb) p variable_name      # Print variable value
(Pdb) pp variable_name     # Pretty print variable
(Pdb) w                    # Show call stack
(Pdb) u                    # Move up call stack
(Pdb) d                    # Move down call stack
(Pdb) q                    # Quit debugger
```

### Advanced Commands
```
(Pdb) b line_number        # Set breakpoint at line
(Pdb) b function_name      # Set breakpoint at function
(Pdb) cl                   # Clear all breakpoints
(Pdb) r                    # Continue until return
(Pdb) j line_number        # Jump to line (dangerous!)
```

## 📊 Debugging the Complete Call Stack

### 1. Model Loading Phase
```python
# BREAKPOINT: Before model loading
pdb.set_trace()

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,  # Key fix for 4GB VRAM
    safety_checker=None,
    requires_safety_checker=False
)

# BREAKPOINT: After model loading
pdb.set_trace()
```

**Debug Commands:**
- `p pipe` - Inspect pipeline object
- `p pipe.config` - Check model configuration
- `p torch.cuda.is_available()` - Verify CUDA availability

### 2. GPU Transfer Phase
```python
# BREAKPOINT: Before GPU transfer
pdb.set_trace()

pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# BREAKPOINT: After GPU transfer
pdb.set_trace()
```

**Debug Commands:**
- `p pipe.device` - Check device location
- `p torch.cuda.memory_allocated(0)` - Check GPU memory usage
- `p pipe.unet.device` - Verify UNet location

### 3. Image Generation Phase
```python
# BREAKPOINT: Before generation
pdb.set_trace()

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    width=width,
    height=height
).images[0]

# BREAKPOINT: After generation
pdb.set_trace()
```

**Debug Commands:**
- `p prompt` - Check input prompt
- `p steps` - Verify inference steps
- `p guidance` - Check guidance scale
- `p image.size` - Verify output dimensions

## 🔧 Debugging Specific Issues

### Memory Issues
```python
# Check GPU memory at any point
if torch.cuda.is_available():
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
```

### Model State Issues
```python
# Inspect model components
print(f"Pipeline device: {pipe.device}")
print(f"UNet device: {pipe.unet.device}")
print(f"VAE device: {pipe.vae.device}")
print(f"Text encoder device: {pipe.text_encoder.device}")
```

### Generation Progress
```python
# Add callback for step-by-step debugging
def debug_callback(step, timestep, latents):
    print(f"Step {step}: timestep {timestep}")
    if step % 5 == 0:  # Break every 5 steps
        pdb.set_trace()

# Use in pipeline call
image = pipe(
    prompt=prompt,
    callback=debug_callback,
    callback_steps=1
).images[0]
```

## 🎯 Step-by-Step Debugging Workflow

### 1. Start with GUI Debug Mode
```bash
python core/stable_diffusion_gui_debug.py
```

### 2. Enable Breakpoints
- Check "Enable Debug Mode"
- Check "Break on Generation" for automatic breakpoints
- Use "🔍 Inspect Model" to check model state
- Use "📊 Memory Status" to monitor GPU memory

### 3. Generate Image
- Enter your prompt
- Click "🎨 Generate Image (DEBUG)"
- Debugger will pause at the pipeline call

### 4. Step Through Execution
```
(Pdb) n                    # Step through line by line
(Pdb) p variable_name      # Inspect variables
(Pdb) c                    # Continue to next breakpoint
```

### 5. Monitor Logs
- Watch the debug log panel in real-time
- Check `debug_gui.log` file for detailed logs
- Use log level selector to filter messages

## 🚨 Common Debug Scenarios

### Black Image Issue
```python
# Check data type
pdb.set_trace()
print(f"Model dtype: {pipe.unet.dtype}")
print(f"Expected: torch.float32")
print(f"Actual: {pipe.unet.dtype}")

# Check device
print(f"Model device: {pipe.unet.device}")
print(f"Expected: cuda:0")
```

### Out of Memory Error
```python
# Check memory before generation
pdb.set_trace()
print(f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# Try reducing resolution
width, height = 256, 256  # Instead of 512x512
```

### Slow Generation
```python
# Profile each step
import time

def timed_callback(step, timestep, latents):
    start = time.time()
    # Your callback logic here
    elapsed = time.time() - start
    print(f"Step {step} took {elapsed:.2f}s")
    
    if step % 5 == 0:
        pdb.set_trace()
```

## 📝 Debug Logging

### Enable Detailed Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

### Log Key Events
```python
logger.debug(f"Loading model: {model_name}")
logger.debug(f"Model loaded, size: {model_size}")
logger.debug(f"Moving to device: {device}")
logger.debug(f"Generation started: {prompt}")
logger.debug(f"Generation completed in {time:.2f}s")
```

## 🎉 Debugging Tips

1. **Start Small**: Test with minimal prompts and low step counts
2. **Check Memory**: Monitor GPU memory usage throughout the process
3. **Verify Data Types**: Ensure you're using `torch.float32` for 4GB VRAM
4. **Step Through**: Use `n` to step line by line through critical sections
5. **Inspect Variables**: Use `p variable_name` to check values at each step
6. **Monitor Logs**: Watch both console and log file output
7. **Break on Errors**: Add `pdb.set_trace()` in exception handlers

## 🔍 Advanced Debugging

### Custom Callbacks
```python
class DebugCallback:
    def __init__(self):
        self.step_count = 0
    
    def __call__(self, step, timestep, latents):
        self.step_count += 1
        print(f"Step {self.step_count}: {step}/{timestep}")
        
        if self.step_count % 10 == 0:
            pdb.set_trace()

# Use in pipeline
callback = DebugCallback()
image = pipe(prompt=prompt, callback=callback, callback_steps=1).images[0]
```

### Memory Profiling
```python
import tracemalloc

tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.1f} MB")
print(f"Peak memory usage: {peak / 1024**2:.1f} MB")
tracemalloc.stop()
```

## 🚀 Quick Start Commands

```bash
# Start GUI debug mode
python core/stable_diffusion_gui_debug.py

# Start command line debug
python debug_stable_diffusion.py

# Check GPU status
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Monitor GPU memory
watch -n 1 nvidia-smi
```

## 📚 Additional Resources

- **Python Debugger Documentation**: https://docs.python.org/3/library/pdb.html
- **PyTorch Debugging**: https://pytorch.org/docs/stable/notes/debug.html
- **Diffusers Debugging**: Check the diffusers library documentation
- **GPU Memory Management**: NVIDIA documentation for your specific GPU

---

**Happy Debugging! 🐛✨**

Remember: The key to successful debugging is to understand the flow, set strategic breakpoints, and inspect variables at each step. Start with the GUI debug mode for the most user-friendly experience!
