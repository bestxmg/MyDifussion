# 🎯 Stable Diffusion LoRA Fine-tuning Guide

## 📋 Overview

This guide shows you how to fine-tune Stable Diffusion on your local computer using LoRA (Low-Rank Adaptation). LoRA is a technique that allows you to train only a small number of additional parameters instead of the entire model, making it much more practical for local training.

## 🚀 Quick Start

### 1. Test Your Setup
```bash
python test_lora_setup.py
```

### 2. Prepare Training Data
Create a directory structure like this:
```
training_data/
├── image1.jpg
├── image1.txt
├── image2.jpg
├── image2.txt
└── ...
```

**Format:**
- **Images:** JPG, PNG, or JPEG files
- **Text files:** Same filename as image but with `.txt` extension
- **Content:** Text description of the image

**Example:**
- `dragon.jpg` → `dragon.txt` containing "a majestic dragon flying over mountains"
- `castle.jpg` → `castle.txt` containing "an ancient castle at sunset"

### 3. Start Training
```bash
python train_lora_practical.py
```

## 🔧 What You Need

### Hardware Requirements:
- **GPU:** NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU:** Can work but will be very slow
- **RAM:** 16GB+ system RAM
- **Storage:** 10GB+ free space

### Software Requirements:
- **Python 3.8+**
- **PyTorch with CUDA support**
- **Required packages:** `diffusers`, `transformers`, `peft`, `accelerate`

## 📚 Understanding LoRA

### What is LoRA?
LoRA (Low-Rank Adaptation) is a technique that:
- **Freezes** the original model weights
- **Adds** small trainable matrices to specific layers
- **Trains** only these new matrices
- **Results** in much faster training and smaller file sizes

### Why Use LoRA?
- ✅ **Faster training** (10-100x faster)
- ✅ **Less memory usage** (can fit on consumer GPUs)
- ✅ **Smaller output files** (few MB vs several GB)
- ✅ **Easier to manage** and share
- ✅ **Maintains quality** of the base model

## 🎨 Training Process

### 1. Data Preparation
Your training data should include:
- **High-quality images** (512x512 or larger)
- **Accurate text descriptions** that match the images
- **Diverse examples** of what you want to learn
- **Consistent style** if you're learning a specific art style

### 2. Training Parameters
```python
lora_config = LoraConfig(
    r=16,                    # Rank (higher = more parameters, more flexible)
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to fine-tune
    lora_dropout=0.1,        # Dropout for regularization
    bias="none",             # Don't train bias terms
    task_type="CAUSAL_LM"    # Task type
)
```

### 3. Training Loop
The script will:
- Load your training images and text
- Apply noise to images
- Predict the noise using the model
- Compute loss and update LoRA weights
- Save checkpoints every few epochs

## 📊 Monitoring Training

### Loss Values:
- **Lower is better** - indicates the model is learning
- **Should decrease** over time
- **Watch for overfitting** (loss stops decreasing)

### Checkpoints:
- Saved every 5 epochs
- Allows you to resume training if interrupted
- Test intermediate results

## 🧪 Testing Your Model

### After Training:
1. **Load your fine-tuned model**
2. **Generate test images** with your prompts
3. **Compare results** with the base model
4. **Iterate and improve**

### Example Test:
```python
# Load your fine-tuned LoRA
pipe.load_lora_weights("./final_lora_model")

# Generate image
image = pipe(
    prompt="your custom prompt here",
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]
```

## 🔍 Troubleshooting

### Common Issues:

#### 1. **Out of Memory (OOM)**
- **Reduce batch size** (try batch_size=1)
- **Use gradient checkpointing**
- **Train on smaller images** (256x256 instead of 512x512)

#### 2. **Training is Too Slow**
- **Use GPU** instead of CPU
- **Reduce number of epochs**
- **Use smaller LoRA rank** (r=8 instead of r=16)

#### 3. **Poor Results**
- **Check training data quality**
- **Increase number of training examples**
- **Adjust learning rate** (try 5e-5 or 2e-4)
- **Train for more epochs**

#### 4. **Model Not Learning**
- **Verify data format** (images + corresponding text files)
- **Check loss values** (should decrease over time)
- **Ensure LoRA is properly applied** (check trainable parameters)

## 📈 Advanced Techniques

### 1. **Hyperparameter Tuning**
- **Learning rate:** Start with 1e-4, try 5e-5 to 2e-4
- **LoRA rank:** Start with 16, try 8 or 32
- **Batch size:** Start with 2, adjust based on memory

### 2. **Data Augmentation**
- **Rotate images** slightly
- **Adjust brightness/contrast**
- **Add slight noise**
- **Use different text variations**

### 3. **Regularization**
- **LoRA dropout:** 0.1 is usually good
- **Weight decay:** 0.01 in optimizer
- **Early stopping:** Stop when loss plateaus

## 🎯 Use Cases

### What You Can Fine-tune:
- **Art styles** (anime, realistic, watercolor)
- **Specific objects** (dragons, castles, characters)
- **Composition styles** (portrait, landscape, abstract)
- **Color schemes** (dark, bright, monochrome)

### What You Cannot Fine-tune:
- **Fundamental physics** (gravity, lighting)
- **Basic concepts** (what a "cat" looks like)
- **Composition rules** (rule of thirds, etc.)

## 📚 Resources

### Documentation:
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Community:
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Reddit r/StableDiffusion](https://reddit.com/r/StableDiffusion)
- [Discord Communities](https://discord.gg/stablediffusion)

## 🎉 Next Steps

1. **Start with the test script** to verify your setup
2. **Prepare a small dataset** (10-20 images) for testing
3. **Run a short training** (5-10 epochs) to see results
4. **Iterate and improve** based on results
5. **Scale up** with more data and longer training

## 💡 Tips for Success

- **Start small** - test with a few images first
- **Quality over quantity** - better images give better results
- **Be patient** - training takes time, even with LoRA
- **Experiment** - try different parameters and approaches
- **Backup** - save checkpoints regularly
- **Test often** - generate images during training to see progress

---

**Happy fine-tuning! 🚀**

If you encounter issues, check the troubleshooting section or ask for help in the community forums.
