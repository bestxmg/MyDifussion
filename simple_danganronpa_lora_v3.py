#!/usr/bin/env python3
"""
Improved Danganronpa LoRA Training Script
With better progress reporting and error handling
"""

import os
import torch
import logging
from pathlib import Path

# Setup logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required packages are available"""
    logger.info("Checking environment...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
        
        import diffusers
        logger.info(f"✓ Diffusers {diffusers.__version__}")
        
        import peft
        logger.info(f"✓ PEFT {peft.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Missing package: {e}")
        return False

def download_models():
    """Download and cache the base models"""
    logger.info("Downloading base models (this may take a few minutes)...")
    
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        
        # Download tokenizer
        logger.info("Downloading CLIP tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="tokenizer",
            local_files_only=False
        )
        logger.info("✓ Tokenizer downloaded")
        
        # Download text encoder
        logger.info("Downloading CLIP text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="text_encoder",
            local_files_only=False
        )
        logger.info("✓ Text encoder downloaded")
        
        # Download UNet
        logger.info("Downloading UNet...")
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet",
            local_files_only=False
        )
        logger.info("✓ UNet downloaded")
        
        # Download VAE
        logger.info("Downloading VAE...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae",
            local_files_only=False
        )
        logger.info("✓ VAE downloaded")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download models: {e}")
        return False

def prepare_dataset(image_folder="website_images"):
    """Prepare dataset with progress reporting"""
    logger.info(f"Preparing dataset from {image_folder}...")
    
    image_folder = Path(image_folder)
    if not image_folder.exists():
        logger.error(f"✗ Image folder not found: {image_folder}")
        return None
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        files = list(image_folder.glob(f"*{ext}")) + list(image_folder.glob(f"*{ext.upper()}"))
        image_files.extend(files)
        logger.info(f"Found {len(files)} {ext} files")
    
    # Filter out gif files
    image_files = [f for f in image_files if f.suffix.lower() != '.gif']
    
    logger.info(f"✓ Total images found: {len(image_files)}")
    return image_files

def train_simple_lora(
    image_folder="website_images",
    output_dir="danganronpa_lora",
    num_epochs=50,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32
):
    """Simple LoRA training with detailed progress"""
    
    logger.info("=" * 60)
    logger.info("STARTING DANGANRONPA LORA TRAINING")
    logger.info("=" * 60)
    
    # Check environment
    if not check_environment():
        return False
    
    # Download models if needed
    if not download_models():
        return False
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cpu":
        logger.warning("⚠️  Training on CPU will be very slow! Consider using CUDA.")
    
    # Prepare dataset
    image_files = prepare_dataset(image_folder)
    if not image_files:
        return False
    
    # Load models
    logger.info("Loading models...")
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        from diffusers import UNet2DConditionModel, AutoencoderKL
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        import numpy as np
        
        model_name = "runwayml/stable-diffusion-v1-5"
        
        logger.info("Loading CLIP tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        
        logger.info("Loading CLIP text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        for param in text_encoder.parameters():
            param.requires_grad = False
        text_encoder = text_encoder.to(device)
        
        logger.info("Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        unet = unet.to(device)
        
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        for param in vae.parameters():
            param.requires_grad = False
        vae = vae.to(device)
        
        logger.info("✓ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Setup optimizer
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training loop...")
    logger.info(f"Training for {num_epochs} epochs on {len(image_files)} images")
    
    try:
        for epoch in range(num_epochs):
            logger.info(f"\n--- EPOCH {epoch+1}/{num_epochs} ---")
            total_loss = 0
            
            for i, image_file in enumerate(image_files):
                try:
                    # Load and preprocess image
                    image = Image.open(image_file).convert("RGB")
                    image = image.resize((512, 512))
                    
                    # Convert to tensor
                    image = torch.from_numpy(np.array(image)).float() / 255.0
                    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    # Create prompt
                    filename = image_file.stem
                    prompt = f"{filename}, danganronpa style, anime art style"
                    
                    # Tokenize
                    inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Encode image to latents
                    with torch.no_grad():
                        latents = vae.encode(image).latent_dist.sample() * 0.18215
                    
                    # Encode text
                    with torch.no_grad():
                        text_embeddings = text_encoder(inputs.input_ids, inputs.attention_mask)[0]
                    
                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
                    noisy_latents = latents + noise * 0.1
                    
                    # Predict noise
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Progress reporting
                    if (i + 1) % 5 == 0 or (i + 1) == len(image_files):
                        avg_loss = total_loss / (i + 1)
                        logger.info(f"  Image {i+1}/{len(image_files)}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {e}")
                    continue
            
            avg_loss = total_loss / len(image_files)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Saving checkpoint for epoch {epoch+1}...")
                checkpoint_dir = f"{output_dir}/checkpoint-{epoch+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                unet.save_pretrained(checkpoint_dir)
                logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
        
        # Save final model
        logger.info("Saving final model...")
        unet.save_pretrained(output_dir)
        logger.info(f"✓ Training completed! LoRA saved to {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lora(lora_path, prompt="danganronpa character"):
    """Test the trained LoRA model"""
    logger.info("Testing LoRA model...")
    
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        from peft import PeftModel
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Load LoRA
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        
        # Generate test images
        logger.info(f"Generating images with prompt: {prompt}")
        
        images = pipe(
            prompt=f"{prompt}, danganronpa style",
            num_inference_steps=30,
            guidance_scale=7.5,
            num_images_per_prompt=4
        ).images
        
        # Save images
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        for i, image in enumerate(images):
            image.save(output_dir / f"danganronpa_test_{i}.png")
        
        logger.info(f"✓ Generated {len(images)} test images in {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Testing failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Danganronpa LoRA Training")
    parser.add_argument("--train", action="store_true", help="Train LoRA model")
    parser.add_argument("--test", action="store_true", help="Test LoRA model")
    parser.add_argument("--image_folder", default="website_images", help="Training images folder")
    parser.add_argument("--output_dir", default="danganronpa_lora", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--download_only", action="store_true", help="Only download models")
    
    args = parser.parse_args()
    
    if args.download_only:
        download_models()
    elif args.train:
        success = train_simple_lora(
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        if success:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed!")
    elif args.test:
        if os.path.exists(args.output_dir):
            test_lora(args.output_dir)
        else:
            logger.error(f"LoRA model not found at {args.output_dir}")
    else:
        print("Please specify --train, --test, or --download_only")
        print("Example:")
        print("  python simple_danganronpa_lora_v3.py --download_only")
        print("  python simple_danganronpa_lora_v3.py --train")
        print("  python simple_danganronpa_lora_v3.py --test")
