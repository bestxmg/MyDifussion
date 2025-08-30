#!/usr/bin/env python3
"""
Simplified Danganronpa LoRA Training Script
Easy-to-use version for training Danganronpa style LoRA
"""

import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(image_folder="website_images", size=512):
    """Prepare dataset from image folder"""
    
    image_folder = Path(image_folder)
    image_files = []
    
    # Get all image files
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_files.extend(list(image_folder.glob(f"*{ext}")))
        image_files.extend(list(image_folder.glob(f"*{ext.upper()}")))
    
    # Filter out gif files
    image_files = [f for f in image_files if f.suffix.lower() != '.gif']
    
    logger.info(f"Found {len(image_files)} images")
    return image_files

def setup_lora(unet, r=16, alpha=32):
    """Setup LoRA for UNet"""
    
    # Configure LoRA
    lora_config = LoRAAttnProcessor(r=r, lora_alpha=alpha)
    
    # Apply to attention layers
    for name, module in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            module.processor = lora_config
    
    return unet

def train_simple_lora(
    image_folder="website_images",
    output_dir="danganronpa_lora",
    num_epochs=50,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32
):
    """Simple LoRA training for Danganronpa style"""
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load base model
    model_name = "runwayml/stable-diffusion-v1-5"
    
    logger.info("Loading tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder = text_encoder.to(device)
    
    logger.info("Loading UNet...")
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    unet = setup_lora(unet, r=lora_r, alpha=lora_alpha)
    unet = unet.to(device)
    
    logger.info("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    vae = vae.to(device)
    
    # Prepare dataset
    image_files = prepare_dataset(image_folder)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate
    )
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for i, image_file in enumerate(image_files):
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
            
            if (i + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Image {i+1}/{len(image_files)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(image_files)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            lora_layers = AttnProcsLayers(unet.attn_processors)
            checkpoint_dir = f"{output_dir}/checkpoint-{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            lora_layers.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.save_pretrained(output_dir)
    
    logger.info(f"Training completed! LoRA saved to {output_dir}")
    return output_dir

def test_lora(lora_path, prompt="danganronpa character"):
    """Test the trained LoRA model"""
    
    logger.info("Loading base model for testing...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA
    logger.info("Loading LoRA weights...")
    pipe.unet.load_attn_procs(lora_path)
    
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
    
    logger.info(f"Generated {len(images)} test images in {output_dir}")
    return images

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Danganronpa LoRA Training")
    parser.add_argument("--train", action="store_true", help="Train LoRA model")
    parser.add_argument("--test", action="store_true", help="Test LoRA model")
    parser.add_argument("--image_folder", default="website_images", help="Training images folder")
    parser.add_argument("--output_dir", default="danganronpa_lora", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.train:
        train_simple_lora(
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
    
    if args.test:
        if os.path.exists(args.output_dir):
            test_lora(args.output_dir)
        else:
            logger.error(f"LoRA model not found at {args.output_dir}")
    
    if not args.train and not args.test:
        print("Please specify --train or --test")
        print("Example:")
        print("  python simple_danganronpa_lora.py --train")
        print("  python simple_danganronpa_lora.py --test")
