#!/usr/bin/env python3
"""
Danganronpa Style LoRA Fine-tuning Script
Trains a LoRA model to generate Danganronpa-style images
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DanganronpaDataset(Dataset):
    """Dataset for Danganronpa style images"""
    
    def __init__(self, image_folder, tokenizer, size=512):
        self.image_folder = Path(image_folder)
        self.tokenizer = tokenizer
        self.size = size
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # Get all image files
        self.image_files = []
        for ext in self.image_extensions:
            self.image_files.extend(list(self.image_folder.glob(f"*{ext}")))
            self.image_files.extend(list(self.image_folder.glob(f"*{ext.upper()}")))
        
        # Filter out non-images and gif files
        self.image_files = [f for f in self.image_files if f.suffix.lower() != '.gif']
        
        logger.info(f"Found {len(self.image_files)} images in {image_folder}")
        
        # Danganronpa style prompts
        self.style_prompt = "danganronpa style, anime art style, character portrait, colorful, vibrant"
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        
        # Load and preprocess image
        image = Image.open(image_file).convert("RGB")
        
        # Resize image
        image = image.resize((self.size, self.size))
        
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC to CHW
        
        # Create prompt with Danganronpa style
        filename = image_file.stem
        prompt = f"{filename}, {self.style_prompt}"
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze()
        }

def setup_lora_attn_processors(unet, r=16, alpha=32):
    """Setup LoRA attention processors for the UNet"""
    
    # LoRA config
    lora_config = LoRAAttnProcessor(r=r, alpha=alpha)
    
    # Apply to all attention processors
    for name, module in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            if hasattr(module, "to_k") and hasattr(module, "to_q") and hasattr(module, "to_v"):
                # Cross attention
                module.processor = lora_config
            elif hasattr(module, "to_k") and hasattr(module, "to_q") and hasattr(module, "to_v"):
                # Self attention
                module.processor = lora_config
    
    return unet

def train_lora(
    model_name="runwayml/stable-diffusion-v1-5",
    image_folder="website_images",
    output_dir="danganronpa_lora",
    num_epochs=100,
    batch_size=1,
    learning_rate=1e-4,
    lora_r=16,
    lora_alpha=32,
    resolution=512,
    gradient_accumulation_steps=4,
    save_steps=500,
    mixed_precision="fp16"
):
    """Train LoRA for Danganronpa style"""
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    
    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # Load UNet and setup LoRA
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    unet = setup_lora_attn_processors(unet, r=lora_r, alpha=lora_alpha)
    
    # Load VAE
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    
    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    
    # Create dataset and dataloader
    dataset = DanganronpaDataset(image_folder, tokenizer, resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate
    )
    
    # Prepare with accelerator
    unet, optimizer, dataloader, text_encoder, vae = accelerator.prepare(
        unet, optimizer, dataloader, text_encoder, vae
    )
    
    # Training loop
    progress_bar = tqdm(range(num_epochs), desc="Training")
    
    for epoch in progress_bar:
        unet.train()
        
        for step, batch in enumerate(dataloader):
            # Get batch data
            pixel_values = batch["pixel_values"].float()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
            
            # Encode text
            with torch.no_grad():
                text_embeddings = text_encoder(
                    input_ids,
                    attention_mask=attention_mask
                )[0]
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            noisy_latents = latents + noise * 0.1
            
            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            
            # Backward pass
            accelerator.backward(loss)
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log progress
            progress_bar.set_postfix({
                "Epoch": epoch,
                "Step": step,
                "Loss": loss.item()
            })
            
            # Save checkpoint
            if (step + 1) % save_steps == 0:
                # Save LoRA weights
                lora_layers = AttnProcsLayers(unet.attn_processors)
                lora_layers.save_pretrained(f"{output_dir}/checkpoint-{epoch}-{step}")
                
                logger.info(f"Saved checkpoint at epoch {epoch}, step {step}")
    
    # Save final model
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "resolution": resolution,
        "style_prompt": "danganronpa style, anime art style, character portrait, colorful, vibrant"
    }
    
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training completed! LoRA saved to {output_dir}")
    return output_dir

def test_lora_model(lora_path, prompt="danganronpa character", num_images=4):
    """Test the trained LoRA model"""
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    pipe.unet.load_attn_procs(lora_path)
    
    # Generate images
    images = pipe(
        prompt=f"{prompt}, danganronpa style",
        num_inference_steps=30,
        guidance_scale=7.5,
        num_images_per_prompt=num_images
    ).images
    
    # Save images
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    for i, image in enumerate(images):
        image.save(output_dir / f"test_lora_{i}.png")
    
    logger.info(f"Generated {len(images)} test images in {output_dir}")
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA for Danganronpa style")
    parser.add_argument("--image_folder", default="website_images", help="Folder containing training images")
    parser.add_argument("--output_dir", default="danganronpa_lora", help="Output directory for LoRA")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--test_only", action="store_true", help="Only test existing model")
    
    args = parser.parse_args()
    
    if args.test_only:
        # Test existing model
        if os.path.exists(args.output_dir):
            test_lora_model(args.output_dir)
        else:
            logger.error(f"LoRA model not found at {args.output_dir}")
    else:
        # Train new model
        train_lora(
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            resolution=args.resolution
        )
        
        # Test the trained model
        test_lora_model(args.output_dir)
