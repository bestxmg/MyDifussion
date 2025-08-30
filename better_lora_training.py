#!/usr/bin/env python3
"""
Better LoRA Training Script with Proper Parameters
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse
from transformers import CLIPTokenizer
from torch.optim import AdamW

class DanganronpaDataset(Dataset):
    def __init__(self, image_dir, tokenizer, image_size=512):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        image_tensor = torch.tensor([list(image.getdata())]).view(image.size[1], image.size[0], 3)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor * 2 - 1  # Normalize to [-1, 1]
        
        # Create prompt (you can customize this)
        prompt = "danganronpa character, anime style, colorful, vibrant"
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        
        return {
            "pixel_values": image_tensor,
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze()
        }

def setup_lora(pipeline):
    """Setup LoRA with better parameters"""
    lora_config = LoraConfig(
        r=16,  # Lower rank for stability
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        scale=1.0,
    )
    
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    return pipeline

def train_lora(pipeline, dataset, num_epochs=50, learning_rate=1e-4, batch_size=1):
    """Train LoRA with better parameters"""
    
    # Setup optimizer
    optimizer = AdamW(pipeline.unet.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Setup dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    pipeline.unet.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move to GPU
            pixel_values = batch["pixel_values"].to("cuda")
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            
            # Forward pass
            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values * 0.5 + 0.5).latent_dist.sample() * 0.18215
                text_embeddings = pipeline.text_encoder(input_ids, attention_mask=attention_mask)[0]
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (latents.shape[0],)).long().to("cuda")
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.unet.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            pipeline.unet.save_pretrained(f"danganronpa_lora_epoch_{epoch+1}")
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    pipeline.unet.save_pretrained("danganronpa_lora_improved")
    print("Training completed! Final model saved as 'danganronpa_lora_improved'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train LoRA")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_dir", type=str, default="website_images", help="Directory with training images")
    
    args = parser.parse_args()
    
    if not args.train:
        print("Use --train to start training")
        return
    
    print("Loading base model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    pipeline = pipeline.to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_attention_slicing()
    
    print("Setting up LoRA...")
    pipeline = setup_lora(pipeline)
    
    print("Loading dataset...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = DanganronpaDataset(args.image_dir, tokenizer)
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Training for {args.num_epochs} epochs...")
    
    train_lora(pipeline, dataset, args.num_epochs, args.learning_rate)

if __name__ == "__main__":
    main()
