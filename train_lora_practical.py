#!/usr/bin/env python3
"""
Practical LoRA Training Script for Stable Diffusion
This script implements actual training with proper loss computation
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json

class StableDiffusionDataset(Dataset):
    """Dataset for Stable Diffusion fine-tuning"""
    
    def __init__(self, data_dir, tokenizer, image_size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Load training data
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load training samples from data directory"""
        samples = []
        
        # Look for image files and corresponding text files
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Remove extension to find corresponding text file
                base_name = os.path.splitext(filename)[0]
                text_file = os.path.join(self.data_dir, f"{base_name}.txt")
                
                if os.path.exists(text_file):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    samples.append({
                        'image_path': os.path.join(self.data_dir, filename),
                        'prompt': prompt
                    })
        
        print(f"Found {len(samples)} training samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Tokenize text
        tokens = self.tokenizer(
            sample['prompt'],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': tokens.input_ids.squeeze(),
            'attention_mask': tokens.attention_mask.squeeze(),
            'prompt': sample['prompt']
        }

def setup_training_components():
    """Setup all components needed for training"""
    print("🔄 Setting up training components...")
    
    # Load base model
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load components
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = torch.load(f"{model_id}/unet/diffusion_pytorch_model.bin", map_location='cpu')
    
    # Load scheduler
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    print("✅ Components loaded successfully")
    return tokenizer, text_encoder, unet, scheduler

def setup_lora_for_training(text_encoder, unet):
    """Apply LoRA to both text encoder and UNet"""
    print("🎯 Setting up LoRA for training...")
    
    # LoRA config for text encoder
    text_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # LoRA config for UNet
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    text_encoder = get_peft_model(text_encoder, text_lora_config)
    unet = get_peft_model(unet, unet_lora_config)
    
    # Print trainable parameters
    print("Text Encoder trainable parameters:")
    text_encoder.print_trainable_parameters()
    
    print("UNet trainable parameters:")
    unet.print_trainable_parameters()
    
    return text_encoder, unet

def compute_training_loss(text_encoder, unet, scheduler, batch, device):
    """Compute training loss for a batch"""
    
    # Move batch to device
    images = batch['image'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
    
    # Add noise to images
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],))
    noisy_images = scheduler.add_noise(images, noise, timesteps)
    
    # Predict noise
    noise_pred = unet(
        noisy_images,
        timesteps,
        encoder_hidden_states=text_embeddings
    ).sample
    
    # Compute loss
    loss = F.mse_loss(noise_pred, noise)
    
    return loss

def train_model(text_encoder, unet, scheduler, train_dataloader, num_epochs=10, device='cuda'):
    """Main training loop"""
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    # Move models to device
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    
    # Setup optimizers
    text_optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=1e-4)
    unet_optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"📖 Epoch {epoch + 1}/{num_epochs}")
        
        text_encoder.train()
        unet.train()
        
        total_loss = 0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Compute loss
            loss = compute_training_loss(text_encoder, unet, scheduler, batch, device)
            
            # Backward pass
            text_optimizer.zero_grad()
            unet_optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            text_optimizer.step()
            unet_optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(text_encoder, unet, epoch + 1, avg_loss)
    
    print("✅ Training completed!")
    return text_encoder, unet

def save_checkpoint(text_encoder, unet, epoch, loss):
    """Save training checkpoint"""
    checkpoint_dir = f"./checkpoints/epoch_{epoch}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save LoRA weights
    text_encoder.save_pretrained(f"{checkpoint_dir}/text_encoder")
    unet.save_pretrained(f"{checkpoint_dir}/unet")
    
    # Save training info
    info = {
        'epoch': epoch,
        'loss': loss,
        'timestamp': str(torch.datetime.now())
    }
    
    with open(f"{checkpoint_dir}/info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"💾 Checkpoint saved to {checkpoint_dir}")

def main():
    """Main function"""
    print("🎯 Stable Diffusion LoRA Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    if device == 'cpu':
        print("⚠️  Warning: Training on CPU will be very slow!")
        print("   Consider using a GPU for reasonable training times.")
    
    # Setup components
    tokenizer, text_encoder, unet, scheduler = setup_training_components()
    
    # Setup LoRA
    text_encoder, unet = setup_lora_for_training(text_encoder, unet)
    
    # Setup dataset
    data_dir = "./training_data"  # You need to create this directory
    if not os.path.exists(data_dir):
        print(f"❌ Training data directory not found: {data_dir}")
        print("   Please create this directory and add your training images + text files")
        print("   Format: image.jpg + image.txt (same filename, different extensions)")
        return
    
    dataset = StableDiffusionDataset(data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Train model
    trained_text_encoder, trained_unet = train_model(
        text_encoder, unet, scheduler, dataloader, 
        num_epochs=10, device=device
    )
    
    # Save final model
    final_output_dir = "./final_lora_model"
    os.makedirs(final_output_dir, exist_ok=True)
    
    trained_text_encoder.save_pretrained(f"{final_output_dir}/text_encoder")
    trained_unet.save_pretrained(f"{final_output_dir}/unet")
    
    print(f"🎉 Training completed! Final model saved to {final_output_dir}")
    print("📚 Next steps:")
    print("   1. Test your fine-tuned model")
    print("   2. Generate images with your custom style")
    print("   3. Iterate and improve")

if __name__ == "__main__":
    main()
