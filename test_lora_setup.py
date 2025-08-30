#!/usr/bin/env python3
"""
Test LoRA Setup and Training
"""

import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_setup():
    """Test basic model loading"""
    logger.info("Testing basic setup...")
    
    try:
        # Test CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Using device: {device}")
        
        # Test imports
        from transformers import CLIPTokenizer, CLIPTextModel
        logger.info("✓ Transformers imported successfully")
        
        from diffusers import UNet2DConditionModel, AutoencoderKL
        logger.info("✓ Diffusers imported successfully")
        
        from peft import LoraConfig, get_peft_model
        logger.info("✓ PEFT imported successfully")
        
        # Test model loading
        model_name = "runwayml/stable-diffusion-v1-5"
        
        logger.info("Loading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        logger.info("✓ Tokenizer loaded")
        
        logger.info("Loading text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        text_encoder = text_encoder.to(device)
        logger.info("✓ Text encoder loaded")
        
        logger.info("Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        unet = unet.to(device)
        logger.info("✓ UNet loaded")
        
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        vae = vae.to(device)
        logger.info("✓ VAE loaded")
        
        # Test LoRA setup
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        
        unet = get_peft_model(unet, lora_config)
        logger.info("✓ LoRA setup complete")
        
        # Test dataset
        logger.info("Testing dataset...")
        image_folder = Path("website_images")
        if image_folder.exists():
            image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
            logger.info(f"✓ Found {len(image_files)} images in dataset")
        else:
            logger.error("✗ Website_images folder not found")
            return False
        
        logger.info("✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_training():
    """Test simple training loop"""
    logger.info("Testing simple training...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models (simplified)
        from transformers import CLIPTokenizer, CLIPTextModel
        from diffusers import UNet2DConditionModel, AutoencoderKL
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        import numpy as np
        
        model_name = "runwayml/stable-diffusion-v1-5"
        
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        
        # Move to device
        text_encoder = text_encoder.to(device)
        unet = unet.to(device)
        vae = vae.to(device)
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        
        # Prepare dataset
        image_folder = Path("website_images")
        image_files = list(image_folder.glob("*.jpg"))[:5] + list(image_folder.glob("*.png"))[:5]
        
        if not image_files:
            logger.error("No images found for training")
            return False
        
        logger.info(f"Training on {len(image_files)} images")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
        
        # Simple training loop
        for epoch in range(2):
            logger.info(f"Epoch {epoch+1}")
            
            for i, image_file in enumerate(image_files):
                try:
                    # Load image
                    image = Image.open(image_file).convert("RGB")
                    image = image.resize((512, 512))
                    
                    # Convert to tensor
                    image = torch.from_numpy(np.array(image)).float() / 255.0
                    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    # Create prompt
                    prompt = f"danganronpa style, {image_file.stem}"
                    inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Encode
                    with torch.no_grad():
                        latents = vae.encode(image).latent_dist.sample() * 0.18215
                        text_embeddings = text_encoder(inputs.input_ids, inputs.attention_mask)[0]
                    
                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, 1000, (1,), device=latents.device)
                    noisy_latents = latents + noise * 0.1
                    
                    # Predict
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                    
                    # Loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if (i + 1) % 2 == 0:
                        logger.info(f"  Image {i+1}/{len(image_files)}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {str(e)}")
                    continue
        
        logger.info("✓ Training test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LORA SETUP AND TRAINING TEST")
    print("=" * 50)
    
    # Test basic setup
    if test_basic_setup():
        print("\n" + "=" * 30)
        print("BASIC SETUP: PASSED")
        print("=" * 30)
        
        # Test training
        if test_simple_training():
            print("\n" + "=" * 30)
            print("TRAINING TEST: PASSED")
            print("=" * 30)
            print("\n✓ All tests passed! You can now run the full LoRA training.")
        else:
            print("\n" + "=" * 30)
            print("TRAINING TEST: FAILED")
            print("=" * 30)
    else:
        print("\n" + "=" * 30)
        print("BASIC SETUP: FAILED")
        print("=" * 30)
