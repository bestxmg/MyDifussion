#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Stable Diffusion
This script fine-tunes Stable Diffusion using LoRA (Low-Rank Adaptation)
"""

import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
from PIL import Image
import json

def setup_model_and_tokenizer():
    """Load the base model and tokenizer"""
    print("🔄 Loading base model and tokenizer...")
    
    # Load the base model
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for training
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float32
    )
    
    print(f"✅ Model loaded: {model_id}")
    print(f"✅ Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    print(f"✅ Text encoder loaded with hidden size: {text_encoder.config.hidden_size}")
    
    return pipe, tokenizer, text_encoder

def setup_lora_config():
    """Configure LoRA parameters"""
    print("🔧 Setting up LoRA configuration...")
    
    lora_config = LoraConfig(
        r=16,  # Rank of the LoRA matrices (higher = more parameters, more flexible)
        lora_alpha=32,  # Scaling factor for LoRA weights
        target_modules=["q_proj", "v_proj"],  # Which attention modules to fine-tune
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Don't train bias terms
        task_type="CAUSAL_LM"  # Task type for the model
    )
    
    print(f"✅ LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"✅ Target modules: {lora_config.target_modules}")
    
    return lora_config

def apply_lora_to_model(text_encoder, lora_config):
    """Apply LoRA to the text encoder"""
    print("🎯 Applying LoRA to text encoder...")
    
    # Convert to PEFT model
    text_encoder = get_peft_model(text_encoder, lora_config)
    
    # Print trainable parameters
    text_encoder.print_trainable_parameters()
    
    return text_encoder

def prepare_training_data():
    """Prepare training data (you'll need to customize this)"""
    print("📚 Preparing training data...")
    
    # Example training data structure
    training_data = [
        {
            "prompt": "a photo of a majestic dragon",
            "image_path": "path/to/dragon_image.jpg"  # You'll need actual images
        },
        {
            "prompt": "a beautiful castle at sunset",
            "image_path": "path/to/castle_image.jpg"
        }
        # Add more training examples
    ]
    
    print(f"✅ Prepared {len(training_data)} training examples")
    print("⚠️  Note: You need to provide actual image files and customize this function")
    
    return training_data

def train_lora_model(text_encoder, training_data, num_epochs=10):
    """Train the LoRA model"""
    print(f"🚀 Starting LoRA training for {num_epochs} epochs...")
    
    # Set to training mode
    text_encoder.train()
    
    # Setup optimizer (only for LoRA parameters)
    optimizer = torch.optim.AdamW(
        text_encoder.parameters(),
        lr=1e-4,  # Learning rate
        weight_decay=0.01
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"📖 Epoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        num_batches = len(training_data)
        
        for i, data in enumerate(training_data):
            # This is a simplified training loop
            # In practice, you'd need to:
            # 1. Load and preprocess images
            # 2. Tokenize text prompts
            # 3. Compute loss (e.g., reconstruction loss)
            # 4. Backpropagate and update weights
            
            print(f"  Batch {i + 1}/{num_batches}: {data['prompt']}")
            
            # Placeholder for actual training logic
            # You'll need to implement the actual training step
            
        print(f"  Epoch {epoch + 1} completed")
    
    print("✅ LoRA training completed!")
    return text_encoder

def save_lora_model(text_encoder, output_dir="./lora_output"):
    """Save the trained LoRA model"""
    print(f"💾 Saving LoRA model to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the LoRA adapter
    text_encoder.save_pretrained(output_dir)
    
    # Save configuration
    config = {
        "model_type": "lora",
        "base_model": "runwayml/stable-diffusion-v1-5",
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ LoRA model saved to {output_dir}")

def load_and_test_lora_model(pipe, lora_path):
    """Load and test the trained LoRA model"""
    print(f"🧪 Loading and testing LoRA model from {lora_path}...")
    
    # Load LoRA weights
    pipe.load_lora_weights(lora_path)
    
    # Test generation
    prompt = "a majestic dragon flying over a castle at sunset"
    
    print(f"🎨 Generating image with prompt: '{prompt}'")
    
    # Generate image
    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    # Save test image
    output_path = "test_generation.png"
    image.save(output_path)
    print(f"✅ Test image saved to {output_path}")
    
    return pipe

def main():
    """Main function"""
    print("🎯 Stable Diffusion LoRA Fine-tuning")
    print("=" * 50)
    
    # Setup
    pipe, tokenizer, text_encoder = setup_model_and_tokenizer()
    lora_config = setup_lora_config()
    
    # Apply LoRA
    text_encoder = apply_lora_to_model(text_encoder, lora_config)
    
    # Prepare training data
    training_data = prepare_training_data()
    
    # Train (this is where you'd implement actual training)
    print("\n⚠️  IMPORTANT: This script shows the setup but doesn't implement actual training!")
    print("   You need to:")
    print("   1. Provide actual training images")
    print("   2. Implement the training loop with proper loss computation")
    print("   3. Handle data loading and preprocessing")
    print("   4. Implement proper evaluation metrics")
    
    # For demonstration, we'll skip actual training
    print("\n🚀 For actual training, you would call:")
    print("   text_encoder = train_lora_model(text_encoder, training_data)")
    
    # Save the model (even without training, to show the structure)
    save_lora_model(text_encoder)
    
    print("\n🎉 LoRA fine-tuning setup completed!")
    print("📚 Next steps:")
    print("   1. Collect training data (images + text descriptions)")
    print("   2. Implement the training loop")
    print("   3. Train the model")
    print("   4. Test and iterate")

if __name__ == "__main__":
    main()
