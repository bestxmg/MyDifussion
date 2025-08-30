#!/usr/bin/env python3
"""
Test LoRA Quality - Compare LoRA vs Base Model
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def test_lora_quality():
    """Test LoRA quality by comparing with base model"""
    
    print("Loading base model...")
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()
    
    # Test prompt
    prompt = "danganronpa character, anime style, colorful, vibrant"
    negative_prompt = "low quality, blurry, distorted, ugly, bad anatomy"
    
    print("Generating image with BASE MODEL...")
    
    # Generate with base model
    base_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.0,
        width=512,
        height=512
    ).images[0]
    
    # Save base image
    base_image.save("test_base_model.png")
    print("Base model image saved as: test_base_model.png")
    
    # Check if LoRA exists
    lora_path = "danganronpa_lora"
    if not os.path.exists(lora_path):
        print(f"LoRA path {lora_path} does not exist!")
        return
    
    print("Loading LoRA...")
    
    try:
        # Save original UNet
        original_unet = pipe.unet
        
        # Load LoRA
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        
        print("Generating image with LoRA...")
        
        # Generate with LoRA
        lora_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.0,
            width=512,
            height=512
        ).images[0]
        
        # Save LoRA image
        lora_image.save("test_with_lora.png")
        print("LoRA image saved as: test_with_lora.png")
        
        # Restore original UNet
        pipe.unet = original_unet
        print("LoRA disabled, back to base model")
        
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Your LoRA might be corrupted or incompatible")

if __name__ == "__main__":
    test_lora_quality()
