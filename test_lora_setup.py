#!/usr/bin/env python3
"""
Test script to verify LoRA setup works correctly
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel

def test_lora_setup():
    """Test if LoRA can be applied to CLIP model"""
    print("🧪 Testing LoRA setup...")
    
    try:
        # Load a small CLIP model for testing
        print("📥 Loading CLIP model...")
        model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32
        )
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Number of layers: {model.config.num_hidden_layers}")
        
        # Setup LoRA config
        print("🔧 Setting up LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        print("🎯 Applying LoRA to model...")
        lora_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        print("📊 Trainable parameters:")
        lora_model.print_trainable_parameters()
        
        # Test forward pass
        print("🚀 Testing forward pass...")
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        print(f"   Expected: [{batch_size}, {seq_len}, {model.config.hidden_size}]")
        
        # Test saving and loading
        print("💾 Testing save/load...")
        test_dir = "./test_lora_model"
        lora_model.save_pretrained(test_dir)
        print(f"✅ Model saved to {test_dir}")
        
        # Load the saved model
        loaded_model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32
        )
        loaded_lora_model = get_peft_model(loaded_model, lora_config)
        loaded_lora_model.load_state_dict(torch.load(f"{test_dir}/adapter_model.bin"))
        
        print("✅ Model loaded successfully!")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        print("🧹 Cleaned up test files")
        
        print("\n🎉 LoRA setup test PASSED!")
        print("✅ You can now proceed with fine-tuning!")
        
    except Exception as e:
        print(f"❌ LoRA setup test FAILED: {e}")
        print("   Please check your installation and try again")

if __name__ == "__main__":
    test_lora_setup()
