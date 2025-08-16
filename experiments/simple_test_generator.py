#!/usr/bin/env python3
"""
Simple Test Generator - Test if complex prompts cause black images
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def test_simple_prompt():
    """Test with the exact same settings as the diagnostic"""
    print("🧪 TESTING SIMPLE PROMPT (same as diagnostic)")
    print("=" * 50)
    
    try:
        # Load model exactly like diagnostic
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to('cuda')
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print("✅ Model loaded with diagnostic settings")
        
        # Test with EXACT same settings as diagnostic
        print("🎨 Testing with diagnostic settings...")
        test_image = pipe(
            prompt="simple test",
            num_inference_steps=5,
            guidance_scale=1.0,
            width=64,
            height=64
        ).images[0]
        
        # Check result
        img_array = np.array(test_image)
        brightness = img_array.mean()
        variation = img_array.std()
        
        print(f"   Result brightness: {brightness:.2f}")
        print(f"   Result variation: {variation:.2f}")
        
        if brightness > 10:
            print("✅ Simple prompt works (as expected)")
            return True
        else:
            print("❌ Simple prompt failed - this is unexpected!")
            return False
        
    except Exception as e:
        print(f"❌ Simple prompt test failed: {e}")
        return False
    finally:
        if 'pipe' in locals():
            del pipe
            torch.cuda.empty_cache()

def test_complex_prompt():
    """Test with our complex INFP prompt"""
    print("\n🎭 TESTING COMPLEX PROMPT (INFP girl)")
    print("=" * 50)
    
    try:
        # Load model
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to('cuda')
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print("✅ Model loaded")
        
        # Test with complex prompt
        prompt = "beautiful girl with glasses, INFP personality type, introverted, creative, idealistic, thoughtful, artistic, anime style, high quality, masterpiece, best quality, detailed face, sharp features, gentle expression, bookish, imaginative"
        
        print("🎨 Testing with complex prompt...")
        test_image = pipe(
            prompt=prompt,
            num_inference_steps=5,  # Same as diagnostic
            guidance_scale=1.0,     # Same as diagnostic
            width=64,               # Same as diagnostic
            height=64               # Same as diagnostic
        ).images[0]
        
        # Check result
        img_array = np.array(test_image)
        brightness = img_array.mean()
        variation = img_array.std()
        
        print(f"   Result brightness: {brightness:.2f}")
        print(f"   Result variation: {variation:.2f}")
        
        if brightness > 10:
            print("✅ Complex prompt works!")
            return True
        else:
            print("❌ Complex prompt produces black image")
            return False
        
    except Exception as e:
        print(f"❌ Complex prompt test failed: {e}")
        return False
    finally:
        if 'pipe' in locals():
            del pipe
            torch.cuda.empty_cache()

def test_step_by_step():
    """Test if the issue is with specific pipeline steps"""
    print("\n🔬 TESTING PIPELINE STEP BY STEP")
    print("=" * 50)
    
    try:
        # Load model
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to('cuda')
        
        print("✅ Model loaded and moved to GPU")
        
        # Test without optimizations first
        print("🎨 Testing without optimizations...")
        test_image = pipe(
            prompt="simple test",
            num_inference_steps=5,
            guidance_scale=1.0,
            width=64,
            height=64
        ).images[0]
        
        # Check result
        img_array = np.array(test_image)
        brightness = img_array.mean()
        
        print(f"   Result brightness: {brightness:.2f}")
        
        if brightness > 10:
            print("✅ Pipeline works without optimizations")
            
            # Now test with optimizations
            print("🎨 Testing with optimizations...")
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            
            test_image2 = pipe(
                prompt="simple test",
                num_inference_steps=5,
                guidance_scale=1.0,
                width=64,
                height=64
            ).images[0]
            
            img_array2 = np.array(test_image2)
            brightness2 = img_array2.mean()
            
            print(f"   Result with optimizations: {brightness2:.2f}")
            
            if brightness2 > 10:
                print("✅ Pipeline works with optimizations")
                return True
            else:
                print("❌ Optimizations break the pipeline")
                return False
        else:
            print("❌ Pipeline broken even without optimizations")
            return False
        
    except Exception as e:
        print(f"❌ Step-by-step test failed: {e}")
        return False
    finally:
        if 'pipe' in locals():
            del pipe
            torch.cuda.empty_cache()

def main():
    print("🔬 SIMPLE TEST GENERATOR - FINDING THE ROOT CAUSE")
    print("=" * 60)
    print("Let's find out why complex prompts fail but simple ones work")
    print()
    
    # Test 1: Simple prompt (should work)
    simple_ok = test_simple_prompt()
    
    # Test 2: Complex prompt (might fail)
    complex_ok = test_complex_prompt()
    
    # Test 3: Pipeline step by step
    pipeline_ok = test_step_by_step()
    
    # Analysis
    print("\n📋 ANALYSIS")
    print("=" * 40)
    
    if simple_ok and not complex_ok:
        print("🎯 The issue is with COMPLEX PROMPTS!")
        print("   Simple prompts work fine")
        print("   Complex prompts cause black images")
        print("   Possible causes:")
        print("   - Token length issues")
        print("   - Memory overflow with complex prompts")
        print("   - Model confusion with detailed descriptions")
        
    elif not simple_ok:
        print("🎯 The issue is FUNDAMENTAL!")
        print("   Even simple prompts fail")
        print("   This contradicts the diagnostic")
        print("   Possible causes:")
        print("   - Model corruption")
        print("   - Library version issues")
        print("   - GPU memory corruption")
        
    elif simple_ok and complex_ok:
        print("🎯 Both work - the issue is elsewhere!")
        print("   Simple and complex prompts both work")
        print("   The problem must be in the generation settings")
        print("   or resource management")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    if simple_ok and not complex_ok:
        print("1. Try shorter, simpler prompts")
        print("2. Break complex prompts into parts")
        print("3. Use fewer descriptive words")
        print("4. Check if it's a token length issue")
    
    elif not simple_ok:
        print("1. Reinstall diffusers library")
        print("2. Clear model cache")
        print("3. Check for library conflicts")
        print("4. Try different model version")
    
    print("\n5. For now, use CPU generation (it works reliably)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    input("\nPress Enter to exit...")
