#!/usr/bin/env python3
"""
Stable Diffusion Parameter Guide - Understanding Inference Steps & Guidance Scale
"""

def print_parameter_guide():
    """Print comprehensive parameter guide"""
    
    print("🎨 STABLE DIFFUSION PARAMETER GUIDE")
    print("=" * 60)
    
    print("\n🔄 INFERENCE STEPS (Denoising Steps)")
    print("-" * 40)
    print("What it controls: Image quality and generation speed")
    print("How it works: Number of noise reduction iterations")
    print()
    
    print("📊 Step Ranges & Results:")
    print("┌─────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Steps   │ Speed       │ Quality     │ Use Case    │")
    print("├─────────┼─────────────┼─────────────┼─────────────┤")
    print("│ 5-10    │ ⚡ Very Fast │ 🟡 Basic     │ Quick tests │")
    print("│ 15-20   │ 🚀 Fast      │ 🟢 Good      │ Daily use   │")
    print("│ 25-30   │ 🐌 Medium    │ 🔵 High      │ Quality     │")
    print("│ 35-50   │ 🐌 Slow      │ 🔴 Maximum   │ Final art   │")
    print("└─────────┴─────────────┴─────────────┴─────────────┘")
    
    print("\n💡 Step-by-Step Process:")
    print("   Step 1-5:   Basic shapes and composition")
    print("   Step 6-15:  Objects and main features emerge")
    print("   Step 16-25: Details and textures develop")
    print("   Step 26-35: Fine details and refinements")
    print("   Step 36-50: Maximum quality (small improvements)")
    
    print("\n🎭 GUIDANCE SCALE (CFG Scale)")
    print("-" * 40)
    print("What it controls: How closely AI follows your prompt")
    print("How it works: Balance between creativity and accuracy")
    print()
    
    print("📊 Scale Ranges & Results:")
    print("┌─────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Scale   │ Creativity  │ Accuracy    │ Result      │")
    print("├─────────┼─────────────┼─────────────┼─────────────┤")
    print("│ 1.0-3.0 │ 🎨 Very High│ 🟡 Low       │ Artistic    │")
    print("│ 4.0-6.0 │ 🎨 High      │ 🟢 Medium    │ Creative    │")
    print("│ 7.0-9.0 │ 🎨 Medium    │ 🔵 High      │ Balanced    │")
    print("│ 10-15   │ 🎨 Low       │ 🔴 Very High │ Accurate    │")
    print("│ 16-20   │ 🎨 Very Low  │ 🔴 Maximum   │ Rigid       │")
    print("└─────────┴─────────────┴─────────────┴─────────────┘")
    
    print("\n💡 Real Examples:")
    print("   Prompt: 'A cat sitting on a chair'")
    print("   Scale 1.0: Might generate abstract art with cat-like elements")
    print("   Scale 5.0: Cat-like creature, creative interpretation")
    print("   Scale 7.0: Clear cat, sitting, chair visible (recommended)")
    print("   Scale 12.0: Exact cat, exact chair, very literal")
    print("   Scale 18.0: Photorealistic, rigid, no artistic flair")

def print_recommended_settings():
    """Print recommended parameter combinations"""
    
    print("\n🎯 RECOMMENDED SETTINGS BY USE CASE")
    print("=" * 60)
    
    print("\n🚀 FAST PREVIEW (Quick testing):")
    print("   Steps: 10-15")
    print("   Guidance: 6.0-7.0")
    print("   Result: Basic quality, fast generation")
    
    print("\n⚖️  BALANCED (Daily use):")
    print("   Steps: 20-25")
    print("   Guidance: 7.0-8.0")
    print("   Result: Good quality, reasonable speed")
    
    print("\n🎨 QUALITY (Important images):")
    print("   Steps: 30-35")
    print("   Guidance: 7.5-8.5")
    print("   Result: High quality, slower generation")
    
    print("\n🔴 MAXIMUM QUALITY (Final artwork):")
    print("   Steps: 40-50")
    print("   Guidance: 8.0-9.0")
    print("   Result: Maximum detail, slow generation")

def print_troubleshooting():
    """Print troubleshooting tips"""
    
    print("\n🔧 TROUBLESHOOTING TIPS")
    print("=" * 60)
    
    print("\n❌ Common Problems & Solutions:")
    print()
    print("Problem: Images are too blurry")
    print("Solution: Increase steps (20→30) and guidance (7.0→8.0)")
    print()
    print("Problem: Images don't match prompt")
    print("Solution: Increase guidance scale (7.0→10.0)")
    print()
    print("Problem: Generation takes too long")
    print("Solution: Decrease steps (30→20) and guidance (8.0→7.0)")
    print()
    print("Problem: Images are too rigid/artificial")
    print("Solution: Decrease guidance scale (10.0→7.0)")
    print()
    print("Problem: Low quality results")
    print("Solution: Increase steps (15→25) and use balanced guidance (7.0)")

def print_advanced_tips():
    """Print advanced usage tips"""
    
    print("\n🚀 ADVANCED USAGE TIPS")
    print("=" * 60)
    
    print("\n🎭 Prompt Engineering:")
    print("   • Use specific, descriptive language")
    print("   • Include style keywords (anime, realistic, oil painting)")
    print("   • Add quality boosters (high quality, detailed, masterpiece)")
    print("   • Use negative prompts to avoid unwanted elements")
    
    print("\n⚡ Speed vs Quality Trade-offs:")
    print("   • Steps: Each step adds ~2-3 seconds on GTX 1650")
    print("   • Guidance: Higher values = more stable but slower")
    print("   • Resolution: Higher = more VRAM usage")
    
    print("\n🔄 Iterative Refinement:")
    print("   • Start with low steps (15) for concept testing")
    print("   • Increase steps (25-30) for final version")
    print("   • Adjust guidance based on prompt adherence needs")
    print("   • Use same seed for consistent variations")

def main():
    """Main function"""
    print_parameter_guide()
    print_recommended_settings()
    print_troubleshooting()
    print_advanced_tips()
    
    print("\n" + "=" * 60)
    print("💡 QUICK REFERENCE:")
    print("   • Daily use: 20 steps, 7.0 guidance")
    print("   • Fast test: 15 steps, 6.5 guidance")
    print("   • High quality: 30 steps, 8.0 guidance")
    print("   • Maximum quality: 40+ steps, 8.5+ guidance")
    
    print("\n🎯 For your GTX 1650 (4GB VRAM):")
    print("   • Recommended: 20-25 steps, 7.0-7.5 guidance")
    print("   • Maximum: 30-35 steps (to avoid VRAM issues)")
    print("   • Resolution: 512x512 (optimal for your card)")
    
    print("\n✅ Your current GUI settings are perfect!")
    print("   Steps: 20 (balanced quality/speed)")
    print("   Guidance: 7.0 (good prompt adherence)")
    print("   Resolution: 512x512 (optimal for 4GB VRAM)")

if __name__ == "__main__":
    main()
