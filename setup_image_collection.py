#!/usr/bin/env python3
"""
Setup script for Image Collection Pipeline
Installs dependencies and helps configure API keys
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "image_collection_requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("Checking installed packages...")
    
    required_packages = ['requests', 'PIL', 'bs4']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'bs4':
                import bs4
            else:
                __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def create_api_keys_file():
    """Create a template for API keys"""
    api_keys_content = """# API Keys for Image Collection Pipeline
# Get free API keys from these services:

# Pixabay (Recommended - 5000 requests/day free)
# Register at: https://pixabay.com/api/docs/
PIXABAY_API_KEY=your_pixabay_api_key_here

# Pexels (1000 requests/hour free)
# Register at: https://www.pexels.com/api/
PEXELS_API_KEY=your_pexels_api_key_here

# Unsplash (5000 requests/hour free)
# Register at: https://unsplash.com/developers
UNSPLASH_ACCESS_KEY=your_unsplash_access_key_here

# Instructions:
# 1. Replace the values above with your actual API keys
# 2. Save this file as .env in your project directory
# 3. The pipeline will automatically load these keys
# 4. Or set them as environment variables in your system
"""
    
    with open("api_keys_template.txt", "w") as f:
        f.write(api_keys_content)
    
    print("✅ Created api_keys_template.txt")
    print("📝 Edit this file with your API keys")

def create_simple_runner():
    """Create a simple script to run the pipeline"""
    runner_content = '''#!/usr/bin/env python3
# Simple Image Collection Runner
# Run this to start collecting images

from image_collection_pipeline import ImageCollectionPipeline

def run_simple_collection():
    # Simple configuration - modify these as needed
    queries = [
        "anime character",
        "cartoon face", 
        "illustration"
    ]
    
    images_per_query = 20  # Start with fewer images for testing
    
    print("Starting simple image collection...")
    print(f"Queries: {queries}")
    print(f"Images per query: {images_per_query}")
    
    # Create pipeline
    pipeline = ImageCollectionPipeline(output_dir="my_training_dataset")
    
    # Run pipeline
    final_path = pipeline.run_complete_pipeline(queries, images_per_query)
    
    if final_path:
        print(f"\\nCollection complete! Dataset at: {final_path}")
    else:
        print("\\nCollection failed. Check the error messages above.")

if __name__ == "__main__":
    run_simple_collection()
'''
    
    with open("run_simple_collection.py", "w") as f:
        f.write(runner_content)
    
    print("✅ Created run_simple_collection.py")
    print("🚀 Run this script to start collecting images")

def main():
    """Main setup function"""
    print("=" * 60)
    print("IMAGE COLLECTION PIPELINE SETUP")
    print("=" * 60)
    
    # Step 1: Install packages
    if not install_requirements():
        print("❌ Setup failed. Please install packages manually.")
        return
    
    # Step 2: Check packages
    if not check_packages():
        print("❌ Some packages are missing. Please install them manually.")
        return
    
    # Step 3: Create API keys template
    create_api_keys_file()
    
    # Step 4: Create simple runner
    create_simple_runner()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE! 🎉")
    print("=" * 60)
    print("Next steps:")
    print("1. Get free API keys from the services listed in api_keys_template.txt")
    print("2. Edit api_keys_template.txt with your API keys")
    print("3. Rename it to .env or set environment variables")
    print("4. Run: python run_simple_collection.py")
    print("5. Or run: python image_collection_pipeline.py")
    print("\nHappy image collecting! 🚀")

if __name__ == "__main__":
    main()
