#!/usr/bin/env python3
"""
Complete Image Collection Pipeline for AI Training
Downloads images from legal sources, compresses them, and organizes them for training.
"""

import requests
import json
import os
import time
from datetime import datetime
from PIL import Image
import io
from urllib.parse import urljoin, urlparse
import hashlib

class ImageCollectionPipeline:
    def __init__(self, output_dir="training_dataset"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "raw"),
            os.path.join(self.output_dir, "compressed"),
            os.path.join(self.output_dir, "final_dataset")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def download_from_unsplash(self, query, count=50, save_dir=None):
        """Download images from Unsplash (requires free API key)"""
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "raw", query.replace(" ", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Downloading {count} images from Unsplash for query: '{query}'")
        print("Note: You need to register for a free API key at https://unsplash.com/developers")
        print("Set your API key in the script or environment variable UNSPLASH_ACCESS_KEY")
        
        # Try to get API key from environment
        access_key = os.getenv('UNSPLASH_ACCESS_KEY')
        if not access_key:
            print("No Unsplash API key found. Skipping Unsplash downloads.")
            return []
        
        try:
            search_url = "https://api.unsplash.com/search/photos"
            params = {
                "query": query,
                "per_page": min(count, 30),  # Unsplash limit per request
                "client_id": access_key
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            downloaded = []
            for i, photo in enumerate(data["results"]):
                try:
                    # Get download link
                    download_url = photo["links"]["download"]
                    
                    # Download image
                    img_response = self.session.get(download_url)
                    img_response.raise_for_status()
                    
                    # Generate filename
                    filename = f"unsplash_{query.replace(' ', '_')}_{i:04d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    downloaded.append(filepath)
                    print(f"Downloaded: {filename}")
                    
                    # Be respectful to API
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error downloading image {i}: {e}")
            
            print(f"Successfully downloaded {len(downloaded)} images from Unsplash")
            return downloaded
            
        except Exception as e:
            print(f"Error with Unsplash API: {e}")
            return []
    
    def download_from_pixabay(self, query, count=50, save_dir=None):
        """Download images from Pixabay (requires free API key)"""
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "raw", query.replace(" ", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Downloading {count} images from Pixabay for query: '{query}'")
        print("Note: You need to register for a free API key at https://pixabay.com/api/docs/")
        print("Set your API key in the script or environment variable PIXABAY_API_KEY")
        
        # Try to get API key from environment
        api_key = os.getenv('PIXABAY_API_KEY')
        if not api_key:
            print("No Pixabay API key found. Skipping Pixabay downloads.")
            return []
        
        try:
            search_url = "https://pixabay.com/api/"
            params = {
                "key": api_key,
                "q": query,
                "per_page": min(count, 200),  # Pixabay limit
                "image_type": "photo",
                "safesearch": "true"
            }
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            downloaded = []
            for i, hit in enumerate(data["hits"]):
                try:
                    # Get image URL (webformatURL is smaller, good for training)
                    img_url = hit["webformatURL"]
                    
                    # Download image
                    img_response = self.session.get(img_url)
                    img_response.raise_for_status()
                    
                    # Generate filename
                    filename = f"pixabay_{query.replace(' ', '_')}_{i:04d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    downloaded.append(filepath)
                    print(f"Downloaded: {filename}")
                    
                    # Be respectful to API
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"Error downloading image {i}: {e}")
            
            print(f"Successfully downloaded {len(downloaded)} images from Pixabay")
            return downloaded
            
        except Exception as e:
            print(f"Error with Pixabay API: {e}")
            return []
    
    def download_from_pexels(self, query, count=50, save_dir=None):
        """Download images from Pexels (requires free API key)"""
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "raw", query.replace(" ", "_"))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Downloading {count} images from Pexels for query: '{query}'")
        print("Note: You need to register for a free API key at https://www.pexels.com/api/")
        print("Set your API key in the script or environment variable PEXELS_API_KEY")
        
        # Try to get API key from environment
        api_key = os.getenv('PEXELS_API_KEY')
        if not api_key:
            print("No Pexels API key found. Skipping Pexels downloads.")
            return []
        
        try:
            search_url = "https://api.pexels.com/v1/search"
            headers = {"Authorization": api_key}
            params = {
                "q": query,
                "per_page": min(count, 80),  # Pexels limit
                "orientation": "landscape"
            }
            
            response = self.session.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            downloaded = []
            for i, photo in enumerate(data["photos"]):
                try:
                    # Get image URL (medium size is good for training)
                    img_url = photo["src"]["medium"]
                    
                    # Download image
                    img_response = self.session.get(img_url)
                    img_response.raise_for_status()
                    
                    # Generate filename
                    filename = f"pexels_{query.replace(' ', '_')}_{i:04d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    downloaded.append(filepath)
                    print(f"Downloaded: {filename}")
                    
                    # Be respectful to API
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"Error downloading image {i}: {e}")
            
            print(f"Successfully downloaded {len(downloaded)} images from Pexels")
            return downloaded
            
        except Exception as e:
            print(f"Error with Pexels API: {e}")
            return []
    
    def compress_images(self, input_dir, output_dir, quality=85, max_size=(512, 512)):
        """Compress and resize images for training"""
        print(f"Compressing images from {input_dir} to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images to compress")
        
        processed = []
        for i, filename in enumerate(image_files):
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"compressed_{filename}")
                
                # Open image
                with Image.open(input_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if larger than max_size
                    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # Save compressed image
                    img.save(output_path, 'JPEG', quality=quality, optimize=True)
                    
                    processed.append(output_path)
                    print(f"Compressed: {filename} ({i+1}/{len(image_files)})")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Successfully compressed {len(processed)} images")
        
        # Calculate storage savings
        input_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in image_files)
        output_size = sum(os.path.getsize(f) for f in processed)
        
        if input_size > 0:
            savings = (input_size - output_size) / input_size * 100
            print(f"Storage savings: {savings:.1f}%")
            print(f"Original size: {input_size / (1024*1024):.1f} MB")
            print(f"Compressed size: {output_size / (1024*1024):.1f} MB")
        
        return processed
    
    def organize_final_dataset(self, input_dir, output_dir, prefix="train_img"):
        """Organize images with consistent naming for training"""
        print(f"Organizing final dataset from {input_dir} to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in os.listdir(input_dir) 
                       if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return []
        
        # Sort files
        image_files.sort()
        
        print(f"Organizing {len(image_files)} images")
        
        organized = []
        for i, filename in enumerate(image_files):
            try:
                # Get file extension
                _, ext = os.path.splitext(filename)
                
                # Create new filename
                new_filename = f"{prefix}_{i:04d}{ext}"
                
                # Copy file with new name
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, new_filename)
                
                with open(input_path, 'rb') as f:
                    content = f.read()
                
                with open(output_path, 'wb') as f:
                    f.write(content)
                
                organized.append(output_path)
                print(f"Organized: {new_filename} ({i+1}/{len(image_files)})")
                
            except Exception as e:
                print(f"Error organizing {filename}: {e}")
        
        print(f"Successfully organized {len(organized)} images")
        return organized
    
    def run_complete_pipeline(self, queries, images_per_query=50):
        """Run the complete image collection pipeline"""
        print("=" * 60)
        print("STARTING COMPLETE IMAGE COLLECTION PIPELINE")
        print("=" * 60)
        
        # Create directories
        self.create_directories()
        
        # Track all downloaded images
        all_downloaded = []
        
        # Download from multiple sources for each query
        for query in queries:
            print(f"\n{'='*40}")
            print(f"Processing query: '{query}'")
            print(f"{'='*40}")
            
            query_downloads = []
            
            # Try multiple sources
            sources = [
                ("Pixabay", self.download_from_pixabay, query, images_per_query),
                ("Pexels", self.download_from_pexels, query, images_per_query),
                ("Unsplash", self.download_from_unsplash, query, images_per_query)
            ]
            
            for source_name, download_func, *args in sources:
                try:
                    downloads = download_func(*args)
                    query_downloads.extend(downloads)
                    print(f"Total from {source_name}: {len(downloads)} images")
                except Exception as e:
                    print(f"Error with {source_name}: {e}")
            
            all_downloaded.extend(query_downloads)
            print(f"Total for query '{query}': {len(query_downloads)} images")
        
        print(f"\n{'='*40}")
        print(f"DOWNLOAD COMPLETE: {len(all_downloaded)} total images")
        print(f"{'='*40}")
        
        if not all_downloaded:
            print("No images were downloaded. Check your API keys and internet connection.")
            return None
        
        # Compress all downloaded images
        print(f"\n{'='*40}")
        print("COMPRESSING IMAGES")
        print(f"{'='*40}")
        
        raw_dir = os.path.join(self.output_dir, "raw")
        compressed_dir = os.path.join(self.output_dir, "compressed")
        
        compressed_images = self.compress_images(raw_dir, compressed_dir)
        
        # Organize final dataset
        print(f"\n{'='*40}")
        print("ORGANIZING FINAL DATASET")
        print(f"{'='*40}")
        
        final_dir = os.path.join(self.output_dir, "final_dataset")
        final_images = self.organize_final_dataset(compressed_dir, final_dir)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"Raw images: {len(all_downloaded)}")
        print(f"Compressed images: {len(compressed_images)}")
        print(f"Final dataset: {len(final_images)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Final dataset location: {final_dir}")
        
        return final_dir

def main():
    """Main function to run the pipeline"""
    
    # Configuration
    queries = [
        "anime character",
        "cartoon face", 
        "illustration",
        "digital art",
        "character portrait"
    ]
    
    images_per_query = 30  # Adjust based on your needs
    
    # Create and run pipeline
    pipeline = ImageCollectionPipeline()
    final_dataset_path = pipeline.run_complete_pipeline(queries, images_per_query)
    
    if final_dataset_path:
        print(f"\nYour training dataset is ready at: {final_dataset_path}")
        print("You can now use these images for training your AI model!")
    else:
        print("\nPipeline failed. Check the error messages above.")

if __name__ == "__main__":
    main()
