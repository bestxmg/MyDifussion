#!/usr/bin/env python3
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
        print(f"\nCollection complete! Dataset at: {final_path}")
    else:
        print("\nCollection failed. Check the error messages above.")

if __name__ == "__main__":
    run_simple_collection()
