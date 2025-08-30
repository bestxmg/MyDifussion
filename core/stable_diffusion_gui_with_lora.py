#!/usr/bin/env python3
"""
Stable Diffusion GUI with LoRA Support - Enhanced version of existing GUI
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
import time
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

class StableDiffusionGUIWithLoRA:
    def __init__(self, root):
        self.root = root
        self.root.title("Stable Diffusion Image Generator with LoRA")
        self.root.geometry("900x800")
        
        # Model and pipeline
        self.pipe = None
        self.is_loading = False
        self.is_generating = False
        self.should_stop = False
        
        # LoRA settings
        self.lora_loaded = False
        self.lora_path = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Load model in background
        self.load_model_async()
    
    def create_widgets(self):
        """Create the GUI widgets with LoRA support"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎨 Stable Diffusion with LoRA", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # LoRA Control Frame
        lora_frame = ttk.LabelFrame(main_frame, text="LoRA Settings", padding="10")
        lora_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # LoRA Path
        ttk.Label(lora_frame, text="LoRA Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.lora_path_var = tk.StringVar()
        self.lora_path_entry = ttk.Entry(lora_frame, textvariable=self.lora_path_var, width=50)
        self.lora_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.browse_lora_btn = ttk.Button(lora_frame, text="Browse", command=self.browse_lora)
        self.browse_lora_btn.grid(row=0, column=2)
        
        # Load LoRA Button
        self.load_lora_btn = ttk.Button(lora_frame, text="Load LoRA", command=self.load_lora_async)
        self.load_lora_btn.grid(row=1, column=1, pady=(10, 0))
        
        # LoRA Status
        self.lora_status_var = tk.StringVar(value="No LoRA loaded")
        self.lora_status_label = ttk.Label(lora_frame, textvariable=self.lora_status_var, 
                                          foreground="orange")
        self.lora_status_label.grid(row=2, column=0, columnspan=3, pady=(5, 0))
        
        # Prompt input
        ttk.Label(main_frame, text="Describe the image you want:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        self.prompt_entry = scrolledtext.ScrolledText(main_frame, height=3, width=70)
        self.prompt_entry.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.prompt_entry.insert(tk.END, "danganronpa character, anime style, colorful, vibrant")
        
        # Negative prompt
        ttk.Label(main_frame, text="Negative prompt (what to avoid):").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        self.negative_prompt_entry = scrolledtext.ScrolledText(main_frame, height=2, width=70)
        self.negative_prompt_entry.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.negative_prompt_entry.insert(tk.END, "low quality, blurry, distorted, ugly, bad anatomy")
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Generation Settings", padding="10")
        settings_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Resolution
        ttk.Label(settings_frame, text="Resolution:").grid(row=0, column=0, sticky=tk.W)
        self.resolution_var = tk.StringVar(value="512x512")
        resolution_combo = ttk.Combobox(settings_frame, textvariable=self.resolution_var, 
                                       values=["256x256", "384x384", "512x512", "448x448"], 
                                       state="readonly", width=10)
        resolution_combo.grid(row=0, column=1, padx=(10, 0))
        
        # Steps
        ttk.Label(settings_frame, text="Inference Steps:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.steps_var = tk.StringVar(value="20")
        steps_spin = ttk.Spinbox(settings_frame, from_=5, to=50, textvariable=self.steps_var, width=8)
        steps_spin.grid(row=0, column=3, padx=(10, 0))
        
        # Guidance
        ttk.Label(settings_frame, text="Guidance Scale:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.guidance_var = tk.StringVar(value="7.0")
        guidance_spin = ttk.Spinbox(settings_frame, from_=1.0, to=20.0, increment=0.5, 
                                   textvariable=self.guidance_var, width=8)
        guidance_spin.grid(row=1, column=1, padx=(10, 0), pady=(10, 0))
        
        # Number of Images
        ttk.Label(settings_frame, text="Number of Images:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=(10, 0))
        self.num_images_var = tk.StringVar(value="1")
        num_images_spin = ttk.Spinbox(settings_frame, from_=1, to=4, textvariable=self.num_images_var, width=8)
        num_images_spin.grid(row=1, column=3, padx=(10, 0), pady=(10, 0))
        
        # Generate/Stop button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)
        
        # Generate button
        self.generate_btn = ttk.Button(button_frame, text="🎨 Generate Image", 
                                      command=self.generate_image, state="disabled")
        self.generate_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Stop button
        self.stop_btn = ttk.Button(button_frame, text="⏹️ Stop", command=self.stop_generation, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Status
        self.status_var = tk.StringVar(value="Loading model...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                     font=("Arial", 10, "bold"))
        self.status_label.grid(row=9, column=0, columnspan=2, pady=(10, 0))
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Generated Image", padding="10")
        image_frame.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Image label
        self.image_label = ttk.Label(image_frame, text="No image generated yet")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Time display
        self.time_var = tk.StringVar(value="")
        self.time_label = ttk.Label(main_frame, textvariable=self.time_var)
        self.time_label.grid(row=11, column=0, columnspan=2, pady=(5, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(10, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
    
    def browse_lora(self):
        """Browse for LoRA file"""
        lora_path = filedialog.askdirectory(title="Select LoRA Directory")
        if lora_path:
            self.lora_path_var.set(lora_path)
            self.lora_status_var.set("LoRA path selected - not loaded yet")
            self.lora_status_label.config(foreground="orange")
    
    def load_lora_async(self):
        """Load LoRA asynchronously"""
        if not self.pipe:
            messagebox.showerror("Error", "Please wait for the base model to load first")
            return
        
        lora_path = self.lora_path_var.get().strip()
        if not lora_path:
            messagebox.showerror("Error", "Please select a LoRA path first")
            return
        
        if not os.path.exists(lora_path):
            messagebox.showerror("Error", f"LoRA path does not exist: {lora_path}")
            return
        
        def load_lora():
            try:
                self.root.after(0, lambda: self.lora_status_var.set("Loading LoRA..."))
                self.root.after(0, lambda: self.lora_status_label.config(foreground="blue"))
                
                # Load LoRA weights
                from peft import PeftModel
                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_path)
                
                self.root.after(0, lambda: self.lora_status_var.set(f"LoRA loaded successfully from {os.path.basename(lora_path)}"))
                self.root.after(0, lambda: self.lora_status_label.config(foreground="green"))
                self.root.after(0, lambda: self.load_lora_btn.config(state="disabled"))
                self.root.after(0, lambda: setattr(self, 'lora_loaded', True))
                
            except Exception as e:
                self.root.after(0, lambda: self.lora_status_var.set(f"Error loading LoRA: {str(e)}"))
                self.root.after(0, lambda: self.lora_status_label.config(foreground="red"))
        
        threading.Thread(target=load_lora, daemon=True).start()
    
    def load_model_async(self):
        """Load the Stable Diffusion model asynchronously"""
        def load_model():
            try:
                self.status_var.set("Loading Stable Diffusion model...")
                
                # Load pipeline
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to('cuda')
                
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
                
                self.status_var.set("Model loaded successfully! Ready to generate")
                self.generate_btn.config(state="normal")
                
            except Exception as e:
                self.status_var.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def generate_image(self):
        """Generate image with LoRA support"""
        if not self.pipe or self.is_generating:
            return
        
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        negative_prompt = self.negative_prompt_entry.get("1.0", tk.END).strip()
        
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt description")
            return
        
        resolution = self.resolution_var.get().split('x')
        width, height = int(resolution[0]), int(resolution[1])
        steps = int(self.steps_var.get())
        guidance = float(self.guidance_var.get())
        num_images = int(self.num_images_var.get())
        
        self.is_generating = True
        self.should_stop = False
        self.generate_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting generation...")
        
        def generate():
            try:
                start_time = time.time()
                
                self.root.after(0, lambda: self.progress_var.set("Generating image... Please wait"))
                self.root.after(0, lambda: self.progress_bar.config(mode='indeterminate'))
                self.root.after(0, lambda: self.progress_bar.start())
                
                # Generate image
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images
                ).images
                
                if self.should_stop:
                    return
                
                generation_time = time.time() - start_time
                
                # Display first image
                self.root.after(0, lambda: self.display_image(images[0], generation_time, len(images)))
                
                # Save all images
                if len(images) > 1:
                    self.root.after(0, lambda: self.save_multiple_images(images, prompt))
                
            except InterruptedError:
                self.root.after(0, lambda: self.show_stopped_message())
            except Exception as e:
                error_msg = f"Generation failed: {str(e)}"
                self.root.after(0, lambda: self.show_error(error_msg))
            finally:
                self.root.after(0, lambda: self.generation_complete())
        
        threading.Thread(target=generate, daemon=True).start()
    
    def display_image(self, image, generation_time, num_images=1):
        """Display the generated image"""
        try:
            # Resize image for display
            display_size = (400, 400)
            display_image = image.copy()
            display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            photo = ImageTk.PhotoImage(display_image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            self.time_var.set(f"Generated {num_images} image(s) in {generation_time:.2f} seconds")
            
        except Exception as e:
            self.image_label.configure(text=f"Error displaying image: {str(e)}")
            self.image_label.image = None
    
    def save_multiple_images(self, images, prompt):
        """Save multiple generated images"""
        try:
            output_dir = "generated_images"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            for i, image in enumerate(images):
                filename = f"lora_generated_{timestamp}_{i}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
            
            messagebox.showinfo("Success", f"Saved {len(images)} images to generated_images/")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save images: {str(e)}")
    
    def stop_generation(self):
        """Stop image generation"""
        self.should_stop = True
        self.stop_btn.config(state="disabled")
    
    def generation_complete(self):
        """Called when generation is complete"""
        self.is_generating = False
        self.generate_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress_bar.config(mode='determinate', value=100)
        self.progress_bar.stop()
        self.progress_var.set("Generation complete")
    
    def show_stopped_message(self):
        """Show stopped message"""
        self.progress_var.set("Generation stopped by user")
        self.progress_bar.stop()
    
    def show_error(self, error_msg):
        """Show error message"""
        self.progress_var.set("Error occurred")
        self.progress_bar.stop()
        messagebox.showerror("Generation Error", error_msg)

def main():
    root = tk.Tk()
    app = StableDiffusionGUIWithLoRA(root)
    root.mainloop()

if __name__ == "__main__":
    main()
