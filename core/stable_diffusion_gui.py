#!/usr/bin/env python3
"""
Stable Diffusion GUI - User-friendly interface for image generation
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

class StableDiffusionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stable Diffusion Image Generator")
        self.root.geometry("800x700")
        
        # Model and pipeline
        self.pipe = None
        self.is_loading = False
        self.is_generating = False
        self.should_stop = False
        
        # Create GUI elements
        self.create_widgets()
        
        # Load model in background
        self.load_model_async()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎨 Stable Diffusion Image Generator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Prompt input
        ttk.Label(main_frame, text="Describe the image you want:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.prompt_entry = scrolledtext.ScrolledText(main_frame, height=3, width=70)
        self.prompt_entry.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.prompt_entry.insert(tk.END, "black dog, sitting, cute, high quality, detailed")
        
        # Negative prompt
        ttk.Label(main_frame, text="Negative prompt (what to avoid):").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        self.negative_prompt_entry = scrolledtext.ScrolledText(main_frame, height=2, width=70)
        self.negative_prompt_entry.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.negative_prompt_entry.insert(tk.END, "low quality, blurry, distorted, ugly, bad anatomy")
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Generation Settings", padding="10")
        settings_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
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
        
        # Help button for parameters
        help_btn = ttk.Button(settings_frame, text="❓ Help", command=self.show_parameter_help, width=8)
        help_btn.grid(row=1, column=2, padx=(20, 0), pady=(10, 0))
        
        # Help info label
        help_info = ttk.Label(settings_frame, text="Click for detailed explanation of steps & guidance", 
                             font=("Arial", 8), foreground="gray")
        help_info.grid(row=1, column=3, padx=(5, 0), pady=(10, 0))
        
        # Generate/Stop button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        # Generate button
        self.generate_btn = ttk.Button(button_frame, text="🎨 Generate Image", 
                                      command=self.generate_image, state="disabled")
        self.generate_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Stop button
        self.stop_btn = ttk.Button(button_frame, text="⏹️ Stop Generation", 
                                  command=self.stop_generation, state="disabled")
        self.stop_btn.grid(row=0, column=1)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Generation Progress", padding="10")
        progress_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Progress text
        self.progress_var = tk.StringVar(value="Ready to generate")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        progress_label.grid(row=1, column=0, columnspan=2)
        
        # Step counter
        self.step_var = tk.StringVar(value="")
        step_label = ttk.Label(progress_frame, textvariable=self.step_var, font=("Arial", 10))
        step_label.grid(row=2, column=0, columnspan=2, pady=(5, 0))
        
        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Generated Image", padding="10")
        image_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="Image will appear here after generation")
        self.image_label.grid(row=0, column=0)
        
        # Save button
        self.save_btn = ttk.Button(main_frame, text="💾 Save Image", 
                                  command=self.save_image, state="disabled")
        self.save_btn.grid(row=9, column=0, columnspan=2, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Loading model... Please wait")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def load_model_async(self):
        """Load the model in a background thread"""
        def load_model():
            try:
                self.status_var.set("Loading Stable Diffusion model...")
                
                # Load with working configuration
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,  # Key fix for 4GB VRAM
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                self.pipe = self.pipe.to('cuda')
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
                
                self.status_var.set("Model loaded successfully! Ready to generate")
                self.generate_btn.config(state="normal")
                
            except Exception as e:
                self.status_var.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
        
        # Run in background thread
        threading.Thread(target=load_model, daemon=True).start()
    
    def generate_image(self):
        """Generate image based on user input"""
        if not self.pipe or self.is_generating:
            return
        
        # Get user input
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        negative_prompt = self.negative_prompt_entry.get("1.0", tk.END).strip()
        
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt description")
            return
        
        # Get settings
        resolution = self.resolution_var.get().split('x')
        width, height = int(resolution[0]), int(resolution[1])
        steps = int(self.steps_var.get())
        guidance = float(self.guidance_var.get())
        
        # Start generation in background
        self.is_generating = True
        self.should_stop = False
        self.generate_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        # Reset progress
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting generation...")
        self.step_var.set("Generating image...")
        
        def generate():
            try:
                start_time = time.time()
                
                # Update progress to show generation is happening
                self.root.after(0, lambda: self.progress_var.set("Generating image... Please wait"))
                self.root.after(0, lambda: self.progress_bar.config(mode='indeterminate'))
                self.root.after(0, lambda: self.progress_bar.start())
                
                # Generate image without callback for now (to fix the error)
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    num_images_per_prompt=1
                ).images[0]
                
                if self.should_stop:
                    return
                
                generation_time = time.time() - start_time
                
                # Display image
                self.root.after(0, lambda: self.display_image(image, generation_time))
                
            except InterruptedError:
                self.root.after(0, lambda: self.show_stopped_message())
            except Exception as e:
                error_msg = f"Generation failed: {str(e)}"
                self.root.after(0, lambda: self.show_error(error_msg))
            finally:
                self.root.after(0, lambda: self.generation_complete())
        
        threading.Thread(target=generate, daemon=True).start()
    
    def update_progress(self, current_step, total_steps, progress_percent):
        """Update progress bar and step counter"""
        self.progress_bar['value'] = progress_percent
        self.progress_var.set(f"Generating... Step {current_step + 1} of {total_steps}")
        self.step_var.set(f"Step {current_step + 1} / {total_steps} ({progress_percent:.1f}%)")
        self.root.update_idletasks()
    
    def display_image(self, image, generation_time):
        """Display the generated image"""
        try:
            # Resize for display (keep aspect ratio)
            display_size = (400, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update display
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Update progress to completion
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')
            self.progress_bar['value'] = 100
            self.progress_var.set(f"✅ Generation completed in {generation_time:.2f} seconds!")
            self.step_var.set("🎉 Image ready!")
            
            # Update status
            self.status_var.set("Image generated successfully! You can save it now.")
            
            # Enable save button
            self.save_btn.config(state="normal")
            
            # Store image for saving
            self.current_image = image
            
        except Exception as e:
            self.show_error(f"Error displaying image: {str(e)}")
    
    def save_image(self):
        """Save the generated image"""
        if not hasattr(self, 'current_image'):
            return
        
        try:
            # Create generated_images directory
            os.makedirs("generated_images", exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"generated_images/gui_generated_{timestamp}.png"
            
            # Save image
            self.current_image.save(filename)
            
            messagebox.showinfo("Success", f"Image saved as:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def show_error(self, message):
        """Show error message"""
        # Reset progress bar
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.progress_bar['value'] = 0
        
        self.progress_var.set("Generation failed")
        self.status_var.set("Error occurred during generation")
        messagebox.showerror("Error", message)
    
    def stop_generation(self):
        """Stop the current generation"""
        if self.is_generating:
            self.should_stop = True
            self.progress_var.set("Stopping generation...")
            self.step_var.set("Please wait...")
    
    def show_stopped_message(self):
        """Show message when generation is stopped"""
        self.progress_var.set("Generation stopped by user")
        self.step_var.set("Ready to generate again")
        self.status_var.set("Generation was stopped. You can start a new one.")
    
    def generation_complete(self):
        """Called when generation is complete"""
        self.is_generating = False
        self.should_stop = False
        self.generate_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
    
    def show_parameter_help(self):
        """Show help dialog explaining inference steps and guidance scale"""
        help_window = tk.Toplevel(self.root)
        help_window.title("🎨 Parameter Help - Understanding Your Settings")
        help_window.geometry("700x600")
        help_window.resizable(True, True)
        
        # Center the help window
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Main frame with scrollbar
        main_frame = ttk.Frame(help_window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        help_window.columnconfigure(0, weight=1)
        help_window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎨 Understanding Stable Diffusion Parameters", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        
        # Tab 1: Inference Steps
        steps_frame = ttk.Frame(notebook, padding="15")
        notebook.add(steps_frame, text="🔄 Inference Steps")
        
        steps_text = """🔄 INFERENCE STEPS (Denoising Steps)

What it controls: Image quality and generation speed
How it works: Number of noise reduction iterations

📊 Step Ranges & Results:
┌─────────┬─────────────┬─────────────┬─────────────┐
│ Steps   │ Speed       │ Quality     │ Use Case    │
├─────────┼─────────────┼─────────────┼─────────────┤
│ 5-10    │ ⚡ Very Fast │ 🟡 Basic     │ Quick tests │
│ 15-20   │ 🚀 Fast      │ 🟢 Good      │ Daily use   │
│ 25-30   │ 🐌 Medium    │ 🔵 High      │ Quality     │
│ 35-50   │ 🐌 Slow      │ 🔴 Maximum   │ Final art   │
└─────────┴─────────────┴─────────────┴─────────────┘

💡 Step-by-Step Process:
   Step 1-5:   Basic shapes and composition
   Step 6-15:  Objects and main features emerge
   Step 16-25: Details and textures develop
   Step 26-35: Fine details and refinements
   Step 36-50: Maximum quality (small improvements)

🎯 For your GTX 1650 (4GB VRAM):
   • Recommended: 20-25 steps (balanced quality/speed)
   • Maximum: 30-35 steps (to avoid VRAM issues)
   • Your current setting: 20 steps ✅"""
        
        steps_scroll = scrolledtext.ScrolledText(steps_frame, wrap=tk.WORD, width=80, height=25)
        steps_scroll.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        steps_scroll.insert(tk.END, steps_text)
        steps_scroll.config(state=tk.DISABLED)
        
        # Tab 2: Guidance Scale
        guidance_frame = ttk.Frame(notebook, padding="15")
        notebook.add(guidance_frame, text="🎭 Guidance Scale")
        
        guidance_text = """🎭 GUIDANCE SCALE (CFG Scale)

What it controls: How closely AI follows your prompt
How it works: Balance between creativity and accuracy

📊 Scale Ranges & Results:
┌─────────┬─────────────┬─────────────┬─────────────┐
│ Scale   │ Creativity  │ Accuracy    │ Result      │
├─────────┼─────────────┼─────────────┼─────────────┤
│ 1.0-3.0 │ 🎨 Very High│ 🟡 Low       │ Artistic    │
│ 4.0-6.0 │ 🎨 High      │ 🟢 Medium    │ Creative    │
│ 7.0-9.0 │ 🎨 Medium    │ 🔵 High      │ Balanced    │
│ 10-15   │ 🎨 Low       │ 🔴 Very High │ Accurate    │
│ 16-20   │ 🎨 Very Low  │ 🔴 Maximum   │ Rigid       │
└─────────┴─────────────┴─────────────┴─────────────┘

💡 Real Examples:
   Prompt: 'A cat sitting on a chair'
   Scale 1.0: Might generate abstract art with cat-like elements
   Scale 5.0: Cat-like creature, creative interpretation
   Scale 7.0: Clear cat, sitting, chair visible (recommended) ✅
   Scale 12.0: Exact cat, exact chair, very literal
   Scale 18.0: Photorealistic, rigid, no artistic flair

🎯 For your GTX 1650 (4GB VRAM):
   • Recommended: 7.0-7.5 (good balance)
   • Your current setting: 7.0 ✅"""
        
        guidance_scroll = scrolledtext.ScrolledText(guidance_frame, wrap=tk.WORD, width=80, height=25)
        guidance_scroll.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        guidance_scroll.insert(tk.END, guidance_text)
        guidance_scroll.config(state=tk.DISABLED)
        
        # Tab 3: Quick Reference
        reference_frame = ttk.Frame(notebook, padding="15")
        notebook.add(reference_frame, text="📋 Quick Reference")
        
        reference_text = """📋 QUICK REFERENCE GUIDE

🚀 FAST PREVIEW (Quick testing):
   Steps: 10-15
   Guidance: 6.0-7.0
   Result: Basic quality, fast generation

⚖️  BALANCED (Daily use):
   Steps: 20-25
   Guidance: 7.0-8.0
   Result: Good quality, reasonable speed ✅

🎨 QUALITY (Important images):
   Steps: 30-35
   Guidance: 7.5-8.5
   Result: High quality, slower generation

🔴 MAXIMUM QUALITY (Final artwork):
   Steps: 40-50
   Guidance: 8.0-9.0
   Result: Maximum detail, slow generation

🔧 TROUBLESHOOTING TIPS:

❌ Common Problems & Solutions:
Problem: Images are too blurry
Solution: Increase steps (20→30) and guidance (7.0→8.0)

Problem: Images don't match prompt
Solution: Increase guidance scale (7.0→10.0)

Problem: Generation takes too long
Solution: Decrease steps (30→20) and guidance (8.0→7.0)

Problem: Images are too rigid/artificial
Solution: Decrease guidance scale (10.0→7.0)

🎯 Your Current Settings Are Perfect! ✅
   Steps: 20 (balanced quality/speed)
   Guidance: 7.0 (good prompt adherence)
   Resolution: 512x512 (optimal for 4GB VRAM)"""
        
        reference_scroll = scrolledtext.ScrolledText(reference_frame, wrap=tk.WORD, width=80, height=25)
        reference_scroll.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        reference_scroll.insert(tk.END, reference_text)
        reference_scroll.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ttk.Button(main_frame, text="✅ Got it!", command=help_window.destroy)
        close_btn.grid(row=2, column=0, pady=(10, 0))
        
        # Configure frame weights for proper scrolling
        steps_frame.columnconfigure(0, weight=1)
        steps_frame.rowconfigure(0, weight=1)
        guidance_frame.columnconfigure(0, weight=1)
        guidance_frame.rowconfigure(0, weight=1)
        reference_frame.columnconfigure(0, weight=1)
        reference_frame.rowconfigure(0, weight=1)

def main():
    """Main function"""
    root = tk.Tk()
    app = StableDiffusionGUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_width() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
