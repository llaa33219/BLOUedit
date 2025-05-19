#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlouEdit AI Image Generator Module
Provides functionality for AI-based image generation using Stable Diffusion, DALL-E, and Midjourney-compatible models.
"""

import os
import sys
import logging
import tempfile
import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("image_generator")

# Try importing the required dependencies
try:
    import torch
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    logger.warning("diffusers package not found. Stable Diffusion functionality will be limited.")
    DIFFUSERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("openai package not found. DALL-E functionality will be unavailable.")
    OPENAI_AVAILABLE = False

class ImageGenerator:
    """Class for generating images using various AI models."""
    
    def __init__(self):
        """Initialize the image generator."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.sd_model = None
        self.dall_e_available = OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY") is not None
        self.current_model = None
    
    def load_model(self, model_name):
        """
        Load the specified AI model.
        
        Args:
            model_name (str): Name of the model to load ('stable-diffusion', 'dall-e', 'midjourney')
            
        Returns:
            bool: True if the model was loaded successfully, False otherwise
        """
        try:
            if model_name == "stable-diffusion":
                if not DIFFUSERS_AVAILABLE:
                    logger.error("Cannot load Stable Diffusion: diffusers package not available")
                    return False
                
                if self.sd_model is None:
                    logger.info("Loading Stable Diffusion model...")
                    # Use a smaller model for faster inference
                    self.sd_model = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    self.sd_model.to(self.device)
                    # Enable memory-efficient attention for lower VRAM usage
                    if hasattr(self.sd_model, "enable_attention_slicing"):
                        self.sd_model.enable_attention_slicing()
                
                self.current_model = "stable-diffusion"
                return True
                
            elif model_name == "dall-e":
                if not self.dall_e_available:
                    logger.error("Cannot use DALL-E: OpenAI API key not set or package not available")
                    return False
                
                self.current_model = "dall-e"
                return True
                
            elif model_name == "midjourney":
                logger.warning("Midjourney API not directly available. Using Stable Diffusion with stylization.")
                # Load Stable Diffusion as fallback
                success = self.load_model("stable-diffusion")
                if success:
                    self.current_model = "midjourney"
                return success
            
            else:
                logger.error(f"Unknown model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def generate_image(self, args, output_path):
        """
        Generate an image based on the provided parameters.
        
        Args:
            args (tuple): Tuple containing (prompt, negative_prompt, width, height, steps, guidance_scale)
            output_path (str): Path where the generated image should be saved
            
        Returns:
            bool: True if generation was successful, False otherwise
        """
        try:
            prompt, negative_prompt, width, height, steps, guidance_scale = args
            
            logger.info(f"Generating image with prompt: {prompt}")
            logger.info(f"Using model: {self.current_model}")
            
            if self.current_model == "stable-diffusion":
                return self._generate_with_stable_diffusion(
                    prompt, negative_prompt, width, height, steps, guidance_scale, output_path
                )
                
            elif self.current_model == "dall-e":
                return self._generate_with_dalle(prompt, width, height, output_path)
                
            elif self.current_model == "midjourney":
                # Add a style modifier for Midjourney-like results
                enhanced_prompt = f"{prompt}, trending on artstation, highly detailed, sharp focus, dramatic lighting, vibrant colors, masterpiece"
                return self._generate_with_stable_diffusion(
                    enhanced_prompt, negative_prompt, width, height, steps, guidance_scale * 1.2, output_path
                )
            
            else:
                logger.error("No model loaded. Call load_model() first.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            return False
    
    def _generate_with_stable_diffusion(self, prompt, negative_prompt, width, height, steps, guidance_scale, output_path):
        """Generate image with Stable Diffusion."""
        if self.sd_model is None:
            logger.error("Stable Diffusion model not loaded")
            return False
        
        # Ensure width and height are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        with torch.no_grad():
            image = self.sd_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return True
    
    def _generate_with_dalle(self, prompt, width, height, output_path):
        """Generate image with DALL-E."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not available")
            return False
            
        # Map dimensions to DALL-E supported sizes
        size_map = {
            (512, 512): "512x512",
            (768, 768): "512x512",  # Will be upscaled later
            (1024, 1024): "1024x1024",
            (512, 768): "512x512",  # Will be resized later
            (768, 512): "512x512",  # Will be resized later
        }
        
        size = size_map.get((width, height), "512x512")
        
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=size
            )
            
            image_url = response['data'][0]['url']
            
            # Download the image
            import requests
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # Resize if needed
                if (width, height) != (512, 512) and (width, height) != (1024, 1024):
                    img = Image.open(output_path)
                    img = img.resize((width, height), Image.LANCZOS)
                    img.save(output_path)
                
                return True
            else:
                logger.error(f"Failed to download image: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"DALL-E generation error: {str(e)}")
            return False
    
    def generate_thumbnail(self, video_path, prompt, output_path):
        """
        Generate a thumbnail for a video with AI enhancement.
        
        Args:
            video_path (str): Path to the video file
            prompt (str): Prompt to guide the image generation
            output_path (str): Path where the generated thumbnail should be saved
            
        Returns:
            bool: True if generation was successful, False otherwise
        """
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Extract a frame from the video
            cap = cv2.VideoCapture(video_path)
            
            # Get video length in frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.error(f"Cannot determine video length: {video_path}")
                return False
            
            # Extract a frame from around 1/3 of the video
            target_frame = total_frames // 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to extract frame from video: {video_path}")
                cap.release()
                return False
            
            # Save the frame to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_frame_path = tmp.name
                cv2.imwrite(temp_frame_path, frame)
            
            cap.release()
            
            # Make sure we have a stable diffusion model loaded
            if not self.current_model or self.sd_model is None:
                self.load_model("stable-diffusion")
            
            # Use img2img with the extracted frame as initialization
            # This is a simplified version - in a real implementation we would use img2img pipeline
            init_image = Image.open(temp_frame_path).convert("RGB")
            width, height = init_image.size
            
            # For now, just generate a new image with the prompt
            # In a real implementation, we would use the frame as a starting point
            success = self._generate_with_stable_diffusion(
                prompt, "low quality, blurry", width, height, 50, 7.5, output_path
            )
            
            # Clean up temporary file
            os.unlink(temp_frame_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {str(e)}")
            return False

# For testing the module directly
if __name__ == "__main__":
    generator = ImageGenerator()
    
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "a beautiful landscape with mountains and a lake, digital art"
    
    success = generator.load_model("stable-diffusion")
    if not success:
        print("Failed to load model")
        sys.exit(1)
    
    output_path = "test_image.png"
    success = generator.generate_image(
        (prompt, "low quality, blurry", 512, 512, 30, 7.5),
        output_path
    )
    
    if success:
        print(f"Image generated successfully at {output_path}")
    else:
        print("Failed to generate image") 