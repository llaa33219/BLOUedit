#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlouEdit Style Transfer Module
Provides functionality for applying artistic styles to images and videos.
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path
import shutil
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("style_transfer_module")

# Try importing required dependencies
try:
    import numpy as np
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV package not found. Basic image processing may be limited.")
    OPENCV_AVAILABLE = False

# Try importing style transfer frameworks
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch/torchvision packages not found. Neural style transfer may be limited.")
    TORCH_AVAILABLE = False

# Optional: Try importing advanced style transfer libraries
ADVANCED_TRANSFER_AVAILABLE = False
try:
    # This is a placeholder for a hypothetical deep style transfer library
    # In a real implementation, this would be replaced with actual libraries
    # like PyTorch implementations of AdaIN, Fast Neural Style, etc.
    pass
except ImportError:
    logger.warning("Advanced style transfer packages not found. Using basic methods.")

class StyleTransferEngine:
    """Engine for applying style transfer to images and videos."""
    
    def __init__(self):
        """Initialize the style transfer engine."""
        self.opencv_available = OPENCV_AVAILABLE
        self.torch_available = TORCH_AVAILABLE
        self.advanced_transfer_available = ADVANCED_TRANSFER_AVAILABLE
        
        # For cancellation
        self.processing_lock = threading.Lock()
        self.cancel_requested = False
        
        # Load style presets
        self._available_styles = self._load_default_styles()
        
        # Initialize models cache
        self.models = {}
        
        # Try to initialize PyTorch models if available
        if self.torch_available:
            try:
                self._init_torch_models()
            except Exception as e:
                logger.error(f"Failed to initialize torch models: {str(e)}")
    
    def _init_torch_models(self):
        """Initialize PyTorch models for style transfer."""
        if not self.torch_available:
            return
        
        # Use torch hub to load pre-trained style transfer models
        try:
            # Load a simple style transfer model from torch hub
            # This is just an example - you might want to use a different model
            logger.info("Loading style transfer models from torch hub...")
            
            # Try to load FastNeuralStyle models
            for style in ["mosaic", "candy", "rain_princess", "udnie"]:
                model_key = f"torch_fast_{style}"
                if model_key not in self.models:
                    self.models[model_key] = torch.hub.load('pytorch/vision:v0.10.0', 
                                                          'deeplabv3_resnet50', 
                                                          pretrained=True)
                    self.models[model_key].eval()
            
            logger.info("PyTorch models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load some PyTorch models: {str(e)}")
    
    def get_available_styles(self, category=None):
        """
        Get a list of available style presets.
        
        Args:
            category (str): Optional filter by category
            
        Returns:
            list: List of style dictionaries with id, name, category, etc.
        """
        styles = self._available_styles
        
        # Filter by category if provided
        if category:
            styles = [s for s in styles if s['category'] == category]
        
        return styles
    
    def apply_style(self, params):
        """
        Apply style transfer to image or video.
        
        Args:
            params (dict): Dictionary containing style transfer parameters
                - input_path (str): Path to input image/video
                - output_path (str): Path to output image/video
                - style_id (str): ID of style to apply
                - strength (float): Strength of style application (0.0-1.0)
                - preserve_colors (bool): Whether to preserve original colors
                - process_audio (bool): Whether to process audio (for videos)
                - target_fps (int): Target FPS (0 = same as input)
                - segments (list): List of (start, end) segments in seconds
                - progress_callback (callable): Function to call with progress updates
            
        Returns:
            bool: True if style transfer was successful, False otherwise
        """
        try:
            # Reset cancellation flag
            with self.processing_lock:
                self.cancel_requested = False
            
            # Extract parameters
            input_path = params.get('input_path', '')
            output_path = params.get('output_path', '')
            style_id = params.get('style_id', '')
            strength = float(params.get('strength', 0.75))
            preserve_colors = bool(params.get('preserve_colors', False))
            process_audio = bool(params.get('process_audio', True))
            target_fps = int(params.get('target_fps', 0))
            segments = params.get('segments', [])
            progress_callback = params.get('progress_callback', None)
            
            # Validate parameters
            if not input_path or not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False, "Input file not found"
            
            if not output_path:
                logger.error("No output path specified")
                return False, "No output path specified"
            
            # Determine if input is image or video
            is_video = self._is_video_file(input_path)
            
            # Apply style
            if is_video:
                return self._apply_style_to_video(
                    input_path, output_path, style_id, strength,
                    preserve_colors, process_audio, target_fps,
                    segments, progress_callback
                )
            else:
                return self._apply_style_to_image(
                    input_path, output_path, style_id, strength,
                    preserve_colors, progress_callback
                )
                
        except Exception as e:
            logger.error(f"Style transfer error: {str(e)}")
            return False, f"Style transfer error: {str(e)}"
    
    def _apply_style_to_image(self, input_path, output_path, style_id, strength, 
                             preserve_colors, progress_callback=None):
        """Apply style transfer to an image."""
        if not self.opencv_available:
            return False, "OpenCV not available for image processing"
        
        try:
            # Load image
            logger.info(f"Loading image: {input_path}")
            image = cv2.imread(input_path)
            if image is None:
                return False, f"Failed to load image: {input_path}"
            
            # Find the requested style
            style_info = self._find_style_by_id(style_id)
            if not style_info:
                return False, f"Style not found: {style_id}"
            
            # Prepare style model
            model_type = style_info.get('model_type', 'basic')
            
            # Apply appropriate style transfer based on available methods
            if model_type == 'torch_fast' and self.torch_available:
                # Apply PyTorch fast neural style transfer
                stylized_image = self._apply_torch_fast_style(image, style_id, strength)
            elif model_type == 'opencv_basic' and self.opencv_available:
                # Apply basic OpenCV filters as a fallback
                stylized_image = self._apply_opencv_basic_style(image, style_id, strength)
            else:
                # Use most basic filter as ultimate fallback
                stylized_image = self._apply_basic_filter(image, style_id, strength)
            
            # Color preservation if requested
            if preserve_colors and self.opencv_available:
                stylized_image = self._preserve_original_colors(image, stylized_image)
            
            # Save output image
            logger.info(f"Saving stylized image to: {output_path}")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, stylized_image)
            
            # Report 100% progress
            if progress_callback:
                progress_callback(1.0)
            
            return True, output_path
            
        except Exception as e:
            logger.error(f"Image style transfer error: {str(e)}")
            return False, f"Image style transfer error: {str(e)}"
    
    def _apply_style_to_video(self, input_path, output_path, style_id, strength, 
                             preserve_colors, process_audio, target_fps,
                             segments, progress_callback=None):
        """Apply style transfer to a video."""
        if not self.opencv_available:
            return False, "OpenCV not available for video processing"
        
        try:
            # Open video file
            logger.info(f"Loading video: {input_path}")
            video = cv2.VideoCapture(input_path)
            if not video.isOpened():
                return False, f"Failed to open video: {input_path}"
            
            # Get video properties
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Use target FPS if specified
            output_fps = target_fps if target_fps > 0 else fps
            
            # Find the requested style
            style_info = self._find_style_by_id(style_id)
            if not style_info:
                return False, f"Style not found: {style_id}"
            
            # Prepare temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix="blouedit_style_")
            frames_dir = os.path.join(temp_dir, "frames")
            styled_frames_dir = os.path.join(temp_dir, "styled_frames")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(styled_frames_dir, exist_ok=True)
            
            try:
                # Determine which frames to process based on segments
                frame_indices = []
                if segments:
                    for start_sec, end_sec in segments:
                        start_frame = int(start_sec * fps)
                        end_frame = int(end_sec * fps)
                        frame_indices.extend(range(start_frame, end_frame + 1))
                else:
                    # Process all frames
                    frame_indices = range(total_frames)
                
                # Limit to actual number of frames
                frame_indices = [i for i in frame_indices if i < total_frames]
                num_frames_to_process = len(frame_indices)
                
                # Extract frames
                logger.info(f"Extracting {num_frames_to_process} frames...")
                frame_paths = []
                for i, frame_idx in enumerate(frame_indices):
                    # Check for cancellation
                    with self.processing_lock:
                        if self.cancel_requested:
                            return False, "Operation cancelled"
                    
                    # Seek to frame position
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = video.read()
                    if not ret:
                        continue
                    
                    # Save frame to file
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append((frame_idx, frame_path))
                    
                    # Report progress (extraction phase: 0-20%)
                    if progress_callback:
                        progress = 0.2 * (i / num_frames_to_process)
                        progress_callback(progress)
                
                # Close video file
                video.release()
                
                # Apply style to frames
                logger.info(f"Applying style to {len(frame_paths)} frames...")
                styled_frame_paths = []
                
                for i, (frame_idx, frame_path) in enumerate(frame_paths):
                    # Check for cancellation
                    with self.processing_lock:
                        if self.cancel_requested:
                            return False, "Operation cancelled"
                    
                    # Load frame
                    frame = cv2.imread(frame_path)
                    
                    # Apply style
                    if style_info.get('model_type') == 'torch_fast' and self.torch_available:
                        styled_frame = self._apply_torch_fast_style(frame, style_id, strength)
                    else:
                        styled_frame = self._apply_opencv_basic_style(frame, style_id, strength)
                    
                    # Preserve colors if requested
                    if preserve_colors:
                        styled_frame = self._preserve_original_colors(frame, styled_frame)
                    
                    # Save styled frame
                    styled_frame_path = os.path.join(styled_frames_dir, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(styled_frame_path, styled_frame)
                    styled_frame_paths.append((frame_idx, styled_frame_path))
                    
                    # Report progress (styling phase: 20-80%)
                    if progress_callback:
                        progress = 0.2 + 0.6 * (i / len(frame_paths))
                        progress_callback(progress)
                
                # Create video from styled frames
                logger.info("Creating output video...")
                
                # Sort frames by index
                styled_frame_paths.sort(key=lambda x: x[0])
                
                # Video writer setup
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
                
                # Write styled frames to video
                for i, (frame_idx, styled_frame_path) in enumerate(styled_frame_paths):
                    # Check for cancellation
                    with self.processing_lock:
                        if self.cancel_requested:
                            out.release()
                            return False, "Operation cancelled"
                    
                    # Read styled frame
                    styled_frame = cv2.imread(styled_frame_path)
                    if styled_frame is not None:
                        # Write frame to video
                        out.write(styled_frame)
                    
                    # Report progress (video creation phase: 80-95%)
                    if progress_callback:
                        progress = 0.8 + 0.15 * (i / len(styled_frame_paths))
                        progress_callback(progress)
                
                # Finalize video
                out.release()
                
                # Process audio if requested
                if process_audio:
                    logger.info("Processing audio...")
                    self._process_audio(input_path, output_path)
                    
                    # Report progress (audio processing: 95-100%)
                    if progress_callback:
                        progress_callback(1.0)
                
                return True, output_path
                
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory: {str(e)}")
                
        except Exception as e:
            logger.error(f"Video style transfer error: {str(e)}")
            return False, f"Video style transfer error: {str(e)}"
    
    def _process_audio(self, input_path, output_path):
        """Extract audio from input and add it to the output video."""
        try:
            # Temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.aac', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Extract audio using ffmpeg (requires ffmpeg to be installed)
            extract_cmd = f"ffmpeg -y -i \"{input_path}\" -vn -acodec copy \"{temp_audio_path}\" -hide_banner -loglevel error"
            logger.info(f"Extracting audio: {extract_cmd}")
            ret = os.system(extract_cmd)
            if ret != 0:
                logger.warning("Failed to extract audio, output will have no sound")
                return
            
            # Create temporary file for output with audio
            output_with_audio = output_path + ".temp.mp4"
            
            # Add audio to video
            add_audio_cmd = f"ffmpeg -y -i \"{output_path}\" -i \"{temp_audio_path}\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \"{output_with_audio}\" -hide_banner -loglevel error"
            logger.info(f"Adding audio: {add_audio_cmd}")
            ret = os.system(add_audio_cmd)
            
            # Clean up
            try:
                os.remove(temp_audio_path)
            except:
                pass
            
            if ret == 0:
                # Replace original output with the one containing audio
                os.replace(output_with_audio, output_path)
            else:
                logger.warning("Failed to add audio to output video")
                try:
                    os.remove(output_with_audio)
                except:
                    pass
                
        except Exception as e:
            logger.warning(f"Audio processing error: {str(e)}")
    
    def _apply_torch_fast_style(self, image, style_id, strength):
        """Apply PyTorch-based fast neural style transfer."""
        if not self.torch_available:
            return self._apply_opencv_basic_style(image, style_id, strength)
        
        try:
            # Get corresponding model key
            model_key = f"torch_fast_{style_id.split('_')[-1]}"
            
            # Use basic style if model not available
            if model_key not in self.models:
                return self._apply_opencv_basic_style(image, style_id, strength)
            
            # Preprocess image for PyTorch
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torchvision.transforms.ToTensor()(image_rgb).unsqueeze(0)
            
            # Apply style transfer
            with torch.no_grad():
                output = self.models[model_key](image_tensor)
            
            # Convert output tensor to numpy array
            output_img = output[0].cpu().numpy().transpose(1, 2, 0)
            
            # Scale output to 0-255 range
            output_img = (output_img * 255).astype(np.uint8)
            
            # Convert back to BGR for OpenCV
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Blend with original based on strength
            if strength < 1.0:
                output_img = cv2.addWeighted(image, 1 - strength, output_img, strength, 0)
            
            return output_img
            
        except Exception as e:
            logger.error(f"PyTorch style transfer error: {str(e)}")
            return self._apply_opencv_basic_style(image, style_id, strength)
    
    def _apply_opencv_basic_style(self, image, style_id, strength):
        """Apply basic OpenCV filters as a fallback style transfer method."""
        if not self.opencv_available:
            return image
        
        try:
            # Apply different effects based on style ID
            style_name = style_id.split('_')[-1]
            
            if style_name == 'grayscale':
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                styled = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            elif style_name == 'sepia':
                # Apply sepia filter
                kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
                styled = cv2.transform(image, kernel)
            
            elif style_name == 'negative':
                # Create negative
                styled = 255 - image
            
            elif style_name == 'sketch':
                # Create sketch effect
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                inv_gray = 255 - gray
                blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
                inv_blurred = 255 - blurred
                sketch = cv2.divide(gray, inv_blurred, scale=256.0)
                styled = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
            elif style_name == 'hdr':
                # Create HDR effect
                hdr = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
                styled = cv2.detailEnhance(hdr, sigma_s=12, sigma_r=0.15)
            
            elif style_name == 'cartoon':
                # Create cartoon effect
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(image, 9, 300, 300)
                styled = cv2.bitwise_and(color, color, mask=edges)
            
            else:
                # Default to simple color enhancement
                styled = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
                
            # Blend with original based on strength
            if strength < 1.0:
                styled = cv2.addWeighted(image, 1 - strength, styled, strength, 0)
            
            return styled
            
        except Exception as e:
            logger.error(f"OpenCV style transfer error: {str(e)}")
            return self._apply_basic_filter(image, style_id, strength)
    
    def _apply_basic_filter(self, image, style_id, strength):
        """Apply a very basic filter as ultimate fallback."""
        # Apply a simple brightness/contrast adjustment as fallback
        alpha = 1.0 + (strength * 0.2)  # Contrast
        beta = strength * 10  # Brightness
        
        # Apply simple linear contrast adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Blend with original based on strength
        if strength < 0.8:  # Ensure some effect is visible
            return cv2.addWeighted(image, 1 - strength, adjusted, strength, 0)
        else:
            return adjusted
    
    def _preserve_original_colors(self, original, styled):
        """Preserve the colors of the original image in the styled image."""
        if not self.opencv_available:
            return styled
        
        try:
            # Convert both images to LAB color space
            original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            styled_lab = cv2.cvtColor(styled, cv2.COLOR_BGR2LAB)
            
            # Replace the L channel (luminance) of the original with the L channel of the styled
            original_lab[:,:,0] = styled_lab[:,:,0]
            
            # Convert back to BGR
            color_preserved = cv2.cvtColor(original_lab, cv2.COLOR_LAB2BGR)
            
            return color_preserved
            
        except Exception as e:
            logger.error(f"Color preservation error: {str(e)}")
            return styled
    
    def _is_video_file(self, file_path):
        """Check if a file is a video based on extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        ext = os.path.splitext(file_path.lower())[1]
        return ext in video_extensions
    
    def _find_style_by_id(self, style_id):
        """Find a style preset by its ID."""
        for style in self._available_styles:
            if style['id'] == style_id:
                return style
        return None
    
    def _load_default_styles(self):
        """Load and return a list of default styles."""
        styles = []
        
        # Add painting styles
        styles.extend([
            {
                'id': 'painting_vangogh',
                'name': 'Van Gogh',
                'category': 'painting',
                'preview_path': 'styles/vangogh_preview.jpg',
                'strength_default': 0.7,
                'model_type': 'torch_fast'
            },
            {
                'id': 'painting_monet',
                'name': 'Monet',
                'category': 'painting',
                'preview_path': 'styles/monet_preview.jpg',
                'strength_default': 0.6,
                'model_type': 'torch_fast'
            },
            {
                'id': 'painting_kandinsky',
                'name': 'Kandinsky',
                'category': 'painting',
                'preview_path': 'styles/kandinsky_preview.jpg',
                'strength_default': 0.8,
                'model_type': 'torch_fast'
            }
        ])
        
        # Add photo styles
        styles.extend([
            {
                'id': 'photo_hdr',
                'name': 'HDR',
                'category': 'photo',
                'preview_path': 'styles/hdr_preview.jpg',
                'strength_default': 0.75,
                'model_type': 'opencv_basic'
            },
            {
                'id': 'photo_noir',
                'name': 'Film Noir',
                'category': 'photo',
                'preview_path': 'styles/noir_preview.jpg',
                'strength_default': 0.8,
                'model_type': 'opencv_basic'
            },
            {
                'id': 'photo_vintage',
                'name': 'Vintage',
                'category': 'photo',
                'preview_path': 'styles/vintage_preview.jpg',
                'strength_default': 0.65,
                'model_type': 'opencv_basic'
            }
        ])
        
        # Add abstract styles
        styles.extend([
            {
                'id': 'abstract_cubism',
                'name': 'Cubism',
                'category': 'abstract',
                'preview_path': 'styles/cubism_preview.jpg',
                'strength_default': 0.9,
                'model_type': 'torch_fast'
            },
            {
                'id': 'abstract_mosaic',
                'name': 'Mosaic',
                'category': 'abstract',
                'preview_path': 'styles/mosaic_preview.jpg',
                'strength_default': 0.8,
                'model_type': 'torch_fast'
            }
        ])
        
        # Add cartoon styles
        styles.extend([
            {
                'id': 'cartoon_anime',
                'name': 'Anime',
                'category': 'cartoon',
                'preview_path': 'styles/anime_preview.jpg',
                'strength_default': 0.7,
                'model_type': 'opencv_basic'
            },
            {
                'id': 'cartoon_comic',
                'name': 'Comic Book',
                'category': 'cartoon',
                'preview_path': 'styles/comic_preview.jpg',
                'strength_default': 0.75,
                'model_type': 'opencv_basic'
            }
        ])
        
        # Add cinematic styles
        styles.extend([
            {
                'id': 'cinematic_scifi',
                'name': 'Sci-Fi',
                'category': 'cinematic',
                'preview_path': 'styles/scifi_preview.jpg',
                'strength_default': 0.6,
                'model_type': 'opencv_basic'
            },
            {
                'id': 'cinematic_western',
                'name': 'Western',
                'category': 'cinematic',
                'preview_path': 'styles/western_preview.jpg',
                'strength_default': 0.7,
                'model_type': 'opencv_basic'
            }
        ])
        
        return styles
    
    def cancel(self):
        """Cancel ongoing style transfer process."""
        with self.processing_lock:
            self.cancel_requested = True
        return True


# For testing the module directly
if __name__ == "__main__":
    style_engine = StyleTransferEngine()
    
    # Get available styles
    styles = style_engine.get_available_styles()
    print(f"Found {len(styles)} styles")
    for i, style in enumerate(styles[:5]):  # Show first 5 only
        print(f"{i+1}. {style['name']} ({style['category']})")
    
    # Test style transfer if arguments provided
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        style_id = sys.argv[3] if len(sys.argv) > 3 else "photo_hdr"
        
        if os.path.exists(input_path):
            print(f"Applying style {style_id} to {input_path}...")
            
            # Define progress callback
            def progress_callback(progress):
                print(f"Progress: {progress*100:.1f}%")
            
            # Set parameters
            params = {
                'input_path': input_path,
                'output_path': output_path,
                'style_id': style_id,
                'strength': 0.75,
                'preserve_colors': False,
                'progress_callback': progress_callback
            }
            
            # Apply style
            success, result = style_engine.apply_style(params)
            
            if success:
                print(f"Style transfer completed: {result}")
            else:
                print(f"Style transfer failed: {result}")
        else:
            print(f"Input file not found: {input_path}")
    else:
        print("Usage: python style_transfer_module.py <input_file> <output_file> [style_id]")
        print("Available styles:", end=" ")
        style_names = [f"{s['id']}" for s in styles[:5]]
        print(", ".join(style_names)) 