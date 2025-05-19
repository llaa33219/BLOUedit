#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlouEdit Smart Cutout Module
Provides functionality for extracting objects from images and videos.
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
logger = logging.getLogger("smart_cutout_module")

# Try importing required dependencies
try:
    import numpy as np
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV package not found. Smart cutout will be limited.")
    OPENCV_AVAILABLE = False

# Check for advanced segmentation libraries
try:
    # In a real implementation, you would use actual segmentation libraries
    # like PyTorch-based models (U2Net, DeepLabV3, etc.)
    # This is just a placeholder
    ADVANCED_SEGMENTATION_AVAILABLE = False
    logger.info("Advanced segmentation models not available, using basic methods")
except:
    ADVANCED_SEGMENTATION_AVAILABLE = False

# Check for ffmpeg
try:
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    FFMPEG_AVAILABLE = result.returncode == 0
except:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg not available. Video processing will be limited.")

class SmartCutoutEngine:
    """Engine for extracting objects from images and videos."""
    
    def __init__(self):
        """Initialize the smart cutout engine."""
        self.opencv_available = OPENCV_AVAILABLE
        self.advanced_segmentation_available = ADVANCED_SEGMENTATION_AVAILABLE
        self.ffmpeg_available = FFMPEG_AVAILABLE
        
        # For cancellation
        self.processing_lock = threading.Lock()
        self.cancel_requested = False
        
        # Initialize models
        if self.opencv_available:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize segmentation models."""
        try:
            # Initialize OpenCV-based segmentation
            self.backSub = cv2.createBackgroundSubtractorMOG2()
            
            # Try to load a more advanced model if available
            if self.advanced_segmentation_available:
                # This would be replaced with actual model loading code
                pass
                
            logger.info("Segmentation models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize segmentation models: {str(e)}")
    
    def apply_cutout(self, params):
        """
        Apply smart cutout to an image or video.
        
        Args:
            params (dict): Dictionary containing cutout parameters
                - input_path (str): Path to input image/video
                - output_path (str): Path to output image/video
                - mask_path (str): Path to save the generated mask
                - method (str): Method to use for cutout
                - mask_type (str): Type of mask to generate
                - threshold (float): Threshold for binary masking
                - invert_mask (bool): Whether to invert the mask
                - apply_feathering (bool): Whether to apply feathering
                - feather_amount (float): Amount of feathering to apply
                - add_shadow (bool): Whether to add a drop shadow
                - background_path (str): Path to background image/video
                - process_audio (bool): Whether to process audio
                - segments (list): List of time segments to process
                - foreground_points (list): Points marked as foreground
                - background_points (list): Points marked as background
                - initial_mask_path (str): Path to initial mask
                - progress_callback (callable): Function to call with progress updates
            
        Returns:
            tuple: (success, message)
        """
        if not self.opencv_available:
            return False, "OpenCV not available for image processing"
        
        # Reset cancellation flag
        with self.processing_lock:
            self.cancel_requested = False
        
        try:
            # Extract parameters
            input_path = params.get('input_path', '')
            output_path = params.get('output_path', '')
            mask_path = params.get('mask_path', '')
            method = params.get('method', 'automatic')
            mask_type = params.get('mask_type', 'alpha')
            threshold = float(params.get('threshold', 0.5))
            invert_mask = bool(params.get('invert_mask', False))
            apply_feathering = bool(params.get('apply_feathering', False))
            feather_amount = float(params.get('feather_amount', 2.0))
            add_shadow = bool(params.get('add_shadow', False))
            background_path = params.get('background_path', '')
            process_audio = bool(params.get('process_audio', True))
            segments = params.get('segments', [])
            foreground_points = params.get('foreground_points', [])
            background_points = params.get('background_points', [])
            initial_mask_path = params.get('initial_mask_path', '')
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
            
            # Apply appropriate processing
            if is_video:
                return self._apply_cutout_to_video(
                    input_path, output_path, mask_path, method,
                    mask_type, threshold, invert_mask, apply_feathering,
                    feather_amount, add_shadow, background_path,
                    process_audio, segments, foreground_points,
                    background_points, initial_mask_path, progress_callback
                )
            else:
                return self._apply_cutout_to_image(
                    input_path, output_path, mask_path, method,
                    mask_type, threshold, invert_mask, apply_feathering,
                    feather_amount, add_shadow, background_path,
                    foreground_points, background_points, initial_mask_path,
                    progress_callback
                )
                
        except Exception as e:
            logger.error(f"Smart cutout error: {str(e)}")
            return False, f"Smart cutout error: {str(e)}"
    
    def _apply_cutout_to_image(self, input_path, output_path, mask_path, method,
                             mask_type, threshold, invert_mask, apply_feathering,
                             feather_amount, add_shadow, background_path,
                             foreground_points, background_points, initial_mask_path,
                             progress_callback=None):
        """Apply cutout to an image."""
        try:
            # Load image
            logger.info(f"Loading image: {input_path}")
            image = cv2.imread(input_path)
            if image is None:
                return False, f"Failed to load image: {input_path}"
            
            # Report initial progress
            if progress_callback:
                progress_callback(0.1)
            
            # Generate mask based on selected method
            mask = self._generate_mask_for_image(
                image, method, threshold, foreground_points,
                background_points, initial_mask_path
            )
            
            if mask is None:
                return False, "Failed to generate mask"
            
            # Process mask
            processed_mask = self._process_mask(
                mask, mask_type, threshold, invert_mask,
                apply_feathering, feather_amount
            )
            
            # Save mask if requested
            if mask_path:
                os.makedirs(os.path.dirname(os.path.abspath(mask_path)), exist_ok=True)
                cv2.imwrite(mask_path, processed_mask)
            
            # Report progress
            if progress_callback:
                progress_callback(0.6)
            
            # Apply mask to image
            if background_path and os.path.exists(background_path):
                # Composite with background
                result = self._composite_with_background(
                    image, processed_mask, background_path, add_shadow
                )
            else:
                # Extract object with transparency
                result = self._apply_mask_to_image(
                    image, processed_mask, add_shadow
                )
            
            # Save output image
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, result)
            
            # Report completion
            if progress_callback:
                progress_callback(1.0)
            
            return True, output_path
            
        except Exception as e:
            logger.error(f"Image cutout error: {str(e)}")
            return False, f"Image cutout error: {str(e)}"
    
    def _apply_cutout_to_video(self, input_path, output_path, mask_path, method,
                            mask_type, threshold, invert_mask, apply_feathering,
                            feather_amount, add_shadow, background_path,
                            process_audio, segments, foreground_points,
                            background_points, initial_mask_path, progress_callback=None):
        """Apply cutout to a video."""
        try:
            # For video processing, we'll use ffmpeg if available
            if not self.ffmpeg_available:
                return False, "ffmpeg not available for video processing"
            
            # Create temporary directories
            temp_dir = tempfile.mkdtemp(prefix="blouedit_cutout_")
            frames_dir = os.path.join(temp_dir, "frames")
            masks_dir = os.path.join(temp_dir, "masks")
            output_frames_dir = os.path.join(temp_dir, "output_frames")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(output_frames_dir, exist_ok=True)
            
            try:
                # Extract audio if needed
                audio_path = None
                if process_audio:
                    audio_path = os.path.join(temp_dir, "audio.aac")
                    audio_cmd = ["ffmpeg", "-y", "-i", input_path, 
                                "-vn", "-acodec", "copy", audio_path]
                    subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Get video info
                probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                            "-show_entries", "stream=width,height,r_frame_rate", 
                            "-of", "csv=p=0", input_path]
                
                result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if result.returncode != 0:
                    return False, "Failed to get video information"
                
                info = result.stdout.strip().split(",")
                width, height, fps_str = int(info[0]), int(info[1]), info[2]
                fps = eval(fps_str)
                
                # Extract frames
                extract_cmd = ["ffmpeg", "-y", "-i", input_path, 
                              "-vf", f"fps={fps}",
                              os.path.join(frames_dir, "frame_%05d.png")]
                
                subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Get frame filenames
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    return False, "No frames extracted"
                
                # Process each frame
                for i, frame_file in enumerate(frame_files):
                    # Check for cancellation
                    with self.processing_lock:
                        if self.cancel_requested:
                            return False, "Operation cancelled"
                    
                    # Report progress
                    if progress_callback:
                        progress = 0.1 + 0.7 * (i / total_frames)
                        progress_callback(progress)
                    
                    # Load frame
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    
                    # Generate mask
                    mask = self._generate_mask_for_image(
                        frame, method, threshold, foreground_points,
                        background_points, initial_mask_path
                    )
                    
                    if mask is None:
                        continue
                    
                    # Process mask
                    processed_mask = self._process_mask(
                        mask, mask_type, threshold, invert_mask,
                        apply_feathering, feather_amount
                    )
                    
                    # Save mask if requested
                    if mask_path:
                        mask_frame_path = os.path.join(masks_dir, frame_file)
                        cv2.imwrite(mask_frame_path, processed_mask)
                    
                    # Apply mask to frame
                    if background_path and os.path.exists(background_path):
                        # Composite with background
                        if self._is_video_file(background_path):
                            # If background is video, extract corresponding frame
                            # This is simplified - real implementation would handle timing
                            bg_frame_index = i % total_frames
                            bg_frame_path = os.path.join(temp_dir, f"bg_frame_{bg_frame_index:05d}.png")
                            
                            if not os.path.exists(bg_frame_path):
                                bg_extract_cmd = ["ffmpeg", "-y", "-i", background_path,
                                                "-vf", f"select=eq(n\\,{bg_frame_index})",
                                                "-vframes", "1", bg_frame_path]
                                subprocess.run(bg_extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            if os.path.exists(bg_frame_path):
                                result = self._composite_with_background(
                                    frame, processed_mask, bg_frame_path, add_shadow
                                )
                            else:
                                result = self._apply_mask_to_image(
                                    frame, processed_mask, add_shadow
                                )
                        else:
                            # Static background
                            result = self._composite_with_background(
                                frame, processed_mask, background_path, add_shadow
                            )
                    else:
                        # Extract with transparency
                        result = self._apply_mask_to_image(
                            frame, processed_mask, add_shadow
                        )
                    
                    # Save processed frame
                    output_frame_path = os.path.join(output_frames_dir, frame_file)
                    cv2.imwrite(output_frame_path, result)
                
                # Create video from processed frames
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                if process_audio and os.path.exists(audio_path):
                    # With audio
                    video_cmd = [
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-i", os.path.join(output_frames_dir, "frame_%05d.png"),
                        "-i", audio_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest",
                        output_path
                    ]
                else:
                    # Without audio
                    video_cmd = [
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-i", os.path.join(output_frames_dir, "frame_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        output_path
                    ]
                
                # Run video creation command
                result = subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Check result
                if result.returncode != 0:
                    return False, "Failed to create output video"
                
                # Report completion
                if progress_callback:
                    progress_callback(1.0)
                
                return True, output_path
            
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"Video cutout error: {str(e)}")
            return False, f"Video cutout error: {str(e)}"
    
    def _generate_mask_for_image(self, image, method, threshold, foreground_points,
                               background_points, initial_mask_path):
        """Generate mask for image based on the selected method."""
        try:
            if method == 'automatic':
                return self._generate_automatic_mask(image)
            elif method == 'salient':
                return self._generate_salient_mask(image)
            elif method == 'portrait':
                return self._generate_portrait_mask(image)
            elif method == 'interactive' and (foreground_points or background_points or initial_mask_path):
                return self._generate_interactive_mask(image, foreground_points, background_points, initial_mask_path)
            else:
                # Fallback to automatic
                return self._generate_automatic_mask(image)
                
        except Exception as e:
            logger.error(f"Mask generation error: {str(e)}")
            return None
    
    def _generate_automatic_mask(self, image):
        """Generate mask automatically using GrabCut."""
        try:
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create initial mask
            mask = np.zeros(image.shape[:2], np.uint8)
            
            # Create rectangle for GrabCut
            rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
            
            # Create temporary arrays for GrabCut
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # Convert mask to binary
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Scale to 0-255
            mask_output = mask2 * 255
            
            return mask_output
            
        except Exception as e:
            logger.error(f"Automatic mask generation error: {str(e)}")
            return None
    
    def _generate_salient_mask(self, image):
        """Generate mask based on salient object detection."""
        try:
            # This is a simplified implementation
            # In a real application, you would use a dedicated saliency model
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"Salient mask generation error: {str(e)}")
            return None
    
    def _generate_portrait_mask(self, image):
        """Generate mask for portraits/people."""
        try:
            # This is a simplified implementation
            # In a real application, you would use a dedicated person segmentation model
            
            # Try to detect face using Haar cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Create mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            if len(faces) > 0:
                # For each face, estimate a body region
                for (x, y, w, h) in faces:
                    # Expand rectangle to include estimated body
                    body_x = max(0, x - w//2)
                    body_y = max(0, y - h//4)
                    body_w = min(image.shape[1] - body_x, w * 2)
                    body_h = min(image.shape[0] - body_y, h * 4)
                    
                    # Fill mask
                    mask[body_y:body_y+body_h, body_x:body_x+body_w] = 255
            else:
                # Fallback to automatic mask if no faces detected
                mask = self._generate_automatic_mask(image)
            
            return mask
            
        except Exception as e:
            logger.error(f"Portrait mask generation error: {str(e)}")
            return None
    
    def _generate_interactive_mask(self, image, foreground_points, background_points, initial_mask_path):
        """Generate mask based on user-provided points or initial mask."""
        try:
            # Start with initial mask if provided
            if initial_mask_path and os.path.exists(initial_mask_path):
                mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape[:2] != image.shape[:2]:
                    # Create empty mask if loading fails or size mismatch
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                # Create empty mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
            # Mark foreground and background regions
            if foreground_points or background_points:
                # Create GrabCut mask
                grabcut_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                grabcut_mask[:] = cv2.GC_PR_BGD  # Initialize with probable background
                
                # Mark foreground points
                for x, y in foreground_points:
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        cv2.circle(grabcut_mask, (x, y), 5, cv2.GC_FGD, -1)
                
                # Mark background points
                for x, y in background_points:
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        cv2.circle(grabcut_mask, (x, y), 5, cv2.GC_BGD, -1)
                
                # Initialize from initial mask if available
                if np.any(mask > 0):
                    grabcut_mask[mask > 127] = cv2.GC_PR_FGD
                
                # Create temporary arrays for GrabCut
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                
                # Apply GrabCut
                cv2.grabCut(image, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                
                # Convert mask to binary
                mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 255).astype('uint8')
            
            return mask
            
        except Exception as e:
            logger.error(f"Interactive mask generation error: {str(e)}")
            return None
    
    def _process_mask(self, mask, mask_type, threshold, invert_mask, apply_feathering, feather_amount):
        """Process the generated mask according to parameters."""
        try:
            # Ensure mask is grayscale
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold if binary mask requested
            if mask_type == 'binary':
                threshold_value = int(threshold * 255)
                _, processed_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                processed_mask = mask.copy()
            
            # Apply feathering if requested
            if apply_feathering and feather_amount > 0:
                kernel_size = max(1, int(feather_amount * 2))
                # Make kernel size odd
                if kernel_size % 2 == 0:
                    kernel_size += 1
                processed_mask = cv2.GaussianBlur(processed_mask, (kernel_size, kernel_size), 0)
            
            # Invert mask if requested
            if invert_mask:
                processed_mask = cv2.bitwise_not(processed_mask)
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Mask processing error: {str(e)}")
            return mask
    
    def _apply_mask_to_image(self, image, mask, add_shadow):
        """Apply mask to image to extract object with transparency."""
        try:
            # Create output image with alpha channel
            h, w = image.shape[:2]
            result = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Copy RGB channels
            result[:, :, :3] = image
            
            # Set alpha channel from mask
            result[:, :, 3] = mask
            
            # Add shadow if requested
            if add_shadow:
                # Create shadow mask
                shadow_mask = np.copy(mask)
                kernel = np.ones((15, 15), np.uint8)
                shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)
                shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
                
                # Apply shadow where alpha > 0 but original mask = 0
                shadow_region = (shadow_mask > 0) & (mask == 0)
                result[shadow_region, 3] = shadow_mask[shadow_region] // 3  # Reduced opacity for shadow
            
            return result
            
        except Exception as e:
            logger.error(f"Mask application error: {str(e)}")
            return image
    
    def _composite_with_background(self, image, mask, background_path, add_shadow):
        """Composite foreground with new background."""
        try:
            # Load background image
            background = cv2.imread(background_path)
            if background is None:
                # If background can't be loaded, just apply mask
                return self._apply_mask_to_image(image, mask, add_shadow)
            
            # Resize background to match foreground
            h, w = image.shape[:2]
            background = cv2.resize(background, (w, h))
            
            # Create mask for alpha blending
            normalized_mask = mask.astype(float) / 255.0
            
            # Add shadow if requested
            if add_shadow:
                # Create shadow mask
                shadow_mask = np.copy(mask)
                kernel = np.ones((15, 15), np.uint8)
                shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)
                shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
                
                # Apply shadow where shadow_mask > 0 but original mask = 0
                shadow_region = (shadow_mask > 0) & (mask == 0)
                
                # Darken background in shadow region
                shadow_factor = 0.7  # 30% darker
                background[shadow_region] = background[shadow_region] * shadow_factor
            
            # Expand dimensions for broadcasting
            alpha = np.expand_dims(normalized_mask, axis=2)
            
            # Blend foreground and background
            result = (image * alpha + background * (1 - alpha)).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Background composition error: {str(e)}")
            return self._apply_mask_to_image(image, mask, add_shadow)
    
    def _is_video_file(self, file_path):
        """Check if a file is a video based on extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        ext = os.path.splitext(file_path.lower())[1]
        return ext in video_extensions
    
    def cancel(self):
        """Cancel ongoing cutout operation."""
        with self.processing_lock:
            self.cancel_requested = True
        return True


# For testing the module directly
if __name__ == "__main__":
    engine = SmartCutoutEngine()
    
    # Test cutout if arguments provided
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) > 3 else "automatic"
        
        if os.path.exists(input_path):
            print(f"Applying {method} cutout to {input_path}...")
            
            # Define progress callback
            def progress_callback(progress):
                print(f"Progress: {progress*100:.1f}%")
            
            # Set parameters
            params = {
                'input_path': input_path,
                'output_path': output_path,
                'method': method,
                'progress_callback': progress_callback
            }
            
            # Apply cutout
            success, result = engine.apply_cutout(params)
            
            if success:
                print(f"Cutout completed: {result}")
            else:
                print(f"Cutout failed: {result}")
        else:
            print(f"Input file not found: {input_path}")
    else:
        print("Usage: python smart_cutout_module.py <input_file> <output_file> [method]")
        print("Available methods: automatic, salient, portrait, interactive") 