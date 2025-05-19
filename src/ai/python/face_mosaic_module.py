#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlouEdit Face Mosaic Module
Provides functionality for detecting and blurring faces in images and videos.
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
logger = logging.getLogger("face_mosaic_module")

# Try importing required dependencies
try:
    import numpy as np
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV package not found. Face detection will be limited.")
    OPENCV_AVAILABLE = False

# Try importing face detection models
try:
    # Check if face detection models are available
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(FACE_CASCADE_PATH):
        FACE_DETECTION_AVAILABLE = True
    else:
        logger.warning(f"Face cascade file not found at {FACE_CASCADE_PATH}")
        FACE_DETECTION_AVAILABLE = False
except:
    FACE_DETECTION_AVAILABLE = False
    logger.warning("Face detection models not available.")

# Check for ffmpeg
try:
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    FFMPEG_AVAILABLE = result.returncode == 0
except:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg not available. Video processing will be limited.")

class FaceMosaicEngine:
    """Engine for detecting and blurring faces in images and videos."""
    
    def __init__(self):
        """Initialize the face mosaic engine."""
        self.opencv_available = OPENCV_AVAILABLE
        self.face_detection_available = FACE_DETECTION_AVAILABLE
        self.ffmpeg_available = FFMPEG_AVAILABLE
        
        # Initialize face detection
        if self.face_detection_available:
            self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
            
            # Try to load DNN face detector if available
            try:
                self.net = cv2.dnn.readNetFromCaffe(
                    "src/ai/python/models/deploy.prototxt",
                    "src/ai/python/models/res10_300x300_ssd_iter_140000.caffemodel"
                )
                self.use_dnn = True
                logger.info("Using DNN face detector")
            except:
                self.use_dnn = False
                logger.info("Using Haar cascade face detector")
        
        # For cancellation
        self.processing_lock = threading.Lock()
        self.cancel_requested = False
    
    def detect_faces(self, params):
        """
        Detect faces in an image or video.
        
        Args:
            params (dict): Dictionary containing parameters
                - input_path (str): Path to input image/video
                - detection_threshold (float): Detection confidence threshold
                - progress_callback (callable): Function to call with progress updates
        
        Returns:
            tuple: (success, faces_list)
        """
        if not self.opencv_available or not self.face_detection_available:
            return False, []
        
        # Reset cancellation flag
        with self.processing_lock:
            self.cancel_requested = False
        
        try:
            # Extract parameters
            input_path = params.get('input_path', '')
            detection_threshold = float(params.get('detection_threshold', 0.5))
            progress_callback = params.get('progress_callback', None)
            
            # Validate parameters
            if not input_path or not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False, []
            
            # Determine if input is image or video
            is_video = self._is_video_file(input_path)
            
            if is_video:
                return self._detect_faces_in_video(input_path, detection_threshold, progress_callback)
            else:
                return self._detect_faces_in_image(input_path, detection_threshold, progress_callback)
                
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return False, []
    
    def _detect_faces_in_image(self, image_path, detection_threshold, progress_callback=None):
        """Detect faces in an image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return False, []
            
            # Report progress at start
            if progress_callback:
                progress_callback(0.1)
            
            # Detect faces using DNN if available
            faces = []
            if self.use_dnn:
                # Get dimensions
                (h, w) = image.shape[:2]
                
                # Create blob from image
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0)
                )
                
                # Pass blob through network
                self.net.setInput(blob)
                detections = self.net.forward()
                
                # Process detections
                for i in range(0, detections.shape[2]):
                    # Extract confidence
                    confidence = detections[0, 0, i, 2]
                    
                    # Filter by threshold
                    if confidence > detection_threshold:
                        # Compute bounding box
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Add to faces list
                        faces.append({
                            'x': startX,
                            'y': startY,
                            'width': endX - startX,
                            'height': endY - startY,
                            'confidence': float(confidence),
                            'tracking_id': -1
                        })
            else:
                # Use Haar cascade
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                face_rects = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Process detections
                for (x, y, w, h) in face_rects:
                    faces.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': 1.0,  # Haar cascade doesn't provide confidence
                        'tracking_id': -1
                    })
            
            # Report completion
            if progress_callback:
                progress_callback(1.0)
            
            return True, faces
            
        except Exception as e:
            logger.error(f"Image face detection error: {str(e)}")
            return False, []
    
    def _detect_faces_in_video(self, video_path, detection_threshold, progress_callback=None):
        """Detect faces in a video (samples frames)."""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                cap.release()
                return False, []
            
            # Calculate sampling rate (check 1 frame per second)
            sample_interval = int(fps)
            
            # Detect faces in sample frames
            faces = []
            frame_count = 0
            
            while True:
                # Check for cancellation
                with self.processing_lock:
                    if self.cancel_requested:
                        cap.release()
                        return False, []
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only sample frames
                if frame_count % sample_interval == 0:
                    # Detect faces in frame
                    faces_in_frame = []
                    
                    # Use DNN if available
                    if self.use_dnn:
                        (h, w) = frame.shape[:2]
                        blob = cv2.dnn.blobFromImage(
                            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                            (104.0, 177.0, 123.0)
                        )
                        
                        self.net.setInput(blob)
                        detections = self.net.forward()
                        
                        for i in range(0, detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            
                            if confidence > detection_threshold:
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")
                                
                                faces_in_frame.append({
                                    'x': startX,
                                    'y': startY,
                                    'width': endX - startX,
                                    'height': endY - startY,
                                    'confidence': float(confidence),
                                    'tracking_id': -1,
                                    'frame': frame_count,
                                    'time': frame_count / fps
                                })
                    else:
                        # Use Haar cascade
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        face_rects = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5,
                            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        for (x, y, w, h) in face_rects:
                            faces_in_frame.append({
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'confidence': 1.0,
                                'tracking_id': -1,
                                'frame': frame_count,
                                'time': frame_count / fps
                            })
                    
                    # Add faces from this frame
                    faces.extend(faces_in_frame)
                    
                    # Report progress
                    if progress_callback and total_frames > 0:
                        progress = min(0.95, frame_count / total_frames)
                        progress_callback(progress)
                
                frame_count += 1
            
            # Release video
            cap.release()
            
            # Final progress
            if progress_callback:
                progress_callback(1.0)
            
            return True, faces
            
        except Exception as e:
            logger.error(f"Video face detection error: {str(e)}")
            return False, []
    
    def apply_mosaic(self, params):
        """
        Apply mosaic to faces in an image or video.
        
        Args:
            params (dict): Dictionary containing parameters
                - input_path (str): Path to input image/video
                - output_path (str): Path to output image/video
                - mosaic_type (str): Type of mosaic effect
                - effect_intensity (float): Intensity of effect
                - track_faces (bool): Whether to track faces across frames
                - process_audio (bool): Whether to process audio
                - detect_only (bool): Whether to only detect faces
                - custom_image_path (str): Path to custom overlay image
                - emoji_type (str): Type of emoji to use
                - detection_threshold (float): Detection confidence threshold
                - auto_expand_rect (bool): Whether to expand rectangles
                - expansion_factor (float): Factor to expand rectangles by
                - segments (list): Time segments to process
                - custom_rects (list): Custom rectangles to blur
                - progress_callback (callable): Function to call with progress updates
        
        Returns:
            tuple: (success, message)
        """
        if not self.opencv_available:
            return False, "OpenCV not available for face mosaic"
        
        # Reset cancellation flag
        with self.processing_lock:
            self.cancel_requested = False
        
        try:
            # Extract parameters
            input_path = params.get('input_path', '')
            output_path = params.get('output_path', '')
            mosaic_type = params.get('mosaic_type', 'blur')
            effect_intensity = float(params.get('effect_intensity', 15.0))
            track_faces = bool(params.get('track_faces', True))
            process_audio = bool(params.get('process_audio', True))
            detect_only = bool(params.get('detect_only', False))
            custom_image_path = params.get('custom_image_path', '')
            emoji_type = params.get('emoji_type', '🙂')
            detection_threshold = float(params.get('detection_threshold', 0.5))
            auto_expand_rect = bool(params.get('auto_expand_rect', True))
            expansion_factor = float(params.get('expansion_factor', 1.2))
            segments = params.get('segments', [])
            custom_rects = params.get('custom_rects', [])
            progress_callback = params.get('progress_callback', None)
            
            # Validate parameters
            if not input_path or not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False, "Input file not found"
            
            if not output_path:
                logger.error("No output path specified")
                return False, "No output path specified"
            
            # For custom image type, verify image exists
            if mosaic_type == 'custom_image' and (not custom_image_path or not os.path.exists(custom_image_path)):
                logger.error(f"Custom image not found: {custom_image_path}")
                return False, "Custom image not found"
            
            # Determine if input is image or video
            is_video = self._is_video_file(input_path)
            
            # Load custom overlay if needed
            custom_overlay = None
            if mosaic_type == 'custom_image' and os.path.exists(custom_image_path):
                custom_overlay = cv2.imread(custom_image_path, cv2.IMREAD_UNCHANGED)
                if custom_overlay is None:
                    logger.error(f"Failed to load custom image: {custom_image_path}")
                    return False, "Failed to load custom image"
            
            # If detect only mode, just return detected faces
            if detect_only:
                success, faces = self.detect_faces({
                    'input_path': input_path,
                    'detection_threshold': detection_threshold,
                    'progress_callback': progress_callback
                })
                
                if success:
                    if is_video:
                        message = f"Detected {len(faces)} faces in {len(set([f.get('frame', 0) for f in faces]))} frames"
                    else:
                        message = f"Detected {len(faces)} faces"
                else:
                    message = "Face detection failed"
                
                return success, message
            
            # Apply mosaic
            if is_video:
                if self.ffmpeg_available:
                    return self._apply_mosaic_to_video_ffmpeg(
                        input_path, output_path, mosaic_type, effect_intensity,
                        track_faces, process_audio, segments, custom_rects,
                        detection_threshold, auto_expand_rect, expansion_factor,
                        custom_overlay, emoji_type, progress_callback
                    )
                else:
                    return self._apply_mosaic_to_video_opencv(
                        input_path, output_path, mosaic_type, effect_intensity,
                        track_faces, process_audio, segments, custom_rects,
                        detection_threshold, auto_expand_rect, expansion_factor,
                        custom_overlay, emoji_type, progress_callback
                    )
            else:
                return self._apply_mosaic_to_image(
                    input_path, output_path, mosaic_type, effect_intensity,
                    custom_rects, detection_threshold, auto_expand_rect, expansion_factor,
                    custom_overlay, emoji_type, progress_callback
                )
                
        except Exception as e:
            logger.error(f"Face mosaic error: {str(e)}")
            return False, f"Face mosaic error: {str(e)}"
    
    def _apply_mosaic_to_image(self, input_path, output_path, mosaic_type, effect_intensity,
                            custom_rects, detection_threshold, auto_expand_rect, expansion_factor,
                            custom_overlay, emoji_type, progress_callback=None):
        """Apply mosaic to faces in an image."""
        try:
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                return False, f"Failed to load image: {input_path}"
            
            # Report initial progress
            if progress_callback:
                progress_callback(0.1)
            
            # Detect faces if no custom rects provided
            face_rects = []
            if not custom_rects:
                success, faces = self._detect_faces_in_image(
                    input_path, detection_threshold
                )
                
                if success:
                    face_rects = [{
                        'x': face['x'],
                        'y': face['y'],
                        'width': face['width'],
                        'height': face['height']
                    } for face in faces]
            else:
                face_rects = [{
                    'x': rect['x'],
                    'y': rect['y'],
                    'width': rect['width'],
                    'height': rect['height']
                } for rect in custom_rects]
            
            # Report progress after detection
            if progress_callback:
                progress_callback(0.5)
            
            # Apply mosaic to each face
            for face in face_rects:
                x, y = face['x'], face['y']
                width, height = face['width'], face['height']
                
                # Expand rectangle if needed
                if auto_expand_rect and expansion_factor > 1.0:
                    center_x = x + width // 2
                    center_y = y + height // 2
                    new_width = int(width * expansion_factor)
                    new_height = int(height * expansion_factor)
                    x = max(0, center_x - new_width // 2)
                    y = max(0, center_y - new_height // 2)
                    width = min(image.shape[1] - x, new_width)
                    height = min(image.shape[0] - y, new_height)
                
                # Get face region
                face_roi = image[y:y+height, x:x+width]
                
                if face_roi.size == 0:
                    continue
                
                # Apply appropriate effect
                if mosaic_type == 'blur':
                    # Apply Gaussian blur
                    blur_level = max(1, int(effect_intensity))
                    # Make blur level odd
                    if blur_level % 2 == 0:
                        blur_level += 1
                    blurred = cv2.GaussianBlur(face_roi, (blur_level, blur_level), 0)
                    image[y:y+height, x:x+width] = blurred
                
                elif mosaic_type == 'pixelate':
                    # Apply pixelation
                    pixel_size = max(1, int(effect_intensity))
                    
                    # Resize down and back up to create pixelation effect
                    h, w = face_roi.shape[:2]
                    temp = cv2.resize(face_roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                                    interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    image[y:y+height, x:x+width] = pixelated
                
                elif mosaic_type == 'black':
                    # Apply black rectangle
                    image[y:y+height, x:x+width] = 0
                
                elif mosaic_type == 'emoji':
                    # Create emoji as text on image
                    # This is a simple implementation - a real one would use proper emoji images
                    emoji_image = np.zeros((height, width, 3), dtype=np.uint8)
                    emoji_image[:] = (255, 255, 255)  # White background
                    
                    # Place text in center
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = min(width, height) / 100.0
                    text_size = cv2.getTextSize(emoji_type, font, font_scale, 2)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    
                    cv2.putText(emoji_image, emoji_type, (text_x, text_y),
                               font, font_scale, (0, 0, 0), 2)
                    
                    image[y:y+height, x:x+width] = emoji_image
                
                elif mosaic_type == 'custom_image' and custom_overlay is not None:
                    # Resize custom image to fit face
                    overlay_resized = cv2.resize(custom_overlay, (width, height))
                    
                    # If has alpha channel, use it for blending
                    if overlay_resized.shape[2] == 4:
                        alpha = overlay_resized[:, :, 3] / 255.0
                        alpha = np.expand_dims(alpha, axis=2)
                        
                        # Get RGB channels
                        overlay_rgb = overlay_resized[:, :, :3]
                        
                        # Blend based on alpha
                        blended = (overlay_rgb * alpha) + (face_roi * (1.0 - alpha))
                        image[y:y+height, x:x+width] = blended.astype(np.uint8)
                    else:
                        # Just copy the overlay
                        image[y:y+height, x:x+width] = overlay_resized
            
            # Save output image
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, image)
            
            # Report completion
            if progress_callback:
                progress_callback(1.0)
            
            return True, output_path
            
        except Exception as e:
            logger.error(f"Image mosaic error: {str(e)}")
            return False, f"Image mosaic error: {str(e)}"
    
    def _apply_mosaic_to_video_opencv(self, input_path, output_path, mosaic_type, effect_intensity,
                                   track_faces, process_audio, segments, custom_rects,
                                   detection_threshold, auto_expand_rect, expansion_factor,
                                   custom_overlay, emoji_type, progress_callback=None):
        """Apply mosaic to faces in a video using OpenCV."""
        try:
            # This is a simplified placeholder - a real implementation would handle
            # extracting frames, processing each, and recomposing the video
            return False, "OpenCV video mosaic not implemented. Please install ffmpeg."
            
        except Exception as e:
            logger.error(f"Video mosaic error: {str(e)}")
            return False, f"Video mosaic error: {str(e)}"
    
    def _apply_mosaic_to_video_ffmpeg(self, input_path, output_path, mosaic_type, effect_intensity,
                                   track_faces, process_audio, segments, custom_rects,
                                   detection_threshold, auto_expand_rect, expansion_factor,
                                   custom_overlay, emoji_type, progress_callback=None):
        """Apply mosaic to faces in a video using ffmpeg and OpenCV."""
        try:
            # Create temporary directories
            temp_dir = tempfile.mkdtemp(prefix="blouedit_mosaic_")
            frames_dir = os.path.join(temp_dir, "frames")
            processed_dir = os.path.join(temp_dir, "processed")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Extract audio if needed
            audio_path = None
            if process_audio:
                audio_path = os.path.join(temp_dir, "audio.aac")
                audio_cmd = [
                    "ffmpeg", "-y", "-i", input_path, 
                    "-vn", "-acodec", "copy", audio_path
                ]
                subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Get video info
            probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                        "-show_entries", "stream=width,height,r_frame_rate", 
                        "-of", "csv=p=0", input_path]
            
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode != 0:
                return False, "Failed to get video information"
            
            info = result.stdout.strip().split(",")
            if len(info) != 3:
                return False, "Failed to parse video information"
            
            width, height, fps_str = int(info[0]), int(info[1]), info[2]
            fps = eval(fps_str)
            
            # Extract frames
            frames_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"fps={fps}",
                os.path.join(frames_dir, "frame_%05d.png")
            ]
            
            extract_process = subprocess.Popen(
                frames_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            )
            
            # Wait for extraction to complete
            extract_process.wait()
            
            if extract_process.returncode != 0:
                return False, "Failed to extract frames"
            
            # Get frame filenames
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
            total_frames = len(frame_files)
            
            if total_frames == 0:
                return False, "No frames extracted"
            
            # Detect faces if no custom rects
            faces_by_frame = {}
            if not custom_rects:
                # Detect faces in frames
                success, faces = self.detect_faces({
                    'input_path': input_path,
                    'detection_threshold': detection_threshold
                })
                
                if success:
                    # Organize by frame number
                    for face in faces:
                        frame_num = face.get('frame', 0)
                        if frame_num not in faces_by_frame:
                            faces_by_frame[frame_num] = []
                        faces_by_frame[frame_num].append(face)
            else:
                # Use custom rects for all frames
                for i in range(total_frames):
                    faces_by_frame[i] = custom_rects
            
            # Process each frame
            for i, frame_file in enumerate(frame_files):
                # Check for cancellation
                with self.processing_lock:
                    if self.cancel_requested:
                        shutil.rmtree(temp_dir)
                        return False, "Operation cancelled"
                
                # Report progress
                if progress_callback:
                    progress = min(0.9, i / total_frames)
                    progress_callback(progress)
                
                # Load frame
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                # Get faces for this frame
                current_faces = faces_by_frame.get(i, [])
                
                # Apply mosaic to each face
                for face in current_faces:
                    x, y = face.get('x', 0), face.get('y', 0)
                    w, h = face.get('width', 0), face.get('height', 0)
                    
                    # Skip invalid faces
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Expand rectangle if needed
                    if auto_expand_rect and expansion_factor > 1.0:
                        center_x = x + w // 2
                        center_y = y + h // 2
                        new_w = int(w * expansion_factor)
                        new_h = int(h * expansion_factor)
                        x = max(0, center_x - new_w // 2)
                        y = max(0, center_y - new_h // 2)
                        w = min(frame.shape[1] - x, new_w)
                        h = min(frame.shape[0] - y, new_h)
                    
                    # Get face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size == 0:
                        continue
                    
                    # Apply appropriate effect
                    if mosaic_type == 'blur':
                        blur_level = max(1, int(effect_intensity))
                        # Make blur level odd
                        if blur_level % 2 == 0:
                            blur_level += 1
                        blurred = cv2.GaussianBlur(face_roi, (blur_level, blur_level), 0)
                        frame[y:y+h, x:x+w] = blurred
                    
                    elif mosaic_type == 'pixelate':
                        pixel_size = max(1, int(effect_intensity))
                        
                        # Resize down and back up to create pixelation effect
                        h_roi, w_roi = face_roi.shape[:2]
                        temp = cv2.resize(face_roi, (max(1, w_roi // pixel_size), max(1, h_roi // pixel_size)),
                                        interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                        
                        frame[y:y+h, x:x+w] = pixelated
                    
                    elif mosaic_type == 'black':
                        frame[y:y+h, x:x+w] = 0
                    
                    elif mosaic_type == 'emoji':
                        emoji_image = np.zeros((h, w, 3), dtype=np.uint8)
                        emoji_image[:] = (255, 255, 255)  # White background
                        
                        # Place text in center
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = min(w, h) / 100.0
                        text_size = cv2.getTextSize(emoji_type, font, font_scale, 2)[0]
                        text_x = (w - text_size[0]) // 2
                        text_y = (h + text_size[1]) // 2
                        
                        cv2.putText(emoji_image, emoji_type, (text_x, text_y),
                                   font, font_scale, (0, 0, 0), 2)
                        
                        frame[y:y+h, x:x+w] = emoji_image
                    
                    elif mosaic_type == 'custom_image' and custom_overlay is not None:
                        # Resize custom image to fit face
                        overlay_resized = cv2.resize(custom_overlay, (w, h))
                        
                        # If has alpha channel, use it for blending
                        if overlay_resized.shape[2] == 4:
                            alpha = overlay_resized[:, :, 3] / 255.0
                            alpha = np.expand_dims(alpha, axis=2)
                            
                            # Get RGB channels
                            overlay_rgb = overlay_resized[:, :, :3]
                            
                            # Blend based on alpha
                            blended = (overlay_rgb * alpha) + (face_roi * (1.0 - alpha))
                            frame[y:y+h, x:x+w] = blended.astype(np.uint8)
                        else:
                            # Just copy the overlay
                            frame[y:y+h, x:x+w] = overlay_resized
                
                # Save processed frame
                processed_path = os.path.join(processed_dir, frame_file)
                cv2.imwrite(processed_path, frame)
            
            # Create video from processed frames
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            if process_audio and os.path.exists(audio_path):
                # With audio
                video_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(processed_dir, "frame_%05d.png"),
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
                    "-i", os.path.join(processed_dir, "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    output_path
                ]
            
            # Run video creation command
            result = subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Clean up
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            # Check result
            if result.returncode != 0:
                return False, "Failed to create output video"
            
            # Report completion
            if progress_callback:
                progress_callback(1.0)
            
            return True, output_path
            
        except Exception as e:
            logger.error(f"Video mosaic error: {str(e)}")
            
            # Clean up on error
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
            except:
                pass
                
            return False, f"Video mosaic error: {str(e)}"
    
    def _is_video_file(self, file_path):
        """Check if a file is a video based on extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        ext = os.path.splitext(file_path.lower())[1]
        return ext in video_extensions
    
    def cancel(self):
        """Cancel ongoing face mosaic operation."""
        with self.processing_lock:
            self.cancel_requested = True
        return True


# For testing the module directly
if __name__ == "__main__":
    engine = FaceMosaicEngine()
    
    # Test mosaic if arguments provided
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        mosaic_type = sys.argv[3] if len(sys.argv) > 3 else "blur"
        
        if os.path.exists(input_path):
            print(f"Applying {mosaic_type} mosaic to {input_path}...")
            
            # Define progress callback
            def progress_callback(progress):
                print(f"Progress: {progress*100:.1f}%")
            
            # Set parameters
            params = {
                'input_path': input_path,
                'output_path': output_path,
                'mosaic_type': mosaic_type,
                'effect_intensity': 15.0,
                'track_faces': True,
                'process_audio': True,
                'detect_only': False,
                'detection_threshold': 0.5,
                'progress_callback': progress_callback
            }
            
            # Apply mosaic
            success, result = engine.apply_mosaic(params)
            
            if success:
                print(f"Mosaic completed: {result}")
            else:
                print(f"Mosaic failed: {result}")
        else:
            print(f"Input file not found: {input_path}")
    else:
        print("Usage: python face_mosaic_module.py <input_file> <output_file> [mosaic_type]")
        print("Available mosaic types: blur, pixelate, black, emoji, custom_image") 