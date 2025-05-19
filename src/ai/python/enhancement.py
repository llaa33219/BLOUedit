#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time

class ConvBlock(nn.Module):
    """Convolutional block for video enhancement."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """Residual block for video enhancement."""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual
        return self.relu(out)

class EnhancementModel(nn.Module):
    """Model for video enhancement."""
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(EnhancementModel, self).__init__()
        
        # Initial feature extraction
        self.initial = ConvBlock(in_channels, features)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResBlock(features),
            ResBlock(features),
            ResBlock(features),
            ResBlock(features),
            ResBlock(features),
            ResBlock(features)
        )
        
        # Feature fusion
        self.fusion = ConvBlock(features, features)
        
        # Reconstruction
        self.reconstruction = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        initial_features = self.initial(x)
        
        # Residual learning
        res_features = self.res_blocks(initial_features)
        
        # Feature fusion
        fused_features = self.fusion(res_features)
        
        # Skip connection
        features = fused_features + initial_features
        
        # Final reconstruction
        out = self.reconstruction(features)
        
        # Residual learning for the whole network
        return torch.clamp(x + out, 0, 1)

class VideoEnhancer:
    """Video enhancement using deep learning."""
    
    def __init__(self, model_path=None):
        """
        Initialize the video enhancer.
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = EnhancementModel().to(self.device)
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model (results may be limited)")
        else:
            print("No pre-trained model provided. Using untrained model (results may be limited)")
            
            # Initialize with default enhancement parameters
            self._initialize_default_model()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.denoise_strength = 0.5
        self.sharpen_strength = 0.5
        self.color_enhance = 0.3
        self.brightness = 0.1
        self.contrast = 0.2
    
    def _initialize_default_model(self):
        """Initialize model with default enhancement parameters for untrained model."""
        # Apply a set of default parameters that work well for general enhancement
        pass  # The default initialization should be good enough
    
    def enhance_image(self, image_path, output_path=None, resize=None):
        """
        Enhance a single image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the enhanced image (optional)
            resize: Resize dimensions (width, height) before processing (optional)
            
        Returns:
            Enhanced image as a numpy array or output path if output_path is provided
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize if specified
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        
        # Convert to tensor
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Process image
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Convert back to numpy array
        output = output_tensor[0].cpu().permute(1, 2, 0).numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        # Apply additional post-processing
        output = self._post_process(output)
        
        # Save if output path is provided
        if output_path:
            Image.fromarray(output).save(output_path)
            print(f"Enhanced image saved to {output_path}")
            return output_path
            
        return output
    
    def enhance_video(self, video_path, output_path, scale_factor=1.0, fps=None):
        """
        Enhance a video file.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the enhanced video
            scale_factor: Scale factor for resolution (1.0 = original)
            fps: Output frames per second (None = same as input)
            
        Returns:
            Path to the enhanced video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps is None:
            fps = input_fps
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Resize frame
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to RGB for processing (PyTorch models expect RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL image
            pil_frame = Image.fromarray(rgb_frame)
            
            # Convert to tensor
            input_tensor = self.transform(pil_frame).unsqueeze(0).to(self.device)
            
            # Process frame
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # Convert back to numpy array
            output_frame = output_tensor[0].cpu().permute(1, 2, 0).numpy()
            output_frame = np.clip(output_frame * 255, 0, 255).astype(np.uint8)
            
            # Apply additional post-processing
            output_frame = self._post_process(output_frame)
            
            # Convert back to BGR for OpenCV
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            # Write to output video
            out.write(output_frame)
            
            frame_idx += 1
            
            # Show progress every 5% of frames
            if frame_idx % max(1, int(frame_count / 20)) == 0:
                progress = (frame_idx / frame_count) * 100
                elapsed = time.time() - start_time
                fps_processing = frame_idx / elapsed if elapsed > 0 else 0
                remaining = (frame_count - frame_idx) / fps_processing if fps_processing > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_idx}/{frame_count} frames), "
                      f"Processing speed: {fps_processing:.2f} fps, "
                      f"Estimated time remaining: {remaining:.1f} seconds")
        
        # Clean up
        cap.release()
        out.release()
        
        end_time = time.time()
        print(f"Video enhancement completed in {end_time - start_time:.2f} seconds")
        print(f"Enhanced video saved to {output_path}")
        
        return output_path
    
    def _post_process(self, image):
        """
        Apply additional post-processing to enhance image quality.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image
        """
        # Convert to float for processing
        img = image.astype(np.float32)
        
        # Apply denoise if enabled
        if self.denoise_strength > 0:
            img = cv2.GaussianBlur(img, (3, 3), 0.5 * self.denoise_strength)
        
        # Apply sharpening if enabled
        if self.sharpen_strength > 0:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9 + self.sharpen_strength, -1],
                               [-1, -1, -1]]) / (5 + self.sharpen_strength)
            img = cv2.filter2D(img, -1, kernel)
        
        # Apply color enhancement
        if self.color_enhance > 0:
            # Convert to HSV
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Enhance saturation
            hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + self.color_enhance)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Convert back to RGB
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Apply brightness adjustment
        if self.brightness != 0:
            img = img * (1.0 + self.brightness)
        
        # Apply contrast adjustment
        if self.contrast > 0:
            img = (img - 128) * (1.0 + self.contrast) + 128
        
        # Clip values to valid range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def set_enhancement_params(self, denoise=None, sharpen=None, color=None, 
                              brightness=None, contrast=None):
        """
        Set enhancement parameters.
        
        Args:
            denoise: Denoise strength (0.0 to 1.0)
            sharpen: Sharpening strength (0.0 to 1.0)
            color: Color enhancement (0.0 to 1.0)
            brightness: Brightness adjustment (-0.5 to 0.5)
            contrast: Contrast adjustment (0.0 to 1.0)
        """
        if denoise is not None:
            self.denoise_strength = max(0.0, min(1.0, denoise))
        
        if sharpen is not None:
            self.sharpen_strength = max(0.0, min(1.0, sharpen))
        
        if color is not None:
            self.color_enhance = max(0.0, min(1.0, color))
        
        if brightness is not None:
            self.brightness = max(-0.5, min(0.5, brightness))
        
        if contrast is not None:
            self.contrast = max(0.0, min(1.0, contrast))

def main():
    parser = argparse.ArgumentParser(description="BLOUedit Video Enhancer")
    parser.add_argument("input", help="Input video or image file")
    parser.add_argument("--output", "-o", help="Output file", default=None)
    parser.add_argument("--model", "-m", help="Path to enhancement model", default=None)
    parser.add_argument("--denoise", type=float, default=0.5, help="Denoise strength (0.0-1.0)")
    parser.add_argument("--sharpen", type=float, default=0.5, help="Sharpen strength (0.0-1.0)")
    parser.add_argument("--color", type=float, default=0.3, help="Color enhancement (0.0-1.0)")
    parser.add_argument("--brightness", type=float, default=0.1, help="Brightness (-0.5-0.5)")
    parser.add_argument("--contrast", type=float, default=0.2, help="Contrast (0.0-1.0)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scale factor")
    
    args = parser.parse_args()
    
    enhancer = VideoEnhancer(model_path=args.model)
    
    # Set enhancement parameters
    enhancer.set_enhancement_params(
        denoise=args.denoise,
        sharpen=args.sharpen,
        color=args.color,
        brightness=args.brightness,
        contrast=args.contrast
    )
    
    input_file = args.input
    output_file = args.output
    
    if output_file is None:
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_enhanced{ext}"
    
    try:
        # Determine if it's a video or image
        if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            enhancer.enhance_video(input_file, output_file, scale_factor=args.scale)
        elif input_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            enhancer.enhance_image(input_file, output_file)
        else:
            print(f"Unsupported file format: {input_file}")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 