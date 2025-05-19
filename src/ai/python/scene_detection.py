#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
import argparse

class SceneDetector:
    """Scene detection using both traditional methods and deep learning-based approaches."""
    
    def __init__(self, threshold=0.5, use_ml=True):
        """
        Initialize scene detector.
        
        Args:
            threshold: Threshold for scene change detection sensitivity (0.0 to 1.0)
            use_ml: Use machine learning for enhanced scene detection
        """
        self.threshold = threshold
        self.use_ml = use_ml
        self.scenes = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ML model if enabled
        if self.use_ml:
            try:
                # Load ResNet model
                self.model = models.resnet18(pretrained=True).to(self.device)
                self.model.eval()
                
                # Remove the final fully connected layer
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                
                # Set up image transformations
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])
                
                print("ML-based scene detection initialized successfully")
            except Exception as e:
                print(f"Error initializing ML model: {e}")
                self.use_ml = False
    
    def process_video(self, video_path):
        """
        Process a video file and detect scene changes.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of scene change timestamps in milliseconds
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        # Reset scenes list
        self.scenes = []
        
        # Process frames
        prev_frame = None
        prev_features = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Adjust threshold based on total video length
            adaptive_threshold = self.threshold
            
            # Calculate frame timestamp in milliseconds
            timestamp_ms = int((frame_idx * 1000) / fps)
            
            # Process every 5th frame for efficiency
            if frame_idx % 5 == 0:
                # Convert frame to RGB for ML processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.use_ml:
                    # Extract deep features
                    features = self._extract_features(rgb_frame)
                    
                    if prev_features is not None:
                        # Calculate cosine similarity between frame features
                        similarity = torch.cosine_similarity(prev_features, features)
                        
                        # Detect scene change using feature similarity
                        if similarity < (1.0 - adaptive_threshold):
                            self.scenes.append(timestamp_ms)
                            print(f"ML scene change detected at {timestamp_ms}ms (frame {frame_idx})")
                    
                    prev_features = features
                
                else:
                    # Traditional method: histogram comparison
                    if prev_frame is not None:
                        # Convert to grayscale for histogram comparison
                        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate histogram
                        hist_current = cv2.calcHist([gray_current], [0], None, [64], [0, 256])
                        hist_prev = cv2.calcHist([gray_prev], [0], None, [64], [0, 256])
                        
                        # Normalize histograms
                        cv2.normalize(hist_current, hist_current, 0, 1.0, cv2.NORM_MINMAX)
                        cv2.normalize(hist_prev, hist_prev, 0, 1.0, cv2.NORM_MINMAX)
                        
                        # Compare histograms
                        diff = cv2.compareHist(hist_prev, hist_current, cv2.HISTCMP_BHATTACHARYYA)
                        
                        # Detect scene change using histogram difference
                        if diff > adaptive_threshold:
                            self.scenes.append(timestamp_ms)
                            print(f"Traditional scene change detected at {timestamp_ms}ms (frame {frame_idx})")
                
                prev_frame = frame.copy()
            
            frame_idx += 1
            
            # Show progress every 5% of frames
            if frame_idx % int(total_frames / 20) == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Clean up
        cap.release()
        
        # Filter out scenes that are too close together (within 500ms)
        filtered_scenes = []
        for i, scene in enumerate(self.scenes):
            if i == 0 or scene - self.scenes[i-1] > 500:
                filtered_scenes.append(scene)
        
        self.scenes = filtered_scenes
        print(f"Detected {len(self.scenes)} scene changes")
        
        return self.scenes
    
    def _extract_features(self, frame):
        """Extract features from a frame using the neural network."""
        try:
            # Convert frame to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Apply transformations
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                
            return features.flatten()
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def save_results(self, output_file):
        """Save the scene detection results to a JSON file."""
        results = {
            "total_scenes": len(self.scenes),
            "scenes": [{"timestamp_ms": ts} for ts in self.scenes]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="BLOUedit Scene Detector")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", help="Output JSON file", default="scenes.json")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                        help="Detection threshold (0.0-1.0)")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML-based detection")
    
    args = parser.parse_args()
    
    detector = SceneDetector(threshold=args.threshold, use_ml=not args.no_ml)
    
    try:
        scenes = detector.process_video(args.video_path)
        detector.save_results(args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 