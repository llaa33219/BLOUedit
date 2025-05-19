#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bridge module for BLOUedit to connect C++ and Python AI functionality.
This script handles the communication between the C++ application and Python AI modules.
"""

import os
import sys
import json
import argparse
import traceback
from scene_detection import SceneDetector
from style_transfer import StyleTransfer
from enhancement import VideoEnhancer

class AIBridge:
    """Bridge between C++ and Python for AI functionality."""
    
    def __init__(self):
        """Initialize the bridge."""
        self.scene_detector = None
        self.style_transfer = None
        self.video_enhancer = None
        
        # Default output directory
        self.output_dir = os.path.expanduser("~/.cache/blouedit/ai")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_scenes(self, video_path, threshold=0.5, use_ml=True, output_file=None):
        """
        Detect scenes in a video.
        
        Args:
            video_path: Path to the video file
            threshold: Detection threshold (0.0-1.0)
            use_ml: Use machine learning for detection
            output_file: Path to save results (optional)
            
        Returns:
            JSON string with detection results
        """
        if self.scene_detector is None:
            self.scene_detector = SceneDetector(threshold=threshold, use_ml=use_ml)
        else:
            self.scene_detector.threshold = threshold
            self.scene_detector.use_ml = use_ml
        
        if output_file is None:
            output_file = os.path.join(self.output_dir, "scenes.json")
        
        try:
            scenes = self.scene_detector.process_video(video_path)
            self.scene_detector.save_results(output_file)
            
            results = {
                "status": "success",
                "scenes": scenes,
                "total_scenes": len(scenes),
                "output_file": output_file
            }
            
            return json.dumps(results)
        except Exception as e:
            error_message = str(e)
            traceback.print_exc()
            
            results = {
                "status": "error",
                "error": error_message
            }
            
            return json.dumps(results)
    
    def apply_style_transfer(self, content_path, style_path, output_path=None, 
                            content_size=512, style_size=512, steps=300):
        """
        Apply style transfer.
        
        Args:
            content_path: Path to content image or video
            style_path: Path to style image
            output_path: Path to save output (optional)
            content_size: Size for content image
            style_size: Size for style image
            steps: Optimization steps
            
        Returns:
            JSON string with results
        """
        if self.style_transfer is None:
            self.style_transfer = StyleTransfer()
        
        if output_path is None:
            name, ext = os.path.splitext(content_path)
            output_path = os.path.join(self.output_dir, f"{os.path.basename(name)}_styled{ext}")
        
        try:
            result_path = self.style_transfer.transfer_style(
                content_path, 
                style_path, 
                output_path,
                content_size=content_size,
                style_size=style_size,
                steps=steps
            )
            
            results = {
                "status": "success",
                "output_path": result_path
            }
            
            return json.dumps(results)
        except Exception as e:
            error_message = str(e)
            traceback.print_exc()
            
            results = {
                "status": "error",
                "error": error_message
            }
            
            return json.dumps(results)
    
    def enhance_video(self, input_path, output_path=None, denoise=0.5, sharpen=0.5,
                     color=0.3, brightness=0.1, contrast=0.2, scale=1.0):
        """
        Enhance a video or image.
        
        Args:
            input_path: Path to input video or image
            output_path: Path to save output (optional)
            denoise: Denoise strength (0.0-1.0)
            sharpen: Sharpening strength (0.0-1.0)
            color: Color enhancement (0.0-1.0)
            brightness: Brightness adjustment (-0.5-0.5)
            contrast: Contrast adjustment (0.0-1.0)
            scale: Resolution scale factor
            
        Returns:
            JSON string with results
        """
        if self.video_enhancer is None:
            self.video_enhancer = VideoEnhancer()
        
        # Set enhancement parameters
        self.video_enhancer.set_enhancement_params(
            denoise=denoise,
            sharpen=sharpen,
            color=color,
            brightness=brightness,
            contrast=contrast
        )
        
        if output_path is None:
            name, ext = os.path.splitext(input_path)
            output_path = os.path.join(self.output_dir, f"{os.path.basename(name)}_enhanced{ext}")
        
        try:
            # Determine if it's a video or image
            if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                result_path = self.video_enhancer.enhance_video(
                    input_path, 
                    output_path, 
                    scale_factor=scale
                )
            elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                result_path = self.video_enhancer.enhance_image(
                    input_path, 
                    output_path
                )
            else:
                raise ValueError(f"Unsupported file format: {input_path}")
            
            results = {
                "status": "success",
                "output_path": result_path
            }
            
            return json.dumps(results)
        except Exception as e:
            error_message = str(e)
            traceback.print_exc()
            
            results = {
                "status": "error",
                "error": error_message
            }
            
            return json.dumps(results)

def main():
    """Main entry point for the bridge script."""
    parser = argparse.ArgumentParser(description="BLOUedit AI Bridge")
    parser.add_argument("action", choices=["scene-detection", "style-transfer", "enhance"],
                       help="AI action to perform")
    
    # Common arguments
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    # Scene detection arguments
    parser.add_argument("--video", help="Path to input video for scene detection")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Scene detection threshold (0.0-1.0)")
    parser.add_argument("--no-ml", action="store_true", 
                       help="Disable ML-based scene detection")
    
    # Style transfer arguments
    parser.add_argument("--content", help="Path to content image/video")
    parser.add_argument("--style", help="Path to style image")
    parser.add_argument("--content-size", type=int, default=512, 
                       help="Content image size")
    parser.add_argument("--style-size", type=int, default=512, 
                       help="Style image size")
    parser.add_argument("--steps", type=int, default=300, 
                       help="Style transfer optimization steps")
    
    # Enhancement arguments
    parser.add_argument("--input", help="Path to input video/image for enhancement")
    parser.add_argument("--denoise", type=float, default=0.5, 
                       help="Denoise strength (0.0-1.0)")
    parser.add_argument("--sharpen", type=float, default=0.5, 
                       help="Sharpen strength (0.0-1.0)")
    parser.add_argument("--color", type=float, default=0.3, 
                       help="Color enhancement (0.0-1.0)")
    parser.add_argument("--brightness", type=float, default=0.1, 
                       help="Brightness adjustment (-0.5-0.5)")
    parser.add_argument("--contrast", type=float, default=0.2, 
                       help="Contrast adjustment (0.0-1.0)")
    parser.add_argument("--scale", type=float, default=1.0, 
                       help="Resolution scale factor")
    
    args = parser.parse_args()
    
    bridge = AIBridge()
    
    # Set output directory if provided
    if args.output_dir:
        bridge.output_dir = args.output_dir
        os.makedirs(bridge.output_dir, exist_ok=True)
    
    try:
        if args.action == "scene-detection":
            if not args.video:
                parser.error("Scene detection requires --video argument")
            
            result = bridge.detect_scenes(
                args.video,
                threshold=args.threshold,
                use_ml=not args.no_ml,
                output_file=args.output
            )
            print(result)
            
        elif args.action == "style-transfer":
            if not args.content or not args.style:
                parser.error("Style transfer requires --content and --style arguments")
            
            result = bridge.apply_style_transfer(
                args.content,
                args.style,
                output_path=args.output,
                content_size=args.content_size,
                style_size=args.style_size,
                steps=args.steps
            )
            print(result)
            
        elif args.action == "enhance":
            if not args.input:
                parser.error("Enhancement requires --input argument")
            
            result = bridge.enhance_video(
                args.input,
                output_path=args.output,
                denoise=args.denoise,
                sharpen=args.sharpen,
                color=args.color,
                brightness=args.brightness,
                contrast=args.contrast,
                scale=args.scale
            )
            print(result)
    
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 