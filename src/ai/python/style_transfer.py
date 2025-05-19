#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import argparse
import time

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()

    def forward(self, x):
        gram = self._gram_matrix(x)
        self.loss = F.mse_loss(gram, self.target)
        return x
    
    def _gram_matrix(self, x):
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class StyleTransfer:
    """Neural style transfer implementation for BLOUedit."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # VGG19 model
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        
        # Normalization mean and std
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Content and style layers
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # Image loader
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        
        # Unloader
        self.unloader = transforms.ToPILImage()
    
    def _image_loader(self, image_path, size=None):
        """Load an image and convert it to a torch tensor."""
        image = Image.open(image_path).convert('RGB')
        
        if size is not None:
            image = image.resize((size, size), Image.LANCZOS)
        
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def _get_style_model_and_losses(self, style_img, content_img):
        """Create a model with content and style loss modules."""
        normalization = Normalization(self.mean, self.std).to(self.device)
        
        content_losses = []
        style_losses = []
        
        model = nn.Sequential(normalization)
        
        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            model.add_module(name, layer)
            
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)
            
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)
        
        # Trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def _run_style_transfer(self, content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer optimization."""
        print('Building the style transfer model...')
        model, style_losses, content_losses = self._get_style_model_and_losses(style_img, content_img)
        
        # Optimize only the input image
        input_img.requires_grad_(True)
        model.requires_grad_(False)
        
        optimizer = optim.LBFGS([input_img])
        
        print('Optimizing...')
        run = [0]
        best_img = None
        best_loss = float('inf')
        
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 255)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}:")
                    print(f'Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}')
                    print()
                
                # Save the best result
                if loss.item() < best_loss[0]:
                    best_loss[0] = loss.item()
                    best_img[0] = input_img.data.clone()
                
                return style_score + content_score
            
            best_loss = [float('inf')]
            best_img = [None]
            optimizer.step(closure)
        
        # Final iteration
        input_img.data.clamp_(0, 255)
        
        # Use the best image if available
        if best_img[0] is not None:
            input_img.data = best_img[0]
        
        return input_img
    
    def transfer_style(self, content_path, style_path, output_path, 
                       content_size=512, style_size=512, steps=300):
        """
        Transfer style from style image to content image.
        
        Args:
            content_path: Path to the content image
            style_path: Path to the style image
            output_path: Path to save the output image
            content_size: Size to resize content image
            style_size: Size to resize style image
            steps: Number of optimization steps
            
        Returns:
            Path to the output image
        """
        print(f"Content image: {content_path}")
        print(f"Style image: {style_path}")
        
        # Load images
        content_img = self._image_loader(content_path, size=content_size)
        style_img = self._image_loader(style_path, size=style_size)
        
        # Use content image as the initial guess
        input_img = content_img.clone()
        
        # Run style transfer
        start_time = time.time()
        output = self._run_style_transfer(content_img, style_img, input_img, 
                                         num_steps=steps, style_weight=1000000, 
                                         content_weight=1)
        end_time = time.time()
        
        print(f"Style transfer completed in {end_time - start_time:.2f} seconds")
        
        # Save output image
        output_img = output[0].cpu().clone()
        output_img = output_img.clamp(0, 255).numpy()
        output_img = output_img.transpose(1, 2, 0).astype("uint8")
        output_img = Image.fromarray(output_img)
        output_img.save(output_path)
        
        print(f"Output image saved to: {output_path}")
        return output_path
    
    def process_video_frame(self, frame, style_img):
        """
        Apply style transfer to a single video frame.
        
        Args:
            frame: Input frame (numpy array)
            style_img: Style image tensor
            
        Returns:
            Styled frame (numpy array)
        """
        # Convert frame to tensor
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = self.loader(pil_frame).unsqueeze(0).to(self.device, torch.float)
        
        # Use frame as initial guess
        input_tensor = frame_tensor.clone()
        
        # Apply fast style transfer (simplified for video)
        output = self._run_style_transfer(frame_tensor, style_img, input_tensor, 
                                         num_steps=100, style_weight=1000000, 
                                         content_weight=1)
        
        # Convert output tensor to numpy array
        output_frame = output[0].cpu().clone()
        output_frame = output_frame.clamp(0, 255).numpy()
        output_frame = output_frame.transpose(1, 2, 0).astype("uint8")
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description='BLOUedit Style Transfer')
    parser.add_argument('content_image', type=str, help='Path to content image')
    parser.add_argument('style_image', type=str, help='Path to style image')
    parser.add_argument('--output', '-o', type=str, default='output.jpg', help='Output image path')
    parser.add_argument('--content-size', type=int, default=512, help='Content image size')
    parser.add_argument('--style-size', type=int, default=512, help='Style image size')
    parser.add_argument('--steps', type=int, default=300, help='Optimization steps')
    
    args = parser.parse_args()
    
    style_transfer = StyleTransfer()
    
    try:
        style_transfer.transfer_style(
            args.content_image, 
            args.style_image,
            args.output,
            content_size=args.content_size,
            style_size=args.style_size,
            steps=args.steps
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 