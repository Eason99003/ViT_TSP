#!/usr/bin/env python3
"""
Script to download ViT model weights using timm library
"""

import os
import sys
import torch
import timm
import argparse

def download_vit_model(model_name, output_path):
    """Download ViT model weights and save them"""
    try:
        print(f"Loading model: {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        print("Model loaded successfully!")
        print(f"Model architecture: {model}")
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Save the model weights
        torch.save(state_dict, output_path)
        print(f"Model weights saved to: {output_path}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Print some key weight shapes
        print("\nKey weight shapes:")
        for name, param in state_dict.items():
            if any(key in name for key in ['patch_embed', 'pos_embed', 'cls_token', 'head']):
                print(f"  {name}: {param.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download ViT model weights')
    parser.add_argument('--model', default='vit_tiny_patch16_224.augreg_in21k_ft_in1k',
                       help='Model name to download')
    parser.add_argument('--output', default='vit_model.pth',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Check if timm is available
    try:
        import timm
        print(f"timm version: {timm.__version__}")
    except ImportError:
        print("Error: timm library not found. Please install it with: pip install timm")
        sys.exit(1)
    
    # Check if torch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Error: PyTorch not found. Please install it with: pip install torch")
        sys.exit(1)
    
    # Download model
    success = download_vit_model(args.model, args.output)
    
    if success:
        print("\nModel download completed successfully!")
        print(f"You can now use the weights from: {args.output}")
    else:
        print("\nModel download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

