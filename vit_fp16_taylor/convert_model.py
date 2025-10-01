#!/usr/bin/env python3
"""
Script to convert PyTorch ViT model weights to C-compatible format
"""

import torch
import struct
import os
import argparse

def save_float_array(data, filename):
    """Save a tensor as a binary file of floats"""
    # Convert to numpy and flatten
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Flatten the array
    flat_data = data.flatten()
    
    # Save as binary float32
    with open(filename, 'wb') as f:
        for value in flat_data:
            f.write(struct.pack('f', float(value)))

def convert_pytorch_to_c(pytorch_path, output_path):
    """Convert PyTorch model to C-compatible format"""
    print(f"Loading PyTorch model from: {pytorch_path}")
    
    # Load the PyTorch model
    state_dict = torch.load(pytorch_path, map_location='cpu')
    
    print(f"Model loaded successfully!")
    print(f"Number of parameters: {len(state_dict)}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert each parameter
    for name, param in state_dict.items():
        print(f"Converting: {name} -> shape: {param.shape}")
        
        # Create filename
        safe_name = name.replace('.', '_').replace('(', '_').replace(')', '_')
        filename = os.path.join(output_path, f"{safe_name}.bin")
        
        # Save the parameter
        save_float_array(param, filename)
    
    # Create a manifest file with parameter information
    manifest_path = os.path.join(output_path, "manifest.txt")
    with open(manifest_path, 'w') as f:
        f.write("ViT Model Parameters\n")
        f.write("===================\n\n")
        for name, param in state_dict.items():
            f.write(f"{name}: {list(param.shape)}\n")
    
    print(f"\nConversion completed!")
    print(f"Parameters saved to: {output_path}")
    print(f"Manifest file: {manifest_path}")
    
    return True

def create_simple_binary_format(pytorch_path, output_path):
    """Create a simple binary format for C to read"""
    print(f"Creating simple binary format from: {pytorch_path}")
    
    # Load the PyTorch model
    state_dict = torch.load(pytorch_path, map_location='cpu')
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'VIT_MODEL_V2')
        f.write(struct.pack('I', len(state_dict)))  # Number of parameters
        
        # Write each parameter
        for name, param in state_dict.items():
            # Convert to numpy and flatten
            data = param.detach().cpu().numpy().flatten()
            
            # Write parameter info
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))  # Name length
            f.write(name_bytes)  # Name
            f.write(struct.pack('I', len(param.shape)))  # Shape length
            for dim in param.shape:
                f.write(struct.pack('I', dim))  # Each dimension
            f.write(struct.pack('I', len(data)))  # Data length
            
            # Write data
            for value in data:
                f.write(struct.pack('f', float(value)))
    
    print(f"Simple binary format saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch ViT model to C format')
    parser.add_argument('--input', default='vit_model.pth',
                       help='Input PyTorch model file')
    parser.add_argument('--output-dir', default='model_weights',
                       help='Output directory for individual parameter files')
    parser.add_argument('--output-bin', default='vit_model.bin',
                       help='Output binary file for simple format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return False
    
    # Convert to individual parameter files
    success1 = convert_pytorch_to_c(args.input, args.output_dir)
    
    # Convert to simple binary format
    success2 = create_simple_binary_format(args.input, args.output_bin)
    
    if success1 and success2:
        print("\nModel conversion completed successfully!")
        print(f"Individual parameters: {args.output_dir}/")
        print(f"Binary format: {args.output_bin}")
        return True
    else:
        print("\nModel conversion failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

