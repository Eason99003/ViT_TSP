# Vision Transformer (ViT) Implementation in C

This project implements a **working Vision Transformer (ViT) model in C** with the following specifications:

- **Input Image Size**: 224×224 pixels
- **Hidden Size**: 192
- **Number of Heads**: 3
- **Number of Layers**: 12
- **Model**: vit_tiny_patch16_224.augreg_in21k_ft_in1k
- **Number of Classes**: 1000 (ImageNet)

## Features

- Complete ViT architecture implementation
- Multi-head self-attention mechanism
- Patch embedding and positional encoding
- Layer normalization and MLP blocks
- Image preprocessing (resizing and normalization)
- Model loading and inference
- Command-line interface

## Requirements

- C99 compatible compiler (GCC, Clang)
- CMake 3.10 or higher
- libcurl development libraries
- libjpeg development libraries
- Standard C library

### Installing Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libcurl4-openssl-dev libjpeg-dev
```

#### macOS:
```bash
brew install cmake curl jpeg
```

#### CentOS/RHEL:
```bash
sudo yum install gcc cmake libcurl-devel libjpeg-devel
```

## Building

1. Clone or download this repository
2. Create a build directory:
```bash
mkdir build
cd build
```

3. Configure with CMake:
```bash
cmake ..
```

4. Build the project:
```bash
make
```

## Usage

### Basic Usage
```bash
./bin/vit -i image.jpg
```

### Advanced Usage
```bash
./bin/vit -i image.jpg -m model.bin -k 10
```

### Command Line Options

- `-i, --image <path>`: Path to input image (JPEG or BMP format)
- `-m, --model <path>`: Path to model file (default: vit_model.bin)
- `-k, --top-k <num>`: Number of top predictions to show (default: 5)
- `-h, --help`: Show help message

### Examples

```bash
# Run inference on an image with default settings
./bin/vit -i my_image.jpg

# Run inference with custom model and show top 10 predictions
./bin/vit -i my_image.jpg -m my_model.bin -k 10

# Show help
./bin/vit --help
```

## Model Loading

The implementation attempts to download the model automatically from Hugging Face. If the download fails, it will initialize the model with random weights for demonstration purposes.

For production use, you would need to:
1. Convert a pre-trained model to a compatible format
2. Implement proper model serialization/deserialization
3. Load actual trained weights

## Image Format Support

Currently, the implementation supports:
- JPEG format images
- BMP format images
- Automatic image resizing to 224×224
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Architecture Details

### ViT Components

1. **Patch Embedding**: Converts 16×16 patches into 192-dimensional embeddings
2. **Positional Encoding**: Adds positional information to embeddings
3. **Multi-Head Attention**: 3 attention heads with 64 dimensions each
4. **MLP Blocks**: Feed-forward networks with GELU activation
5. **Layer Normalization**: Applied before attention and MLP blocks
6. **Classification Head**: Final linear layer for 1000-class prediction

### Memory Layout

- Input: 224×224×3 RGB image
- Patches: 196 patches of 16×16 pixels
- Embeddings: 197 tokens (196 patches + 1 CLS token) × 192 dimensions
- Output: 1000 class probabilities

## Performance

The implementation is optimized for clarity and educational purposes. For production use, consider:
- Using BLAS libraries for matrix operations
- Implementing batch processing
- Using SIMD instructions
- GPU acceleration

## Limitations

- Random weight initialization (not pre-trained)
- Limited image format support (JPEG and BMP)
- No batch processing
- Simplified attention implementation
- No gradient computation (inference only)

## File Structure

```
vit_c/
├── include/
│   └── vit.h              # Main header file
├── src/
│   ├── vit.c              # Core ViT implementation
│   ├── image_processing.c # Image loading and preprocessing
│   ├── model_loader.c     # Model loading and downloading
│   └── main.c             # Main inference program
├── CMakeLists.txt         # Build configuration
└── README.md              # This file
```

## Contributing

This is an educational implementation. For improvements, consider:
- Adding support for more image formats (PNG, TIFF)
- Implementing proper model format support (ONNX, PyTorch)
- Adding batch processing capabilities
- Optimizing matrix operations
- Adding unit tests

## License

This project is for educational purposes. Please ensure compliance with the licenses of any pre-trained models you use.

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
