#!/bin/bash

# Build script for ViT C implementation

set -e  # Exit on any error

echo "Building ViT C Implementation"
echo "=============================="

# Check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake is not installed. Please install cmake first."
    exit 1
fi

# Check if make is installed
if ! command -v make &> /dev/null; then
    echo "Error: make is not installed. Please install make first."
    exit 1
fi

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Build completed successfully!"
    echo "Executable location: build/bin/vit"
    echo ""
    echo "Usage examples:"
    echo "  ./build/bin/vit --help"
    echo "  ./build/bin/vit -i image.bmp"
    echo ""
else
    echo "Build failed!"
    exit 1
fi

