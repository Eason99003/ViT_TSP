#include "../include/vit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef HAVE_JPEG
#include <jpeglib.h>
#endif

// Improved bilinear interpolation for image resizing
bool resize_image(const vit_float_t* input, int input_width, int input_height, 
                 vit_float_t* output, int output_width, int output_height) {
    if (!input || !output) return false;
    
    // PyTorch uses align_corners=False by default
    // Exact formula: gx = (x + 0.5) * input_width / output_width - 0.5
    //                gy = (y + 0.5) * input_height / output_height - 0.5
    
    vit_float_t x_scale = (vit_float_t)input_width / output_width;
    vit_float_t y_scale = (vit_float_t)input_height / output_height;
    
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            // Map output coordinates to input coordinates using PyTorch's exact formula
            vit_float_t gx = (x + 0.5f) * x_scale - 0.5f;
            vit_float_t gy = (y + 0.5f) * y_scale - 0.5f;
            
            int gxi = (int)gx;
            int gyi = (int)gy;
            
            // Clamp to valid range
            if (gxi >= input_width - 1) gxi = input_width - 2;
            if (gyi >= input_height - 1) gyi = input_height - 2;
            if (gxi < 0) gxi = 0;
            if (gyi < 0) gyi = 0;
            
            vit_float_t dx = gx - gxi;
            vit_float_t dy = gy - gyi;
            
            for (int c = 0; c < 3; c++) {
                int idx = y * output_width + x;
                int input_idx1 = gyi * input_width + gxi;
                int input_idx2 = gyi * input_width + (gxi + 1);
                int input_idx3 = (gyi + 1) * input_width + gxi;
                int input_idx4 = (gyi + 1) * input_width + (gxi + 1);
                
                vit_float_t c1 = input[input_idx1 * 3 + c] * (1 - dx) * (1 - dy);
                vit_float_t c2 = input[input_idx2 * 3 + c] * dx * (1 - dy);
                vit_float_t c3 = input[input_idx3 * 3 + c] * (1 - dx) * dy;
                vit_float_t c4 = input[input_idx4 * 3 + c] * dx * dy;
                
                output[idx * 3 + c] = c1 + c2 + c3 + c4;
            }
        }
    }
    
    return true;
}

// Center crop function for better preprocessing
bool center_crop(const vit_float_t* input, int input_width, int input_height, 
                vit_float_t* output, int crop_width, int crop_height) {
    if (!input || !output) return false;
    
    int start_x = (input_width - crop_width) / 2;
    int start_y = (input_height - crop_height) / 2;
    
    printf("   Center crop coordinates: start_x=%d, start_y=%d\n", start_x, start_y);
    
    
    for (int y = 0; y < crop_height; y++) {
        for (int x = 0; x < crop_width; x++) {
            int src_x = start_x + x;
            int src_y = start_y + y;
            
            // Clamp to valid range
            if (src_x < 0) src_x = 0;
            if (src_x >= input_width) src_x = input_width - 1;
            if (src_y < 0) src_y = 0;
            if (src_y >= input_height) src_y = input_height - 1;
            
            int dst_idx = y * crop_width + x;
            int src_idx = src_y * input_width + src_x;
            
            for (int c = 0; c < 3; c++) {
                output[dst_idx * 3 + c] = input[src_idx * 3 + c];
            }
        }
    }
    
    
    return true;
}

// Improved image preprocessing with center crop
bool preprocess_image_improved(const vit_float_t* input, int input_width, int input_height,
                              vit_float_t* output, int target_size) {
    if (!input || !output) return false;
    
    // Step 1: Resize to maintain aspect ratio (PyTorch style - resize smaller side to 256)
    int resize_width, resize_height;
    int resize_size = 256;  // PyTorch uses 256, not target_size
    
    if (input_width >= input_height) {
        // Landscape or square: resize smaller side (height) to resize_size
        resize_height = resize_size;
        resize_width = (int)((vit_float_t)resize_size * input_width / input_height);
    } else {
        // Portrait: resize smaller side (width) to resize_size
        resize_width = resize_size;
        resize_height = (int)((vit_float_t)resize_size * input_height / input_width);
    }
    
    printf("=== C Preprocessing Step by Step ===\n");
    printf("1. Original image size: %dx%d\n", input_width, input_height);
    printf("2. After resize(%d): %dx%d\n", resize_size, resize_width, resize_height);
    
    vit_float_t* resized = (vit_float_t*)malloc(resize_width * resize_height * 3 * sizeof(vit_float_t));
    if (!resized) return false;
    
    if (!resize_image(input, input_width, input_height, resized, resize_width, resize_height)) {
        free(resized);
        return false;
    }
    
    // Debug: Save resized image values for comparison
    vit_float_t resize_min = resized[0];
    vit_float_t resize_max = resized[0];
    for (int i = 0; i < resize_width * resize_height * 3; i++) {
        if (resized[i] < resize_min) resize_min = resized[i];
        if (resized[i] > resize_max) resize_max = resized[i];
    }
    printf("   Resized image min/max: %.6f / %.6f\n", resize_min, resize_max);
    printf("   First 5 values (red channel, top row): ");
    for (int j = 0; j < 5; j++) {
        printf("%.8f ", resized[j * 3 + 0]);  // Red channel
    }
    printf("\n");
    
    
    // Step 2: Center crop to target_size x target_size
    printf("3. After center crop(%d): %dx%d\n", target_size, target_size, target_size);
    
    if (!center_crop(resized, resize_width, resize_height, output, target_size, target_size)) {
        free(resized);
        return false;
    }
    
    
    // Step 3: Normalize (this is done in the calling function)
    printf("5. After normalize (done in calling function)\n");
    
    free(resized);
    return true;
}

// JPEG loader (same as before)
#ifdef HAVE_JPEG
bool load_jpeg_image(const char* filename, vit_float_t* output, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return false;
    }
    
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        printf("Error: Not a valid JPEG file\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return false;
    }
    
    // Set output color space to RGB
    cinfo.out_color_space = JCS_RGB;
    cinfo.out_color_components = 3;
    
    *width = cinfo.image_width;
    *height = cinfo.image_height;
    
    if (jpeg_start_decompress(&cinfo) != TRUE) {
        printf("Error: Failed to start JPEG decompression\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return false;
    }
    
    // Allocate row buffer
    JSAMPLE* row_buffer = (JSAMPLE*)malloc(cinfo.output_width * cinfo.output_components);
    if (!row_buffer) {
        printf("Error: Failed to allocate row buffer\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return false;
    }
    
    // Read image data row by row
    for (int y = 0; y < cinfo.output_height; y++) {
        if (jpeg_read_scanlines(&cinfo, &row_buffer, 1) != 1) {
            printf("Error: Failed to read scanline %d\n", y);
            free(row_buffer);
            jpeg_destroy_decompress(&cinfo);
            fclose(file);
            return false;
        }
        
        for (int x = 0; x < cinfo.output_width; x++) {
            int pixel_idx = y * cinfo.output_width + x;
            // JPEG stores RGB, convert to float and normalize to [0,1]
            output[pixel_idx * 3 + 0] = (vit_float_t)row_buffer[x * 3 + 0] / 255.0f; // R
            output[pixel_idx * 3 + 1] = (vit_float_t)row_buffer[x * 3 + 1] / 255.0f; // G
            output[pixel_idx * 3 + 2] = (vit_float_t)row_buffer[x * 3 + 2] / 255.0f; // B
        }
    }
    
    free(row_buffer);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    
    printf("Successfully loaded JPEG image: %dx%d\n", *width, *height);
    
    
    return true;
}
#else
bool load_jpeg_image(const char* filename, vit_float_t* output, int* width, int* height) {
    printf("Error: JPEG support not compiled. Please install libjpeg and rebuild.\n");
    return false;
}
#endif

// BMP loader (same as before)
bool load_bmp_image(const char* filename, vit_float_t* output, int* width, int* height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return false;
    }
    
    // Read BMP header
    unsigned char header[54];
    if (fread(header, 1, 54, file) != 54) {
        fclose(file);
        return false;
    }
    
    // Check if it's a valid BMP file
    if (header[0] != 'B' || header[1] != 'M') {
        printf("Error: Not a valid BMP file\n");
        fclose(file);
        return false;
    }
    
    // Extract width and height
    *width = *(int*)&header[18];
    *height = *(int*)&header[22];
    
    // Check for valid dimensions
    if (*width <= 0 || *height <= 0) {
        printf("Error: Invalid image dimensions\n");
        fclose(file);
        return false;
    }
    
    // Calculate row size (BMP rows are padded to 4-byte boundaries)
    int row_size = ((*width * 3 + 3) / 4) * 4;
    
    // Allocate buffer for image data
    unsigned char* image_data = (unsigned char*)malloc(row_size * *height);
    if (!image_data) {
        printf("Error: Failed to allocate memory for image data\n");
        fclose(file);
        return false;
    }
    
    // Read image data
    fseek(file, 54, SEEK_SET); // Skip header
    if (fread(image_data, 1, row_size * *height, file) != (size_t)(row_size * *height)) {
        printf("Error: Failed to read image data\n");
        free(image_data);
        fclose(file);
        return false;
    }
    
    fclose(file);
    
    // Convert to float and flip vertically (BMP is stored bottom-to-top)
    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            int src_y = *height - 1 - y; // Flip vertically
            int src_idx = src_y * row_size + x * 3;
            int dst_idx = y * *width + x;
            
            // BMP stores BGR, we need RGB
            output[dst_idx * 3 + 0] = (vit_float_t)image_data[src_idx + 2] / 255.0f; // R
            output[dst_idx * 3 + 1] = (vit_float_t)image_data[src_idx + 1] / 255.0f; // G
            output[dst_idx * 3 + 2] = (vit_float_t)image_data[src_idx + 0] / 255.0f; // B
        }
    }
    
    free(image_data);
    return true;
}


// Improved load and preprocess image function
bool load_and_preprocess_image(const char* image_path, vit_float_t* output) {
    vit_float_t* input_image = NULL;
    int input_width, input_height;
    
    // Try to load as BMP first
    input_image = (vit_float_t*)malloc(MAX_IMAGE_SIZE * MAX_IMAGE_SIZE * 3 * sizeof(vit_float_t));
    if (!input_image) {
        printf("Error: Failed to allocate memory for input image\n");
        return false;
    }
    
    bool loaded = false;
    if (strstr(image_path, ".jpg") || strstr(image_path, ".jpeg") || 
        strstr(image_path, ".JPG") || strstr(image_path, ".JPEG")) {
        loaded = load_jpeg_image(image_path, input_image, &input_width, &input_height);
    } else if (strstr(image_path, ".bmp") || strstr(image_path, ".BMP")) {
        loaded = load_bmp_image(image_path, input_image, &input_width, &input_height);
    } else {
        // For unsupported formats, return error
#ifdef HAVE_JPEG
        printf("Error: Only JPEG and BMP formats are supported. Unsupported format: %s\n", image_path);
#else
        printf("Error: Only BMP format is supported (JPEG support not compiled). Unsupported format: %s\n", image_path);
#endif
        loaded = false;
    }
    
    if (!loaded) {
        printf("Error: Failed to load image\n");
        free(input_image);
        return false;
    }
    
    // Use improved preprocessing with center crop
    vit_float_t* preprocessed = (vit_float_t*)malloc(IMG_SIZE * IMG_SIZE * 3 * sizeof(vit_float_t));
    if (!preprocessed) {
        printf("Error: Failed to allocate memory for preprocessed image\n");
        free(input_image);
        return false;
    }
    
    if (!preprocess_image_improved(input_image, input_width, input_height, preprocessed, IMG_SIZE)) {
        printf("Error: Failed to preprocess image\n");
        free(input_image);
        free(preprocessed);
        return false;
    }
    
    free(input_image);
    
    // Normalize the image
    vit_float_t* normalized = normalize_image(preprocessed, IMG_SIZE);
    if (!normalized) {
        printf("Error: Failed to normalize image\n");
        free(preprocessed);
        return false;
    }
    
    
    // Copy to output
    memcpy(output, normalized, IMG_SIZE * IMG_SIZE * 3 * sizeof(vit_float_t));
    
    free(preprocessed);
    free(normalized);
    
    printf("Successfully loaded and preprocessed image: %s\n", image_path);
    printf("Original size: %dx%d, Preprocessed to: %dx%d (with center crop)\n", input_width, input_height, IMG_SIZE, IMG_SIZE);
    
    return true;
}

// Improved normalize_image function - converts HWC to CHW format first, then normalizes
vit_float_t* normalize_image(const vit_float_t* image, int size) {
    vit_float_t* normalized = (vit_float_t*)malloc(size * size * 3 * sizeof(vit_float_t));
    
    // ViT Tiny normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    vit_float_t mean[] = {0.5f, 0.5f, 0.5f};
    vit_float_t std[] = {0.5f, 0.5f, 0.5f};
    
    // Step 1: Convert from HWC to CHW format first (like PyTorch ToTensor())
    vit_float_t* chw_image = (vit_float_t*)malloc(size * size * 3 * sizeof(vit_float_t));
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < size * size; i++) {
            // HWC: image[i * 3 + c] -> CHW: chw_image[c * size * size + i]
            chw_image[c * size * size + i] = image[i * 3 + c];
        }
    }
    
    // Step 2: Normalize the CHW data (like PyTorch Normalize())
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < size * size; i++) {
            normalized[c * size * size + i] = (chw_image[c * size * size + i] - mean[c]) / std[c];
        }
    }
    
    free(chw_image);
    
    
    return normalized;
}


