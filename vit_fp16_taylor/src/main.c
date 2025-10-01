#include "../include/vit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#ifdef HAVE_JPEG
#include <jpeglib.h>
#endif

// Function declarations
void print_top_k_predictions(const Matrix* logits, int k);
void softmax(Matrix* logits);
bool load_and_preprocess_image(const char* image_path, vit_float_t* output);
void print_usage(const char* program_name);

// External function declarations from other source files
bool load_jpeg_image(const char* filename, vit_float_t* output, int* width, int* height);
bool load_bmp_image(const char* filename, vit_float_t* output, int* width, int* height);
bool resize_image(const vit_float_t* input, int input_width, int input_height, 
                 vit_float_t* output, int output_width, int output_height);
vit_float_t* normalize_image(const vit_float_t* image, int size);

// Standard softmax without temperature scaling
void softmax(Matrix* logits) {
    // Find maximum for numerical stability
    vit_float_t max_val = logits->data[0];
    for (int i = 1; i < logits->cols; i++) {
        if (logits->data[i] > max_val) {
            max_val = logits->data[i];
        }
    }
    
    // Compute exponentials
    vit_float_t sum = 0.0f;
    for (int i = 0; i < logits->cols; i++) {
        logits->data[i] = expf(logits->data[i] - max_val);
        sum += logits->data[i];
    }
    
    // Normalize with small epsilon to avoid division by zero
    vit_float_t eps = 1e-8f;
    sum = fmaxf(sum, eps);
    
    for (int i = 0; i < logits->cols; i++) {
        logits->data[i] /= sum;
    }
}



// Print top-k predictions with standard formatting
void print_top_k_predictions(const Matrix* logits, int k) {
    if (k > NUM_CLASSES) k = NUM_CLASSES;
    
    // Create arrays to store scores and indices
    vit_float_t* scores = (vit_float_t*)malloc(k * sizeof(vit_float_t));
    int* indices = (int*)malloc(k * sizeof(int));
    
    // Initialize arrays
    for (int i = 0; i < k; i++) {
        scores[i] = -INFINITY;
        indices[i] = -1;
    }
    
    
    // Find top-k predictions
    for (int i = 0; i < NUM_CLASSES; i++) {
        vit_float_t score = logits->data[i];
        
        // Find position to insert
        int pos = k;
        while (pos > 0 && score > scores[pos - 1]) {
            pos--;
        }
        
        // Shift elements and insert
        if (pos < k) {
            for (int j = k - 1; j > pos; j--) {
                scores[j] = scores[j - 1];
                indices[j] = indices[j - 1];
            }
            scores[pos] = score;
            indices[pos] = i;
        }
    }
    
    // Print results
    printf("\nTop-%d predictions:\n", k);
    printf("Rank | Class ID | Probability | Confidence\n");
    printf("-----|----------|-------------|----------\n");
    
    for (int i = 0; i < k; i++) {
        if (indices[i] >= 0) {
            vit_float_t confidence = scores[i] * 100.0f;
            printf("%4d | %8d | %10.4f | %8.2f%%\n", i + 1, indices[i], scores[i], confidence);
        }
    }
    
    free(scores);
    free(indices);
}

#ifdef USE_FP16
// Test function for exponential lookup table
void test_exponential_table() {
    printf("\n=== Testing Exponential Lookup Table ===\n");
    printf("Range: [%.3f, %.3f], Step: %.3f, Size: %d\n\n", 
           EXP_TABLE_MIN, EXP_TABLE_MAX, EXP_TABLE_STEP, EXP_TABLE_SIZE);
    
    float test_values[] = {-8.0f, -7.875f, -5.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 5.0f, 7.875f, 8.0f, 9.0f};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    printf("Input     | FP32 Standard | FP32 Table | FP16 Table | Table Error | Clamped?\n");
    printf("----------|---------------|------------|------------|-------------|----------\n");
    
    for (int i = 0; i < num_tests; i++) {
        float x = test_values[i];
        float standard = expf(x);
        float table_fp32 = exp_table_fp32(x);
        float table_fp16 = fp16_to_float(exp_table_fp16(float_to_fp16(x)));
        
        float error_fp32 = fabsf(table_fp32 - standard) / standard * 100.0f;
        float error_fp16 = fabsf(table_fp16 - standard) / standard * 100.0f;
        
        bool clamped = (x < EXP_TABLE_MIN || x > EXP_TABLE_MAX);
        const char* clamped_str = clamped ? "Yes" : "No";
        
        printf("%8.3f | %13.6f | %10.6f | %10.6f | %10.2f%% | %8s\n", 
               x, standard, table_fp32, table_fp16, error_fp32, clamped_str);
    }
    
    printf("\n=== Table Sample Values ===\n");
    printf("Index | Input   | FP16 Table Value | FP32 Value\n");
    printf("------|---------|------------------|-----------\n");
    for (int i = 0; i < 10; i++) {
        float x = EXP_TABLE_MIN + i * EXP_TABLE_STEP;
        vit_fp16_t fp16_val = get_exp_table_entry(i);
        float fp32_val = fp16_to_float(fp16_val);
        printf("%5d | %7.3f | %16.6f | %9.6f\n", i, x, fp32_val, expf(x));
    }
    printf("\n");
}
#endif


// Print usage information
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -i <path>    Input image path\n");
    printf("  -k <num>     Number of top predictions to show (default: 5)\n");
    printf("  -m <path>    Model file path (default: vit_model.bin)\n");
#ifdef USE_FP16
    printf("  -f           Use FP16 for transformer encoder (faster, lower precision)\n");
    printf("  -t           Test exponential lookup table\n");
#endif
    printf("  -h           Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s -i image.jpg\n", program_name);
    printf("  %s -i image.jpg -k 10\n", program_name);
    printf("\n");
    printf("Standard ViT Features:\n");
    printf("  - Standard Vision Transformer architecture\n");
    printf("  - Standard softmax for probability distribution\n");
#ifdef USE_FP16
    printf("  - FP16 support for transformer encoder\n");
#endif
    printf("\n");
#ifdef HAVE_JPEG
    printf("Supported formats: JPEG, BMP\n");
#else
    printf("Supported formats: BMP (JPEG support not compiled)\n");
#endif
}

int main(int argc, char* argv[]) {
    printf("ViT Tiny Implementation - Standard Architecture\n");
    printf("===============================================\n");
    printf("Image size: %dx%d\n", IMG_SIZE, IMG_SIZE);
    printf("Embedding dimension: %d\n", EMBED_DIM);
    printf("Number of heads: %d\n", NUM_HEADS);
    printf("Number of layers: %d\n", NUM_LAYERS);
    printf("Number of classes: %d\n\n", NUM_CLASSES);
    
    // Parse command line arguments
    const char* image_path = NULL;
    int top_k = 5;
    const char* model_path = "vit_model.bin";
    bool use_fp16 = false;
    bool test_table = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
#ifdef USE_FP16
        } else if (strcmp(argv[i], "-f") == 0) {
            use_fp16 = true;
        } else if (strcmp(argv[i], "-t") == 0) {
            test_table = true;
#endif
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Test exponential table if requested
#ifdef USE_FP16
    if (test_table) {
        test_exponential_table();
        return 0;
    }
#endif
    
    // If no image path provided, create a test image
    if (!image_path) {
        printf("No image path provided. Creating a test image for demonstration.\n");
        image_path = "test_image.jpg";
        
#ifdef HAVE_JPEG
        // Create a simple test JPEG image
        FILE* test_file = fopen(image_path, "wb");
        if (test_file) {
            struct jpeg_compress_struct cinfo;
            struct jpeg_error_mgr jerr;
            
            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_compress(&cinfo);
            jpeg_stdio_dest(&cinfo, test_file);
            
            cinfo.image_width = 224;
            cinfo.image_height = 224;
            cinfo.input_components = 3;
            cinfo.in_color_space = JCS_RGB;
            
            jpeg_set_defaults(&cinfo);
            jpeg_set_quality(&cinfo, 95, TRUE);
            jpeg_start_compress(&cinfo, TRUE);
            
            // Create gradient image data
            JSAMPLE* row_buffer = (JSAMPLE*)malloc(224 * 3);
            for (int y = 0; y < 224; y++) {
                for (int x = 0; x < 224; x++) {
                    row_buffer[x * 3 + 0] = (JSAMPLE)(x * 255 / 224);     // R
                    row_buffer[x * 3 + 1] = (JSAMPLE)(y * 255 / 224);     // G
                    row_buffer[x * 3 + 2] = (JSAMPLE)(128);               // B
                }
                jpeg_write_scanlines(&cinfo, &row_buffer, 1);
            }
            
            free(row_buffer);
            jpeg_finish_compress(&cinfo);
            jpeg_destroy_compress(&cinfo);
            fclose(test_file);
            printf("Created test image: %s\n", image_path);
        }
#else
        // Create a simple test BMP image
        FILE* test_file = fopen(image_path, "wb");
        if (test_file) {
            // Write a minimal BMP header for a 224x224 image
            unsigned char bmp_header[54] = {
                'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0,
                224, 0, 0, 0,  // Width: 224
                224, 0, 0, 0,  // Height: 224
                1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };
            
            // Calculate file size
            int row_size = ((224 * 3 + 3) / 4) * 4;
            int file_size = 54 + row_size * 224;
            
            // Update file size in header
            *(int*)&bmp_header[2] = file_size;
            *(int*)&bmp_header[34] = row_size * 224;
            
            fwrite(bmp_header, 1, 54, test_file);
            
            // Write image data (simple gradient)
            unsigned char* row = (unsigned char*)malloc(row_size);
            for (int y = 0; y < 224; y++) {
                memset(row, 0, row_size);
                for (int x = 0; x < 224; x++) {
                    int pixel_idx = x * 3;
                    row[pixel_idx + 0] = (unsigned char)(x * 255 / 224);     // B
                    row[pixel_idx + 1] = (unsigned char)(y * 255 / 224);     // G
                    row[pixel_idx + 2] = (unsigned char)(128);               // R
                }
                fwrite(row, 1, row_size, test_file);
            }
            
            free(row);
            fclose(test_file);
            printf("Created test image: %s\n", image_path);
        }
#endif
    }
    
    // Create ViT model
    printf("Creating ViT model...\n");
    ViTModel* model = create_vit_model();
    if (!model) {
        printf("Error: Failed to create ViT model\n");
        return 1;
    }
    
    // Load model weights
    printf("Loading model from: %s\n", model_path);
    if (!load_vit_model(model, model_path)) {
        printf("Error: Failed to load model\n");
        free_vit_model(model);
        return 1;
    }
    
    printf("Model loaded successfully!\n");
    
    // Load and preprocess image
    printf("Loading image: %s\n", image_path);
    vit_float_t* processed_image = (vit_float_t*)malloc(IMG_SIZE * IMG_SIZE * 3 * sizeof(vit_float_t));
    if (!processed_image) {
        printf("Error: Failed to allocate memory for processed image\n");
        free_vit_model(model);
        return 1;
    }
    
    // Load and preprocess image using C implementation
    printf("Loading and preprocessing image...\n");
    if (!load_and_preprocess_image(image_path, processed_image)) {
        printf("Error: Failed to load and preprocess image\n");
        free(processed_image);
        free_vit_model(model);
        return 1;
    }
    
    // Run inference
    printf("Running inference...\n");
#ifdef USE_FP16
    if (use_fp16) {
        printf("Using FP16 for transformer encoder computations\n");
    } else {
        printf("Using FP32 for all computations\n");
    }
#else
    printf("Using FP32 for all computations\n");
#endif
    
    clock_t start_time = clock();
    
    Matrix logits;
    printf("Running single prediction...\n");
#ifdef USE_FP16
    if (use_fp16) {
        logits = vit_forward_fp16(model, processed_image);
    } else {
        logits = vit_forward(model, processed_image);
    }
#else
    logits = vit_forward(model, processed_image);
#endif
    
    clock_t end_time = clock();
    vit_float_t inference_time = (vit_float_t)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Inference completed in %.3f seconds\n", inference_time);
    
    // Apply softmax
    printf("Applying softmax...\n");
    softmax(&logits);
    
    // Print top-k predictions
    print_top_k_predictions(&logits, top_k);
    
    printf("\nStandard ViT inference completed successfully!\n");
    
    // Cleanup
    free_matrix(&logits);
    free(processed_image);
    free_vit_model(model);
    
    return 0;
}
