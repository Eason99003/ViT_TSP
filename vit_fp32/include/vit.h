#ifndef VIT_H
#define VIT_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

// Ensure bool is defined for C99 compatibility
#ifndef __cplusplus
#include <stdbool.h>
#endif

// Configuration parameters for ViT Tiny
#define IMG_SIZE 224
#define PATCH_SIZE 16
#define NUM_PATCHES ((IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE))
#define EMBED_DIM 192
#define NUM_HEADS 3
#define NUM_LAYERS 12
#define MLP_RATIO 4.0f
#define MLP_DIM (int)(EMBED_DIM * MLP_RATIO)
#define NUM_CLASSES 1000
#define MAX_IMAGE_SIZE 1024

// Data types
typedef float vit_float_t;

// Matrix structure for computations
typedef struct {
    vit_float_t* data;
    int rows;
    int cols;
} Matrix;


// ViT model structure
typedef struct {
    // Embedding layers
    Matrix patch_embed_weight;    // [PATCH_SIZE*PATCH_SIZE*3, EMBED_DIM]
    Matrix patch_embed_bias;      // [EMBED_DIM]
    Matrix pos_embed;             // [NUM_PATCHES+1, EMBED_DIM]
    
    // Transformer blocks
    Matrix* layer_norm1_weight;   // [NUM_LAYERS][EMBED_DIM]
    Matrix* layer_norm1_bias;    // [NUM_LAYERS][EMBED_DIM]
    Matrix* attention_weights;    // [NUM_LAYERS][EMBED_DIM, EMBED_DIM]
    Matrix* attention_bias;       // [NUM_LAYERS][EMBED_DIM]
    Matrix* qkv_weights;          // [NUM_LAYERS][3*EMBED_DIM, EMBED_DIM]
    Matrix* qkv_bias;             // [NUM_LAYERS][3*EMBED_DIM]
    Matrix* proj_weights;         // [NUM_LAYERS][EMBED_DIM, EMBED_DIM]
    Matrix* proj_bias;            // [NUM_LAYERS][EMBED_DIM]
    
    Matrix* layer_norm2_weight;   // [NUM_LAYERS][EMBED_DIM]
    Matrix* layer_norm2_bias;    // [NUM_LAYERS][EMBED_DIM]
    Matrix* mlp_weights1;         // [NUM_LAYERS][EMBED_DIM, MLP_DIM]
    Matrix* mlp_bias1;            // [NUM_LAYERS][MLP_DIM]
    Matrix* mlp_weights2;         // [NUM_LAYERS][MLP_DIM, EMBED_DIM]
    Matrix* mlp_bias2;            // [NUM_LAYERS][EMBED_DIM]
    
    // Final layers
    Matrix final_norm_weight;      // [EMBED_DIM]
    Matrix final_norm_bias;       // [EMBED_DIM]
    Matrix head_weights;          // [EMBED_DIM, NUM_CLASSES]
    Matrix head_bias;             // [NUM_CLASSES]
    
    // Class token
    Matrix cls_token;             // [EMBED_DIM]
} ViTModel;

// Function declarations
Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix* m);
Matrix matrix_multiply(const Matrix* a, const Matrix* b);
Matrix matrix_add(const Matrix* a, const Matrix* b);
Matrix gelu(const Matrix* m);
Matrix layer_norm(const Matrix* m, const Matrix* gamma, const Matrix* beta);
Matrix multi_head_attention(const Matrix* x, const Matrix* qkv_weight, const Matrix* qkv_bias,
                           const Matrix* proj_weight, const Matrix* proj_bias, int num_heads);
Matrix mlp_block(const Matrix* x, const Matrix* weight1, const Matrix* bias1,
                const Matrix* weight2, const Matrix* bias2);

// ViT specific functions
ViTModel* create_vit_model();
void free_vit_model(ViTModel* model);
bool load_vit_model(ViTModel* model, const char* model_path);
Matrix patch_embedding(const vit_float_t* image, const Matrix* patch_weight, const Matrix* patch_bias, const Matrix* cls_token);
Matrix vit_forward(ViTModel* model, const vit_float_t* image);
Matrix vit_inference(ViTModel* model, const vit_float_t* image);

// Image preprocessing
bool load_and_preprocess_image(const char* image_path, vit_float_t* output);
bool resize_image(const vit_float_t* input, int input_width, int input_height, 
                 vit_float_t* output, int output_width, int output_height);

// Utility functions
void print_matrix(const Matrix* m, const char* name);
vit_float_t* normalize_image(const vit_float_t* image, int size);

#endif // VIT_H
