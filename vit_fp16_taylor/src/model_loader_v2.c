#include "../include/vit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Load float array from binary file
bool load_float_array(const char* filename, vit_float_t* data, int expected_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file: %s\n", filename);
        return false;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Check if size matches expected
    int actual_size = file_size / sizeof(vit_float_t);
    if (actual_size != expected_size) {
        printf("Error: Size mismatch for %s. Expected %d, got %d\n", filename, expected_size, actual_size);
        fclose(file);
        return false;
    }
    
    // Read data
    size_t read = fread(data, sizeof(vit_float_t), expected_size, file);
    fclose(file);
    
    if (read != expected_size) {
        printf("Error: Failed to read complete file: %s\n", filename);
        return false;
    }
    
    return true;
}

// Load model weights from converted binary files
bool load_vit_model_weights(ViTModel* model) {
    printf("Loading ViT model weights from binary files...\n");
    
    // Load patch embedding weights
    // Note: PyTorch conv2d weight shape is [out_channels, in_channels, kernel_h, kernel_w]
    // We need to reshape to [in_channels * kernel_h * kernel_w, out_channels]
    vit_float_t patch_weight_temp[192 * 3 * 16 * 16];
    if (!load_float_array("model_weights/patch_embed_proj_weight.bin", patch_weight_temp, 192 * 3 * 16 * 16)) {
        return false;
    }
    
    // Reshape from [192, 3, 16, 16] to [768, 192]
    for (int out = 0; out < 192; out++) {
        for (int in = 0; in < 3; in++) {
            for (int h = 0; h < 16; h++) {
                for (int w = 0; w < 16; w++) {
                    int src_idx = out * (3 * 16 * 16) + in * (16 * 16) + h * 16 + w;
                    int dst_idx = (in * 16 * 16 + h * 16 + w) * 192 + out;
                    model->patch_embed_weight.data[dst_idx] = patch_weight_temp[src_idx];
                }
            }
        }
    }
    
    if (!load_float_array("model_weights/patch_embed_proj_bias.bin", model->patch_embed_bias.data, EMBED_DIM)) {
        return false;
    }
    
    // Load positional embedding (remove batch dimension)
    vit_float_t pos_embed_temp[197 * 192];
    if (!load_float_array("model_weights/pos_embed.bin", pos_embed_temp, 197 * 192)) {
        return false;
    }
    memcpy(model->pos_embed.data, pos_embed_temp, (NUM_PATCHES + 1) * EMBED_DIM * sizeof(vit_float_t));
    
    // Load CLS token
    vit_float_t cls_token_temp[192];
    if (!load_float_array("model_weights/cls_token.bin", cls_token_temp, 192)) {
        return false;
    }
    memcpy(model->cls_token.data, cls_token_temp, EMBED_DIM * sizeof(vit_float_t));
    
    // Load transformer layers
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        char filename[256];
        
        // Layer norm 1 weight
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_norm1_weight.bin", layer);
        if (!load_float_array(filename, model->layer_norm1_weight[layer].data, EMBED_DIM)) {
            return false;
        }
        
        // Layer norm 1 bias
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_norm1_bias.bin", layer);
        if (!load_float_array(filename, model->layer_norm1_bias[layer].data, EMBED_DIM)) {
            return false;
        }
        
        // Attention QKV weights (keep original shape [576, 192] - no transpose needed)
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_attn_qkv_weight.bin", layer);
        vit_float_t qkv_weight_temp[576 * 192];
        if (!load_float_array(filename, qkv_weight_temp, 576 * 192)) {
            return false;
        }
        
        // Copy weights directly (no transpose - PyTorch uses [576, 192] shape)
        memcpy(model->qkv_weights[layer].data, qkv_weight_temp, 576 * 192 * sizeof(vit_float_t));
        
        // Attention QKV bias
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_attn_qkv_bias.bin", layer);
        if (!load_float_array(filename, model->qkv_bias[layer].data, 3 * EMBED_DIM)) {
            return false;
        }
        
        // Attention projection weights (transpose from [192, 192] to [192, 192])
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_attn_proj_weight.bin", layer);
        vit_float_t proj_weight_temp[192 * 192];
        if (!load_float_array(filename, proj_weight_temp, 192 * 192)) {
            return false;
        }
        
        // Transpose the weight matrix
        for (int i = 0; i < 192; i++) {
            for (int j = 0; j < 192; j++) {
                model->proj_weights[layer].data[i * 192 + j] = proj_weight_temp[j * 192 + i];
            }
        }
        
        // Attention projection bias
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_attn_proj_bias.bin", layer);
        if (!load_float_array(filename, model->proj_bias[layer].data, EMBED_DIM)) {
            return false;
        }
        
        // Layer norm 2 weight
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_norm2_weight.bin", layer);
        if (!load_float_array(filename, model->layer_norm2_weight[layer].data, EMBED_DIM)) {
            return false;
        }
        
        // Layer norm 2 bias
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_norm2_bias.bin", layer);
        if (!load_float_array(filename, model->layer_norm2_bias[layer].data, EMBED_DIM)) {
            return false;
        }
        
        // MLP fc1 weights (transpose from [768, 192] to [192, 768])
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_mlp_fc1_weight.bin", layer);
        vit_float_t mlp1_weight_temp[768 * 192];
        if (!load_float_array(filename, mlp1_weight_temp, 768 * 192)) {
            return false;
        }
        
        // Transpose the weight matrix
        for (int i = 0; i < 192; i++) {
            for (int j = 0; j < 768; j++) {
                model->mlp_weights1[layer].data[i * 768 + j] = mlp1_weight_temp[j * 192 + i];
            }
        }
        
        // MLP fc1 bias
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_mlp_fc1_bias.bin", layer);
        if (!load_float_array(filename, model->mlp_bias1[layer].data, MLP_DIM)) {
            return false;
        }
        
        // MLP fc2 weights (transpose from [192, 768] to [768, 192])
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_mlp_fc2_weight.bin", layer);
        vit_float_t mlp2_weight_temp[192 * 768];
        if (!load_float_array(filename, mlp2_weight_temp, 192 * 768)) {
            return false;
        }
        
        // Transpose the weight matrix
        for (int i = 0; i < 768; i++) {
            for (int j = 0; j < 192; j++) {
                model->mlp_weights2[layer].data[i * 192 + j] = mlp2_weight_temp[j * 768 + i];
            }
        }
        
        // MLP fc2 bias
        snprintf(filename, sizeof(filename), "model_weights/blocks_%d_mlp_fc2_bias.bin", layer);
        if (!load_float_array(filename, model->mlp_bias2[layer].data, EMBED_DIM)) {
            return false;
        }
    }
    
    // Load final layer norm weight
    if (!load_float_array("model_weights/norm_weight.bin", model->final_norm_weight.data, EMBED_DIM)) {
        return false;
    }
    
    // Load final layer norm bias
    if (!load_float_array("model_weights/norm_bias.bin", model->final_norm_bias.data, EMBED_DIM)) {
        return false;
    }
    
    // Load classification head weights (transpose from [1000, 192] to [192, 1000])
    vit_float_t head_weight_temp[1000 * 192];
    if (!load_float_array("model_weights/head_weight.bin", head_weight_temp, 1000 * 192)) {
        return false;
    }
    
    // Transpose the weight matrix
    for (int i = 0; i < 192; i++) {
        for (int j = 0; j < 1000; j++) {
            model->head_weights.data[i * 1000 + j] = head_weight_temp[j * 192 + i];
        }
    }
    
    // Load classification head bias
    if (!load_float_array("model_weights/head_bias.bin", model->head_bias.data, NUM_CLASSES)) {
        return false;
    }
    
    printf("Model weights loaded successfully!\n");
    return true;
}

// Load ViT model with real weights
bool load_vit_model(ViTModel* model, const char* model_path) {
    if (!model) {
        printf("Error: Invalid model parameter\n");
        return false;
    }
    
    // Check if we have the converted weights
    if (access("model_weights", F_OK) == 0) {
        printf("Found converted model weights. Loading...\n");
        return load_vit_model_weights(model);
    }
    
    printf("Error: Model weights not found in 'model_weights/' directory.\n");
    printf("Please run the conversion script to generate the required weight files:\n");
    printf("  python3 download_model.py\n");
    printf("  python3 convert_model.py --input vit_model.pth\n");
    
    return false;
}
