#include "../include/vit.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef USE_FP16
// FP16 conversion functions
vit_fp16_t float_to_fp16(float f) {
    // Simple FP16 conversion (IEEE 754 half precision)
    // This is a basic implementation - for production use, consider using hardware intrinsics
    uint32_t bits = *(uint32_t*)&f;
    uint32_t sign = bits & 0x80000000;
    uint32_t exp = bits & 0x7F800000;
    uint32_t mantissa = bits & 0x007FFFFF;
    
    // Handle special cases
    if (exp == 0x7F800000) {
        // Infinity or NaN
        return (sign >> 16) | 0x7C00 | (mantissa ? 0x0200 : 0);
    }
    
    if (exp == 0) {
        // Zero or denormalized
        return sign >> 16;
    }
    
    // Normalized number
    int32_t fp16_exp = (exp >> 23) - 127 + 15;
    
    if (fp16_exp <= 0) {
        // Underflow to zero
        return sign >> 16;
    }
    
    if (fp16_exp >= 31) {
        // Overflow to infinity
        return (sign >> 16) | 0x7C00;
    }
    
    uint32_t fp16_mantissa = mantissa >> 13;
    if (mantissa & 0x1000) {
        // Round up
        fp16_mantissa++;
        if (fp16_mantissa & 0x0400) {
            fp16_mantissa = 0;
            fp16_exp++;
            if (fp16_exp >= 31) {
                return (sign >> 16) | 0x7C00;
            }
        }
    }
    
    return (sign >> 16) | (fp16_exp << 10) | (fp16_mantissa & 0x03FF);
}

float fp16_to_float(vit_fp16_t h) {
    // Simple FP16 to FP32 conversion
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mantissa = h & 0x03FF;
    
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            return *(float*)&sign;
        } else {
            // Denormalized
            uint32_t fp32_exp = 127 - 15 + 1;
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                fp32_exp--;
            }
            mantissa &= 0x03FF;
            uint32_t bits = sign | (fp32_exp << 23) | (mantissa << 13);
            return *(float*)&bits;
        }
    } else if (exp == 31) {
        // Infinity or NaN
        uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&bits;
    } else {
        // Normalized
        uint32_t fp32_exp = (exp - 15 + 127) << 23;
        uint32_t bits = sign | fp32_exp | (mantissa << 13);
        return *(float*)&bits;
    }
}
#endif

// Matrix operations
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (vit_float_t*)calloc(rows * cols, sizeof(vit_float_t));
    if (!m.data) {
        fprintf(stderr, "Error: Failed to allocate matrix memory\n");
        exit(1);
    }
    return m;
}

void free_matrix(Matrix* m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->rows = 0;
        m->cols = 0;
    }
}

Matrix matrix_multiply(const Matrix* a, const Matrix* b) {
    assert(a->cols == b->rows);
    Matrix result = create_matrix(a->rows, b->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            vit_float_t sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result.data[i * b->cols + j] = sum;
        }
    }
    return result;
}

Matrix matrix_add(const Matrix* a, const Matrix* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Matrix result = create_matrix(a->rows, a->cols);
    
    for (int i = 0; i < a->rows * a->cols; i++) {
        result.data[i] = a->data[i] + b->data[i];
    }
    return result;
}

// Activation functions
Matrix gelu(const Matrix* m) {
    Matrix result = create_matrix(m->rows, m->cols);
    
    for (int i = 0; i < m->rows * m->cols; i++) {
        vit_float_t x = m->data[i];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        vit_float_t inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        result.data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    return result;
}

// Layer normalization
Matrix layer_norm(const Matrix* m, const Matrix* gamma, const Matrix* beta) {
    assert(m->cols == gamma->rows && m->cols == beta->rows);
    Matrix result = create_matrix(m->rows, m->cols);
    
    for (int i = 0; i < m->rows; i++) {
        // Calculate mean
        vit_float_t mean = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            mean += m->data[i * m->cols + j];
        }
        mean /= m->cols;
        
        // Calculate variance
        vit_float_t var = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            vit_float_t diff = m->data[i * m->cols + j] - mean;
            var += diff * diff;
        }
        var /= m->cols;
        vit_float_t std = sqrtf(var + 1e-5f);
        
        // Normalize and scale
        for (int j = 0; j < m->cols; j++) {
            vit_float_t normalized = (m->data[i * m->cols + j] - mean) / std;
            result.data[i * m->cols + j] = gamma->data[j] * normalized + beta->data[j];
        }
    }
    return result;
}

// Multi-head attention
Matrix multi_head_attention(const Matrix* x, const Matrix* qkv_weight, const Matrix* qkv_bias,
                           const Matrix* proj_weight, const Matrix* proj_bias, int num_heads) {
    int seq_len = x->rows;
    int embed_dim = x->cols;
    int head_dim = embed_dim / num_heads;
    
    // Linear projection to Q, K, V (input @ qkv_weight.T)
    Matrix qkv_weight_T = create_matrix(qkv_weight->cols, qkv_weight->rows);
    for (int i = 0; i < qkv_weight->rows; i++) {
        for (int j = 0; j < qkv_weight->cols; j++) {
            qkv_weight_T.data[j * qkv_weight->rows + i] = qkv_weight->data[i * qkv_weight->cols + j];
        }
    }
    Matrix qkv = matrix_multiply(x, &qkv_weight_T);
    free_matrix(&qkv_weight_T);
    
    // Add bias
    for (int i = 0; i < qkv.rows * qkv.cols; i++) {
        qkv.data[i] += qkv_bias->data[i % qkv_bias->rows];
    }
    
    // QKV computation completed
    
    // Reshape and compute attention for each head
    Matrix result = create_matrix(seq_len, embed_dim);
    
    for (int head = 0; head < num_heads; head++) {
        int start_idx = head * head_dim;
        int end_idx = start_idx + head_dim;
        
        // Extract Q, K, V for this head
        Matrix Q = create_matrix(seq_len, head_dim);
        Matrix K = create_matrix(seq_len, head_dim);
        Matrix V = create_matrix(seq_len, head_dim);
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q.data[i * head_dim + j] = qkv.data[i * qkv.cols + start_idx + j];
                K.data[i * head_dim + j] = qkv.data[i * qkv.cols + embed_dim + start_idx + j];
                V.data[i * head_dim + j] = qkv.data[i * qkv.cols + 2 * embed_dim + start_idx + j];
            }
        }
        
        // QKV extraction completed
        
        
        // K matrix extraction completed
        
        // Compute attention scores: Q * K^T / sqrt(head_dim)
        // First transpose K: K_T[j, i] = K[i, j]
        Matrix K_T = create_matrix(K.cols, K.rows);
        for (int i = 0; i < K.rows; i++) {
            for (int j = 0; j < K.cols; j++) {
                // K_T has shape [K.cols, K.rows] = [64, 197]
                // K_T[j, i] = K_T.data[j * K_T.cols + i]
                K_T.data[j * K_T.cols + i] = K.data[i * K.cols + j];
            }
        }
        
        // K transpose completed
        
        Matrix attention_scores = matrix_multiply(&Q, &K_T);
        free_matrix(&K_T);
        
        vit_float_t scale = 1.0f / sqrtf((vit_float_t)head_dim);
        for (int i = 0; i < attention_scores.rows * attention_scores.cols; i++) {
            attention_scores.data[i] *= scale;
        }
        
        
        // Apply softmax (without max shift)
        for (int i = 0; i < seq_len; i++) {
            vit_float_t sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                attention_scores.data[i * seq_len + j] = expf(attention_scores.data[i * seq_len + j]);
                sum += attention_scores.data[i * seq_len + j];
            }
            
            for (int j = 0; j < seq_len; j++) {
                attention_scores.data[i * seq_len + j] /= sum;
            }
        }
        
        
        // Apply attention to values
        Matrix attention_output = matrix_multiply(&attention_scores, &V);
        
        
        // Copy back to result
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                result.data[i * embed_dim + start_idx + j] = attention_output.data[i * head_dim + j];
            }
        }
        
        free_matrix(&Q);
        free_matrix(&K);
        free_matrix(&V);
        free_matrix(&attention_scores);
        free_matrix(&attention_output);
    }
    
    // Final projection
    Matrix final_output = matrix_multiply(&result, proj_weight);
    for (int i = 0; i < final_output.rows * final_output.cols; i++) {
        final_output.data[i] += proj_bias->data[i % proj_bias->rows];
    }
    
    free_matrix(&qkv);
    free_matrix(&result);
    
    return final_output;
}

// MLP block
Matrix mlp_block(const Matrix* x, const Matrix* weight1, const Matrix* bias1,
                const Matrix* weight2, const Matrix* bias2) {
    Matrix hidden = matrix_multiply(x, weight1);
    
    // Add bias
    for (int i = 0; i < hidden.rows * hidden.cols; i++) {
        hidden.data[i] += bias1->data[i % bias1->rows];
    }
    
    // Apply GELU
    Matrix activated = gelu(&hidden);
    free_matrix(&hidden);
    
    // Second linear layer
    Matrix output = matrix_multiply(&activated, weight2);
    free_matrix(&activated);
    
    // Add bias
    for (int i = 0; i < output.rows * output.cols; i++) {
        output.data[i] += bias2->data[i % bias2->rows];
    }
    
    return output;
}

#ifdef USE_FP16
// FP16 Matrix operations
MatrixFP16 create_matrix_fp16(int rows, int cols) {
    MatrixFP16 m;
    m.rows = rows;
    m.cols = cols;
    m.data = (vit_fp16_t*)calloc(rows * cols, sizeof(vit_fp16_t));
    if (!m.data) {
        fprintf(stderr, "Error: Failed to allocate FP16 matrix memory\n");
        exit(1);
    }
    return m;
}

void free_matrix_fp16(MatrixFP16* m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->rows = 0;
        m->cols = 0;
    }
}

MatrixFP16 matrix_multiply_fp16(const MatrixFP16* a, const MatrixFP16* b) {
    assert(a->cols == b->rows);
    MatrixFP16 result = create_matrix_fp16(a->rows, b->cols);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += fp16_to_float(a->data[i * a->cols + k]) * fp16_to_float(b->data[k * b->cols + j]);
            }
            result.data[i * b->cols + j] = float_to_fp16(sum);
        }
    }
    return result;
}

MatrixFP16 matrix_add_fp16(const MatrixFP16* a, const MatrixFP16* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    MatrixFP16 result = create_matrix_fp16(a->rows, a->cols);
    
    for (int i = 0; i < a->rows * a->cols; i++) {
        float sum = fp16_to_float(a->data[i]) + fp16_to_float(b->data[i]);
        result.data[i] = float_to_fp16(sum);
    }
    return result;
}

MatrixFP16 gelu_fp16(const MatrixFP16* m) {
    MatrixFP16 result = create_matrix_fp16(m->rows, m->cols);
    
    for (int i = 0; i < m->rows * m->cols; i++) {
        float x = fp16_to_float(m->data[i]);
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        float gelu_val = 0.5f * x * (1.0f + tanhf(inner));
        result.data[i] = float_to_fp16(gelu_val);
    }
    return result;
}

MatrixFP16 layer_norm_fp16(const MatrixFP16* m, const MatrixFP16* gamma, const MatrixFP16* beta) {
    assert(m->cols == gamma->rows && m->cols == beta->rows);
    MatrixFP16 result = create_matrix_fp16(m->rows, m->cols);
    
    for (int i = 0; i < m->rows; i++) {
        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            mean += fp16_to_float(m->data[i * m->cols + j]);
        }
        mean /= m->cols;
        
        // Calculate variance
        float var = 0.0f;
        for (int j = 0; j < m->cols; j++) {
            float diff = fp16_to_float(m->data[i * m->cols + j]) - mean;
            var += diff * diff;
        }
        var /= m->cols;
        float std = sqrtf(var + 1e-5f);
        
        // Normalize and scale
        for (int j = 0; j < m->cols; j++) {
            float normalized = (fp16_to_float(m->data[i * m->cols + j]) - mean) / std;
            float gamma_val = fp16_to_float(gamma->data[j]);
            float beta_val = fp16_to_float(beta->data[j]);
            result.data[i * m->cols + j] = float_to_fp16(gamma_val * normalized + beta_val);
        }
    }
    return result;
}

MatrixFP16 multi_head_attention_fp16(const MatrixFP16* x, const MatrixFP16* qkv_weight, const MatrixFP16* qkv_bias,
                                    const MatrixFP16* proj_weight, const MatrixFP16* proj_bias, int num_heads) {
    int seq_len = x->rows;
    int embed_dim = x->cols;
    int head_dim = embed_dim / num_heads;
    
    // Linear projection to Q, K, V (input @ qkv_weight.T)
    MatrixFP16 qkv_weight_T = create_matrix_fp16(qkv_weight->cols, qkv_weight->rows);
    for (int i = 0; i < qkv_weight->rows; i++) {
        for (int j = 0; j < qkv_weight->cols; j++) {
            qkv_weight_T.data[j * qkv_weight->rows + i] = qkv_weight->data[i * qkv_weight->cols + j];
        }
    }
    MatrixFP16 qkv = matrix_multiply_fp16(x, &qkv_weight_T);
    free_matrix_fp16(&qkv_weight_T);
    
    // Add bias
    for (int i = 0; i < qkv.rows * qkv.cols; i++) {
        float sum = fp16_to_float(qkv.data[i]) + fp16_to_float(qkv_bias->data[i % qkv_bias->rows]);
        qkv.data[i] = float_to_fp16(sum);
    }
    
    // Reshape and compute attention for each head
    MatrixFP16 result = create_matrix_fp16(seq_len, embed_dim);
    
    for (int head = 0; head < num_heads; head++) {
        int start_idx = head * head_dim;
        int end_idx = start_idx + head_dim;
        
        // Extract Q, K, V for this head
        MatrixFP16 Q = create_matrix_fp16(seq_len, head_dim);
        MatrixFP16 K = create_matrix_fp16(seq_len, head_dim);
        MatrixFP16 V = create_matrix_fp16(seq_len, head_dim);
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q.data[i * head_dim + j] = qkv.data[i * qkv.cols + start_idx + j];
                K.data[i * head_dim + j] = qkv.data[i * qkv.cols + embed_dim + start_idx + j];
                V.data[i * head_dim + j] = qkv.data[i * qkv.cols + 2 * embed_dim + start_idx + j];
            }
        }
        
        // Compute attention scores: Q * K^T / sqrt(head_dim)
        MatrixFP16 K_T = create_matrix_fp16(K.cols, K.rows);
        for (int i = 0; i < K.rows; i++) {
            for (int j = 0; j < K.cols; j++) {
                K_T.data[j * K_T.cols + i] = K.data[i * K.cols + j];
            }
        }
        
        MatrixFP16 attention_scores = matrix_multiply_fp16(&Q, &K_T);
        free_matrix_fp16(&K_T);
        
        float scale = 1.0f / sqrtf((float)head_dim);
        for (int i = 0; i < attention_scores.rows * attention_scores.cols; i++) {
            float scaled = fp16_to_float(attention_scores.data[i]) * scale;
            attention_scores.data[i] = float_to_fp16(scaled);
        }
        
        
        // Apply softmax (without max shift)
        for (int i = 0; i < seq_len; i++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                float val = exp_table_fp32(fp16_to_float(attention_scores.data[i * seq_len + j]));
                attention_scores.data[i * seq_len + j] = float_to_fp16(val);
                sum += val;
            }
            
            for (int j = 0; j < seq_len; j++) {
                float val = fp16_to_float(attention_scores.data[i * seq_len + j]) / sum;
                attention_scores.data[i * seq_len + j] = float_to_fp16(val);
            }
        }
        
        
        // Apply attention to values
        MatrixFP16 attention_output = matrix_multiply_fp16(&attention_scores, &V);
        
        // Copy back to result
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                result.data[i * embed_dim + start_idx + j] = attention_output.data[i * head_dim + j];
            }
        }
        
        free_matrix_fp16(&Q);
        free_matrix_fp16(&K);
        free_matrix_fp16(&V);
        free_matrix_fp16(&attention_scores);
        free_matrix_fp16(&attention_output);
    }
    
    // Final projection
    MatrixFP16 final_output = matrix_multiply_fp16(&result, proj_weight);
    for (int i = 0; i < final_output.rows * final_output.cols; i++) {
        float sum = fp16_to_float(final_output.data[i]) + fp16_to_float(proj_bias->data[i % proj_bias->rows]);
        final_output.data[i] = float_to_fp16(sum);
    }
    
    free_matrix_fp16(&qkv);
    free_matrix_fp16(&result);
    
    return final_output;
}

MatrixFP16 mlp_block_fp16(const MatrixFP16* x, const MatrixFP16* weight1, const MatrixFP16* bias1,
                         const MatrixFP16* weight2, const MatrixFP16* bias2) {
    MatrixFP16 hidden = matrix_multiply_fp16(x, weight1);
    
    // Add bias
    for (int i = 0; i < hidden.rows * hidden.cols; i++) {
        float sum = fp16_to_float(hidden.data[i]) + fp16_to_float(bias1->data[i % bias1->rows]);
        hidden.data[i] = float_to_fp16(sum);
    }
    
    // Apply GELU
    MatrixFP16 activated = gelu_fp16(&hidden);
    free_matrix_fp16(&hidden);
    
    // Second linear layer
    MatrixFP16 output = matrix_multiply_fp16(&activated, weight2);
    free_matrix_fp16(&activated);
    
    // Add bias
    for (int i = 0; i < output.rows * output.cols; i++) {
        float sum = fp16_to_float(output.data[i]) + fp16_to_float(bias2->data[i % bias2->rows]);
        output.data[i] = float_to_fp16(sum);
    }
    
    return output;
}

// Conversion functions
MatrixFP16 matrix_to_fp16(const Matrix* m) {
    MatrixFP16 result = create_matrix_fp16(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        result.data[i] = float_to_fp16(m->data[i]);
    }
    return result;
}

Matrix matrix_to_fp32(const MatrixFP16* m) {
    Matrix result = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        result.data[i] = fp16_to_float(m->data[i]);
    }
    return result;
}

// Exponential lookup table implementation
static vit_fp16_t exp_table_fp16_data[EXP_TABLE_SIZE];
static bool exp_table_initialized = false;

void init_exp_table_fp16() {
    if (exp_table_initialized) return;
    
    printf("Initializing exponential lookup table (FP16)...\n");
    printf("Range: [%.3f, %.3f], Step: %.3f, Size: %d\n", 
           EXP_TABLE_MIN, EXP_TABLE_MAX, EXP_TABLE_STEP, EXP_TABLE_SIZE);
    
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        float x = EXP_TABLE_MIN + i * EXP_TABLE_STEP;
        float exp_val = expf(x);
        exp_table_fp16_data[i] = float_to_fp16(exp_val);
    }
    
    exp_table_initialized = true;
    printf("Exponential lookup table initialized successfully!\n");
}

vit_fp16_t exp_table_fp16(vit_fp16_t x) {
    if (!exp_table_initialized) {
        init_exp_table_fp16();
    }
    
    float x_fp32 = fp16_to_float(x);
    
    // Clamp values outside the table range
    if (x_fp32 < EXP_TABLE_MIN) {
        x_fp32 = EXP_TABLE_MIN;
    } else if (x_fp32 > EXP_TABLE_MAX) {
        x_fp32 = EXP_TABLE_MAX;
    }
    
    // Calculate table index
    int index = (int)((x_fp32 - EXP_TABLE_MIN) / EXP_TABLE_STEP + 0.5f);
    
    // Ensure index is within bounds
    if (index < 0) index = 0;
    if (index >= EXP_TABLE_SIZE) index = EXP_TABLE_SIZE - 1;
    
    return exp_table_fp16_data[index];
}

float exp_table_fp32(float x) {
    if (!exp_table_initialized) {
        init_exp_table_fp16();
    }
    
    // Clamp values outside the table range
    if (x < EXP_TABLE_MIN) {
        x = EXP_TABLE_MIN;
    } else if (x > EXP_TABLE_MAX) {
        x = EXP_TABLE_MAX;
    }
    
    // Calculate table index
    int index = (int)((x - EXP_TABLE_MIN) / EXP_TABLE_STEP + 0.5f);
    
    // Ensure index is within bounds
    if (index < 0) index = 0;
    if (index >= EXP_TABLE_SIZE) index = EXP_TABLE_SIZE - 1;
    
    return fp16_to_float(exp_table_fp16_data[index]);
}

vit_fp16_t get_exp_table_entry(int index) {
    if (!exp_table_initialized) {
        init_exp_table_fp16();
    }
    
    if (index < 0 || index >= EXP_TABLE_SIZE) {
        return 0;  // Return zero for invalid index
    }
    
    return exp_table_fp16_data[index];
}

#endif

// ViT model creation and management
ViTModel* create_vit_model() {
    ViTModel* model = (ViTModel*)malloc(sizeof(ViTModel));
    if (!model) {
        fprintf(stderr, "Error: Failed to allocate ViT model\n");
        return NULL;
    }
    
    // Initialize embeddings
    model->patch_embed_weight = create_matrix(PATCH_SIZE * PATCH_SIZE * 3, EMBED_DIM);
    model->patch_embed_bias = create_matrix(EMBED_DIM, 1);
    model->pos_embed = create_matrix(NUM_PATCHES + 1, EMBED_DIM);
    model->cls_token = create_matrix(1, EMBED_DIM);
    
    // CLS token will be loaded from trained model weights
    // No initialization needed - will be overwritten by load_vit_model()
    
    // Initialize transformer layers
    model->layer_norm1_weight = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->layer_norm1_bias = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->layer_norm2_weight = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->layer_norm2_bias = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->attention_weights = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->attention_bias = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->qkv_weights = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->qkv_bias = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->proj_weights = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->proj_bias = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->mlp_weights1 = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->mlp_bias1 = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->mlp_weights2 = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    model->mlp_bias2 = (Matrix*)malloc(NUM_LAYERS * sizeof(Matrix));
    
    for (int i = 0; i < NUM_LAYERS; i++) {
        model->layer_norm1_weight[i] = create_matrix(EMBED_DIM, 1);
        model->layer_norm1_bias[i] = create_matrix(EMBED_DIM, 1);
        model->layer_norm2_weight[i] = create_matrix(EMBED_DIM, 1);
        model->layer_norm2_bias[i] = create_matrix(EMBED_DIM, 1);
        model->attention_weights[i] = create_matrix(EMBED_DIM, EMBED_DIM);
        model->attention_bias[i] = create_matrix(EMBED_DIM, 1);
        model->qkv_weights[i] = create_matrix(3 * EMBED_DIM, EMBED_DIM);
        model->qkv_bias[i] = create_matrix(3 * EMBED_DIM, 1);
        model->proj_weights[i] = create_matrix(EMBED_DIM, EMBED_DIM);
        model->proj_bias[i] = create_matrix(EMBED_DIM, 1);
        model->mlp_weights1[i] = create_matrix(EMBED_DIM, MLP_DIM);
        model->mlp_bias1[i] = create_matrix(MLP_DIM, 1);
        model->mlp_weights2[i] = create_matrix(MLP_DIM, EMBED_DIM);
        model->mlp_bias2[i] = create_matrix(EMBED_DIM, 1);
    }
    
    // Initialize final layers
    model->final_norm_weight = create_matrix(EMBED_DIM, 1);
    model->final_norm_bias = create_matrix(EMBED_DIM, 1);
    model->head_weights = create_matrix(EMBED_DIM, NUM_CLASSES);
    model->head_bias = create_matrix(NUM_CLASSES, 1);
    
    return model;
}

void free_vit_model(ViTModel* model) {
    if (!model) return;
    
    free_matrix(&model->patch_embed_weight);
    free_matrix(&model->patch_embed_bias);
    free_matrix(&model->pos_embed);
    free_matrix(&model->cls_token);
    
    if (model->layer_norm1_weight) {
        for (int i = 0; i < NUM_LAYERS; i++) {
            free_matrix(&model->layer_norm1_weight[i]);
            free_matrix(&model->layer_norm1_bias[i]);
            free_matrix(&model->layer_norm2_weight[i]);
            free_matrix(&model->layer_norm2_bias[i]);
            free_matrix(&model->attention_weights[i]);
            free_matrix(&model->attention_bias[i]);
            free_matrix(&model->qkv_weights[i]);
            free_matrix(&model->qkv_bias[i]);
            free_matrix(&model->proj_weights[i]);
            free_matrix(&model->proj_bias[i]);
            free_matrix(&model->mlp_weights1[i]);
            free_matrix(&model->mlp_bias1[i]);
            free_matrix(&model->mlp_weights2[i]);
            free_matrix(&model->mlp_bias2[i]);
        }
        free(model->layer_norm1_weight);
        free(model->layer_norm1_bias);
        free(model->layer_norm2_weight);
        free(model->layer_norm2_bias);
        free(model->attention_weights);
        free(model->attention_bias);
        free(model->qkv_weights);
        free(model->qkv_bias);
        free(model->proj_weights);
        free(model->proj_bias);
        free(model->mlp_weights1);
        free(model->mlp_bias1);
        free(model->mlp_weights2);
        free(model->mlp_bias2);
    }
    
    free_matrix(&model->final_norm_weight);
    free_matrix(&model->final_norm_bias);
    free_matrix(&model->head_weights);
    free_matrix(&model->head_bias);
    
    free(model);
}

// Patch embedding
Matrix patch_embedding(const vit_float_t* image, const Matrix* patch_weight, const Matrix* patch_bias, const Matrix* cls_token) {
    Matrix patches = create_matrix(NUM_PATCHES + 1, EMBED_DIM);
    
    // Add CLS token (learnable parameter with proper initialization)
    for (int i = 0; i < EMBED_DIM; i++) {
        patches.data[i] = cls_token->data[i];
    }
    
    // Extract patches and embed them
    for (int patch_idx = 0; patch_idx < NUM_PATCHES; patch_idx++) {
        int patch_row = patch_idx / (IMG_SIZE / PATCH_SIZE);
        int patch_col = patch_idx % (IMG_SIZE / PATCH_SIZE);
        
        // Create patch vector
        vit_float_t patch_vector[PATCH_SIZE * PATCH_SIZE * 3];
        for (int i = 0; i < PATCH_SIZE; i++) {
            for (int j = 0; j < PATCH_SIZE; j++) {
                int img_row = patch_row * PATCH_SIZE + i;
                int img_col = patch_col * PATCH_SIZE + j;
                int patch_pixel_idx = i * PATCH_SIZE + j;
                int img_pixel_idx = img_row * IMG_SIZE + img_col;
                
                // RGB channels (CHW format: channel first, then spatial)
                // PyTorch patch vector format: [R0, R1, R2, ..., G0, G1, G2, ..., B0, B1, B2, ...]
                for (int c = 0; c < 3; c++) {
                    // CHW format: image[c * IMG_SIZE * IMG_SIZE + img_row * IMG_SIZE + img_col]
                    int chw_idx = c * IMG_SIZE * IMG_SIZE + img_row * IMG_SIZE + img_col;
                    // Organize as [R0, R1, R2, ..., G0, G1, G2, ..., B0, B1, B2, ...]
                    patch_vector[c * PATCH_SIZE * PATCH_SIZE + patch_pixel_idx] = image[chw_idx];
                }
            }
        }
        
        // Patch extraction completed
        
        // Linear transformation: patch_vector * patch_weight + patch_bias
        for (int emb_dim = 0; emb_dim < EMBED_DIM; emb_dim++) {
            vit_float_t sum = 0.0f;
            for (int patch_dim = 0; patch_dim < PATCH_SIZE * PATCH_SIZE * 3; patch_dim++) {
                sum += patch_vector[patch_dim] * patch_weight->data[patch_dim * EMBED_DIM + emb_dim];
            }
            patches.data[(patch_idx + 1) * EMBED_DIM + emb_dim] = sum + patch_bias->data[emb_dim];
        }
        
        // Patch embedding computation completed
    }
    
    // Patch embedding completed successfully
    
    return patches;
}

// ViT forward pass
Matrix vit_forward(ViTModel* model, const vit_float_t* image) {
    Matrix x;
    
    // Patch embedding
    x = patch_embedding(image, &model->patch_embed_weight, &model->patch_embed_bias, &model->cls_token);
    
    // Add positional embedding
    for (int i = 0; i < x.rows * x.cols; i++) {
        x.data[i] += model->pos_embed.data[i];
    }
    // Transformer blocks
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        
        // Layer norm 1
        Matrix norm1 = layer_norm(&x, &model->layer_norm1_weight[layer], &model->layer_norm1_bias[layer]);
        
        // Multi-head attention
        Matrix attention = multi_head_attention(&norm1, &model->qkv_weights[layer], 
                                              &model->qkv_bias[layer], &model->proj_weights[layer], 
                                              &model->proj_bias[layer], NUM_HEADS);
        
        free_matrix(&norm1);
        
        // Residual connection
        Matrix x_plus_attn = matrix_add(&x, &attention);
        free_matrix(&attention);
        
        
        
        
        
        // Layer norm 2
        Matrix norm2 = layer_norm(&x_plus_attn, &model->layer_norm2_weight[layer], &model->layer_norm2_bias[layer]);
        
        
        // MLP block
        Matrix mlp_out = mlp_block(&norm2, &model->mlp_weights1[layer], &model->mlp_bias1[layer],
                                  &model->mlp_weights2[layer], &model->mlp_bias2[layer]);
        
        
        free_matrix(&norm2);
        
        
        
        
        // Residual connection
        Matrix x_plus_mlp = matrix_add(&x_plus_attn, &mlp_out);
        
        
        free_matrix(&x_plus_attn);  // Free x_plus_attn after debug code
        
        free_matrix(&mlp_out);
        
        // Update x for next layer
        free_matrix(&x);
        x = x_plus_mlp;
    }
    
    // Final layer norm
    Matrix final_norm = layer_norm(&x, &model->final_norm_weight, &model->final_norm_bias);
    
    
    free_matrix(&x);
    
    // Classification head (use CLS token)
    Matrix cls_output = create_matrix(1, EMBED_DIM);
    memcpy(cls_output.data, final_norm.data, EMBED_DIM * sizeof(vit_float_t));
    free_matrix(&final_norm);
    
    Matrix logits = matrix_multiply(&cls_output, &model->head_weights);
    free_matrix(&cls_output);
    
    // Add bias
    for (int i = 0; i < logits.cols; i++) {
        logits.data[i] += model->head_bias.data[i];
    }
    
    return logits;
}

#ifdef USE_FP16
// FP16 forward pass - uses FP16 for transformer encoder computations
Matrix vit_forward_fp16(ViTModel* model, const vit_float_t* image) {
    Matrix x;
    
    // Patch embedding (keep in FP32 for input processing)
    x = patch_embedding(image, &model->patch_embed_weight, &model->patch_embed_bias, &model->cls_token);
    
    // Add positional embedding
    for (int i = 0; i < x.rows * x.cols; i++) {
        x.data[i] += model->pos_embed.data[i];
    }
    
    // Convert to FP16 for transformer computations
    MatrixFP16 x_fp16 = matrix_to_fp16(&x);
    free_matrix(&x);
    
    // Convert model weights to FP16 for transformer layers
    MatrixFP16* layer_norm1_weight_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* layer_norm1_bias_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* layer_norm2_weight_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* layer_norm2_bias_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* qkv_weights_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* qkv_bias_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* proj_weights_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* proj_bias_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* mlp_weights1_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* mlp_bias1_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* mlp_weights2_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    MatrixFP16* mlp_bias2_fp16 = (MatrixFP16*)malloc(NUM_LAYERS * sizeof(MatrixFP16));
    
    for (int i = 0; i < NUM_LAYERS; i++) {
        layer_norm1_weight_fp16[i] = matrix_to_fp16(&model->layer_norm1_weight[i]);
        layer_norm1_bias_fp16[i] = matrix_to_fp16(&model->layer_norm1_bias[i]);
        layer_norm2_weight_fp16[i] = matrix_to_fp16(&model->layer_norm2_weight[i]);
        layer_norm2_bias_fp16[i] = matrix_to_fp16(&model->layer_norm2_bias[i]);
        qkv_weights_fp16[i] = matrix_to_fp16(&model->qkv_weights[i]);
        qkv_bias_fp16[i] = matrix_to_fp16(&model->qkv_bias[i]);
        proj_weights_fp16[i] = matrix_to_fp16(&model->proj_weights[i]);
        proj_bias_fp16[i] = matrix_to_fp16(&model->proj_bias[i]);
        mlp_weights1_fp16[i] = matrix_to_fp16(&model->mlp_weights1[i]);
        mlp_bias1_fp16[i] = matrix_to_fp16(&model->mlp_bias1[i]);
        mlp_weights2_fp16[i] = matrix_to_fp16(&model->mlp_weights2[i]);
        mlp_bias2_fp16[i] = matrix_to_fp16(&model->mlp_bias2[i]);
    }
    
    // Transformer blocks (using FP16)
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        // Layer norm 1
        MatrixFP16 norm1 = layer_norm_fp16(&x_fp16, &layer_norm1_weight_fp16[layer], &layer_norm1_bias_fp16[layer]);
        
        // Multi-head attention
        MatrixFP16 attention = multi_head_attention_fp16(&norm1, &qkv_weights_fp16[layer], 
                                                        &qkv_bias_fp16[layer], &proj_weights_fp16[layer], 
                                                        &proj_bias_fp16[layer], NUM_HEADS);
        
        free_matrix_fp16(&norm1);
        
        // Residual connection
        MatrixFP16 x_plus_attn = matrix_add_fp16(&x_fp16, &attention);
        free_matrix_fp16(&attention);
        
        // Layer norm 2
        MatrixFP16 norm2 = layer_norm_fp16(&x_plus_attn, &layer_norm2_weight_fp16[layer], &layer_norm2_bias_fp16[layer]);
        
        // MLP block
        MatrixFP16 mlp_out = mlp_block_fp16(&norm2, &mlp_weights1_fp16[layer], &mlp_bias1_fp16[layer],
                                           &mlp_weights2_fp16[layer], &mlp_bias2_fp16[layer]);
        
        free_matrix_fp16(&norm2);
        
        // Residual connection
        MatrixFP16 x_plus_mlp = matrix_add_fp16(&x_plus_attn, &mlp_out);
        
        free_matrix_fp16(&x_plus_attn);
        free_matrix_fp16(&mlp_out);
        
        // Update x for next layer
        free_matrix_fp16(&x_fp16);
        x_fp16 = x_plus_mlp;
    }
    
    // Convert back to FP32 for final processing
    Matrix x_final = matrix_to_fp32(&x_fp16);
    free_matrix_fp16(&x_fp16);
    
    // Final layer norm (FP32)
    Matrix final_norm = layer_norm(&x_final, &model->final_norm_weight, &model->final_norm_bias);
    free_matrix(&x_final);
    
    // Classification head (FP32)
    Matrix cls_output = create_matrix(1, EMBED_DIM);
    memcpy(cls_output.data, final_norm.data, EMBED_DIM * sizeof(vit_float_t));
    free_matrix(&final_norm);
    
    Matrix logits = matrix_multiply(&cls_output, &model->head_weights);
    free_matrix(&cls_output);
    
    // Add bias
    for (int i = 0; i < logits.cols; i++) {
        logits.data[i] += model->head_bias.data[i];
    }
    
    // Clean up FP16 weight matrices
    for (int i = 0; i < NUM_LAYERS; i++) {
        free_matrix_fp16(&layer_norm1_weight_fp16[i]);
        free_matrix_fp16(&layer_norm1_bias_fp16[i]);
        free_matrix_fp16(&layer_norm2_weight_fp16[i]);
        free_matrix_fp16(&layer_norm2_bias_fp16[i]);
        free_matrix_fp16(&qkv_weights_fp16[i]);
        free_matrix_fp16(&qkv_bias_fp16[i]);
        free_matrix_fp16(&proj_weights_fp16[i]);
        free_matrix_fp16(&proj_bias_fp16[i]);
        free_matrix_fp16(&mlp_weights1_fp16[i]);
        free_matrix_fp16(&mlp_bias1_fp16[i]);
        free_matrix_fp16(&mlp_weights2_fp16[i]);
        free_matrix_fp16(&mlp_bias2_fp16[i]);
    }
    free(layer_norm1_weight_fp16);
    free(layer_norm1_bias_fp16);
    free(layer_norm2_weight_fp16);
    free(layer_norm2_bias_fp16);
    free(qkv_weights_fp16);
    free(qkv_bias_fp16);
    free(proj_weights_fp16);
    free(proj_bias_fp16);
    free(mlp_weights1_fp16);
    free(mlp_bias1_fp16);
    free(mlp_weights2_fp16);
    free(mlp_bias2_fp16);
    
    return logits;
}
#endif

// Utility functions
void print_matrix(const Matrix* m, const char* name) {
    printf("%s (%dx%d):\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.4f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// normalize_image function is defined in image_processing_improved.c
