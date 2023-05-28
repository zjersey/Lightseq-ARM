#include "common.h"
#include "float.h"
#include "string.h"
#include <iostream>
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "omp.h"

namespace lightseq {

void quantize(const float *input, int8_t *output, float scale, int size);
void quantize(const int32_t *input, int8_t *output, float scale, int size);

void embedding(const int *tokens, const float *token_emb, const float *pos_emb,
               const float *type_emb, float *output, int seq_len,
               int hidden_dim);
void embedding_int8(const int *tokens, const int8_t *token_emb,
                    const float *pos_emb, float *output, int seq_len,
                    int hidden_dim, float dequant_scale, bool scaled = false);
void decoder_embedding_int8(const int *tokens, const int8_t *token_emb,
                            const float *pos_emb, float *output, int seq_len,
                            int vocab_size, int hidden_dim, float dequant_scale,
                            bool scaled = false);
void layernorm(float *input, const float *gamma, const float *beta,
               int hidden_size, int seq_len);
void gemm_int8(const int8_t *A, const int8_t *B, int8_t *C, int m, int k, int n,
               float scale, int32_t *C_i32 = nullptr);
void multi_head_attention(int8_t *input, const float *project_bias_qkv,
                          int8_t *buffer_qkv_project, float *buffer_qk,
                          int seq_len, int hidden_size, int head_num,
                          float dequant_scale_qkv, float dequant_scale_v);
void dec_self_attention(int8_t *q, int8_t *k, int8_t *v, int8_t *output,
                        float *buffer_qk, int hidden_size, int head_num,
                        int len_kv, float dequant_scale_q,
                        float dequant_scale_kv, float quant_scale_v);
void bias_gelu_i8IO(int8_t *input, const float *residual_bias, int seq_len,
                    int feature_dim, float dequant_scale, float quant_scale);
void bias_relu_i8IO(int8_t *input, const float *bias, int seq_len,
                    int feature_dim, float dequant_scale, float quant_scale,
                    float clip_max);
void residual_bias_ln_i32I(int32_t *input, const float *residual_bias,
                           const float *gamma, const float *beta, float *output,
                           int8_t *output_i8, int feature_dim, int seq_len,
                           float dequant_scale, float quant_scale,
                           bool is_post_ln, const float *colsum = nullptr);
void layernorm_residual_i8O(float *input, int8_t *output, const float *gamma,
                            const float *beta, const float *residual_bias,
                            int hidden_size, int seq_len, float scale,
                            bool is_post_ln);
void scaled_colsum(const int8_t *inp, float *out, int hidden_size,
                   int inner_size, float dequant_scale, float scale);
void gemm_fp32(const float *A, const float *B, float *C, int m, int k, int n);
#if __ARM_NEON
void embedding_int8_arm(const int *tokens, const int8_t *token_emb,
                        const float *pos_emb, float *output, int seq_len,
                        int hidden_dim, float dequant_scale,
                        bool scaled = false);
void layernorm_arm(float *input, const float *gamma, const float *beta,
                   int hidden_size, int seq_len);
void gemm_int8_arm(const int8_t *A, const int8_t *B, int8_t *C, int m, int k,
                   int n, float scale, int32_t *C_i32 = nullptr);
void batch_gemm_int8_arm(const int8_t *A, const int8_t *B, int8_t *C, int m,
                         int k, int n, int sA, int sB, int sC, int num,
                         float scale, int32_t *C_i32);
#endif // __ARM_NEON

} // namespace lightseq
