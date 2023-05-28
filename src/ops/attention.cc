#include "ops.h"

namespace lightseq {
void multi_head_attention(int8_t *input, const float *project_bias_qkv,
                          int8_t *buffer_qkv_project, float *buffer_qk,
                          int seq_len, int hidden_size, int head_num,
                          float dequant_scale_qkv, float dequant_scale_v) {

  int dim_per_head = hidden_size / head_num;
  const float inv_sqrt_dim_per_head = 1.f / sqrt(dim_per_head);

#pragma omp parallel for
  for (int q = 0; q < head_num; ++q) {
    const float *p_bias_qkv = project_bias_qkv + q * dim_per_head;
    // q * k
    for (int i = 0; i < seq_len; ++i) {
      float *p_out = buffer_qk + q * seq_len * seq_len + i * seq_len;
      const int8_t *pq =
          buffer_qkv_project + i * hidden_size * 3 + q * dim_per_head;

      for (int j = 0; j < seq_len; ++j) {

        const int8_t *pk = buffer_qkv_project + j * hidden_size * 3 +
                           hidden_size + q * dim_per_head;
        float sum = 0.f;

        for (int k = 0; k < dim_per_head; ++k) {
          sum +=
              ((float)pq[k] * dequant_scale_qkv + p_bias_qkv[k]) *
              ((float)pk[k] * dequant_scale_qkv + p_bias_qkv[hidden_size + k]);
        }
        p_out[j] = sum * inv_sqrt_dim_per_head;
      }
    }

    // softmax
    for (int i = 0; i < seq_len; ++i) {
      float *ptr = buffer_qk + q * seq_len * seq_len + i * seq_len;

      float vmax = -FLT_MAX;
      for (int j = 0; j < seq_len; ++j) {
        vmax = std::max(vmax, ptr[j]);
      }

      float vsum = 0.f;
      for (int j = 0; j < seq_len; j++) {
        ptr[j] = exp(ptr[j] - vmax);
        vsum += ptr[j];
      }

      for (int j = 0; j < seq_len; j++) {
        ptr[j] /= vsum;
      }
    }

    // score * v
    for (int i = 0; i < seq_len; ++i) {

      float *pscore = buffer_qk + q * seq_len * seq_len + i * seq_len;
      int8_t *pout = input + i * hidden_size + q * dim_per_head;

      for (int j = 0; j < dim_per_head; ++j) {
        int8_t *pv =
            buffer_qkv_project + 2 * hidden_size + q * dim_per_head + j;
        float sum = 0.f;
        float bias = p_bias_qkv[hidden_size * 2 + j];

        for (int k = 0; k < seq_len; ++k) {
          sum += ((float)pv[k * hidden_size * 3] * dequant_scale_qkv + bias) *
                 pscore[k];
        }

        pout[j] = float2int8(sum, dequant_scale_v);
      }
    }
  }
}

void dec_self_attention(int8_t *q, int8_t *k, int8_t *v, int8_t *output,
                        float *buffer_qk, int hidden_size, int head_num,
                        int len_kv, float dequant_scale_q,
                        float dequant_scale_kv, float quant_scale_v) {
  int dim_per_head = hidden_size / head_num;
  const float inv_sqrt_dim_per_head = 1.f / sqrt(dim_per_head);

#pragma omp parallel for
  for (int h = 0; h < head_num; ++h) {

    float *ptr = buffer_qk + h * len_kv;
    float vmax = -FLT_MAX;

    for (int j = 0; j < len_kv; ++j) {
      const int8_t *pq = q + h * dim_per_head;
      const int8_t *pk = k + h * dim_per_head + j * hidden_size;
      float sum = 0.f;

      for (int k = 0; k < dim_per_head; ++k) {
        sum += (float)pq[k] * (float)pk[k];
      }
      sum *= dequant_scale_q * dequant_scale_kv * inv_sqrt_dim_per_head;
      vmax = std::max(vmax, sum);
      ptr[j] = sum;
    }

    // softmax
    float vsum = 0.f;
    for (int j = 0; j < len_kv; j++) {
      ptr[j] = exp(ptr[j] - vmax);
      vsum += ptr[j];
    }

    for (int j = 0; j < len_kv; j++) {
      ptr[j] /= vsum;
    }

    // score * v
    int8_t *pout = output + h * dim_per_head;
    for (int j = 0; j < dim_per_head; ++j) {
      float sum = 0.f;

      for (int k = 0; k < len_kv; ++k) {
        sum += ptr[k] * (float)v[k * hidden_size + h * dim_per_head + j];
      }
      sum *= dequant_scale_kv;
      pout[j] = float2int8(sum, quant_scale_v);
    }
  }
}

} // namespace lightseq
