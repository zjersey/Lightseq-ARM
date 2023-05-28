#include "ops.h"

namespace lightseq {

void layernorm(float *input, const float *gamma, const float *beta,
               int hidden_size, int seq_len) {

#pragma omp parallel for
  for (int i = 0; i < seq_len; ++i) {
    float *ptr = input + i * hidden_size;

    // mean and var
    float sum = 0.f;
    float sqsum = 0.f;

    for (int j = 0; j < hidden_size; ++j) {
      sum += ptr[j];
    }
    float mean = sum / hidden_size;
    float tmp = 0.f;

    for (int j = 0; j < hidden_size; ++j) {
      tmp = ptr[j] - mean;
      sqsum += tmp * tmp;
    }

    float var = sqsum / hidden_size;

    float a = static_cast<float>(1.f / (sqrt(var + epsilon)));
    float b = -mean * a;

    for (int j = 0; j < hidden_size; ++j) {
      ptr[j] = (ptr[j] * a + b) * gamma[j] + beta[j];
    }
  }
}

#if __ARM_NEON
void layernorm_arm(float *input, const float *gamma, const float *beta,
                   int hidden_size, int seq_len) {
  for (int i = 0; i < seq_len; ++i) {
    float *ptr = input + i * hidden_size;

    // mean and var
    float sum = 0.f;
    float sqsum = 0.f;

    for (int j = 0; j < hidden_size; ++j) {
      sum += ptr[j];
    }
    float mean = sum / hidden_size;
    float tmp = 0.f;
    for (int j = 0; j < hidden_size; ++j) {
      tmp = ptr[j] - mean;
      sqsum += tmp * tmp;
    }
    float var = sqsum / hidden_size;

    float a = static_cast<float>(1.f / (sqrt(var + epsilon)));
    float b = -mean * a;

    float32x4_t _a = vdupq_n_f32(a), _b = vdupq_n_f32(b);

    for (int j = 0; j < hidden_size; j += 4) {
      float32x4_t _v = vld1q_f32(ptr + j);
      _v = vmlaq_f32(_b, _v, _a);
      float32x4_t _gamma = vld1q_f32(gamma + j), _beta = vld1q_f32(beta + j);
      _v = vmlaq_f32(_beta, _v, _gamma);
      vst1q_f32(ptr + j, _v);
    }
  }
}
#endif

void residual_bias_ln_i32I(int32_t *input, const float *residual_bias,
                           const float *gamma, const float *beta, float *output,
                           int8_t *output_i8, int feature_dim, int seq_len,
                           float dequant_scale, float quant_scale,
                           bool is_post_ln, const float *colsum) {
  int32_t *pin = input;
  float *pout = output;
  for (int i = 0; i < seq_len; ++i) {
#pragma omp parallel for
    for (int j = 0; j < feature_dim; ++j) {
      if (colsum) {
        pout[j] += (float)pin[j] * dequant_scale + colsum[j];
      } else {
        pout[j] += (float)pin[j] * dequant_scale;
      }
    }
    pin += feature_dim;
    pout += feature_dim;
  }

  if (output_i8) {
    layernorm_residual_i8O(output, output_i8, gamma, beta, residual_bias,
                           feature_dim, seq_len, quant_scale, is_post_ln);
    return;
  }

  layernorm(output, gamma, beta, feature_dim, seq_len);
}

void layernorm_residual_i8O(float *input, int8_t *output, const float *gamma,
                            const float *beta, const float *residual_bias,
                            int hidden_size, int seq_len, float scale,
                            bool is_post_ln) {

#pragma omp parallel for
  for (int i = 0; i < seq_len; ++i) {
    float *ptr = input + i * hidden_size;
    int8_t *out_ptr = output + i * hidden_size;

    // mean and var
    float sum = 0.f;
    float sqsum = 0.f;

    for (int j = 0; j < hidden_size; ++j) {
      sum += ptr[j];
    }
    float mean = sum / hidden_size;
    float tmp = 0.f;
    for (int j = 0; j < hidden_size; ++j) {
      tmp = ptr[j] - mean;
      sqsum += tmp * tmp;
    }
    float var = sqsum / hidden_size;

    float a = static_cast<float>(1.f / (sqrt(var + epsilon)));
    float b = -mean * a;

    float out_f, residual;
    for (int j = 0; j < hidden_size; ++j) {
      out_f = (ptr[j] * a + b) * gamma[j] + beta[j];
      out_ptr[j] = float2int8(out_f, scale);
      residual = residual_bias ? residual_bias[j] : 0.f;
      if (is_post_ln) {
        ptr[j] = out_f + residual;
      } else {
        ptr[j] += residual;
      }
    }
  }
}

} // namespace lightseq
