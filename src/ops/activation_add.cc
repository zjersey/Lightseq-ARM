#include "ops.h"

namespace lightseq {

void bias_gelu_i8IO(int8_t *input, const float *bias, int seq_len,
                    int feature_dim, float dequant_scale, float quant_scale) {

  for (int i = 0; i < seq_len; ++i) {
#pragma omp parallel for
    for (int j = 0; j < feature_dim; ++j) {
      float vfp = (float)input[j] * dequant_scale + bias[j];
      vfp = gelu(vfp);
      input[j] = float2int8(vfp, quant_scale);
    }
    input += feature_dim;
  }
}

void bias_relu_i8IO(int8_t *input, const float *bias, int seq_len,
                    int feature_dim, float dequant_scale, float quant_scale,
                    float clip_max) {

  for (int i = 0; i < seq_len; ++i) {
#pragma omp parallel for
    for (int j = 0; j < feature_dim; ++j) {
      float vfp = (float)input[j] * dequant_scale + bias[j];
      vfp = fmaxf(vfp, 0.f);
      input[j] = float2int8(vfp, quant_scale);
    }
    input += feature_dim;
  }
}

void scaled_colsum(const int8_t *inp, float *out, int hidden_size,
                   int inner_size, float dequant_scale, float scale) {
  float val;
  for (int i = 0; i < hidden_size; ++i) {
    val = 0.f;
    for (int j = 0; j < inner_size; ++j) {
      val += (float)inp[j * hidden_size + i] * dequant_scale;
    }
    out[i] = val * scale;
  }
}

} // namespace lightseq
