#include "ops.h"

namespace lightseq {

void embedding(const int *tokens, const float *token_emb, const float *pos_emb,
               const float *type_emb, float *output, int seq_len,
               int hidden_dim) {
  for (int q = 0; q < seq_len; q++) {
    float *outptr = output + q * hidden_dim;

    int word_index = tokens[q];

    const float *em = token_emb + hidden_dim * word_index;

    memcpy(outptr, em, hidden_dim * sizeof(float));

    for (int p = 0; p < hidden_dim; p++) {
      outptr[p] += pos_emb[q * hidden_dim + p];
      outptr[p] += type_emb[p]; // only support type_id = 0 for now
    }
  }
}

void embedding_int8(const int *tokens, const int8_t *token_emb,
                    const float *pos_emb, float *output, int seq_len,
                    int hidden_dim, float dequant_scale, bool scaled) {
  if (scaled)
    dequant_scale *= sqrtf(hidden_dim);
#pragma omp parallel for
  for (int q = 0; q < seq_len; ++q) {
    float *outptr = output + q * hidden_dim;

    int word_index = tokens[q];

    const int8_t *em = token_emb + hidden_dim * word_index;

    for (int p = 0; p < hidden_dim; ++p) {
      outptr[p] = (float)em[p] * dequant_scale + pos_emb[q * hidden_dim + p];
    }
  }
}

void decoder_embedding_int8(const int *tokens, const int8_t *token_emb,
                            const float *pos_emb, float *output, int seq_len,
                            int vocab_size, int hidden_dim, float dequant_scale,
                            bool scaled) {
  if (scaled)
    dequant_scale *= sqrtf(hidden_dim);

  for (int q = 0; q < seq_len; ++q) {
    int word_index = tokens[q];

#pragma omp parallel for
    for (int p = 0; p < hidden_dim; p++) {
      output[p] =
          (float)token_emb[p * vocab_size + word_index] * dequant_scale +
          pos_emb[p];
    }
    pos_emb += hidden_dim;
    output += hidden_dim;
  }
}

#if __ARM_NEON
void embedding_int8_arm(const int *tokens, const int8_t *token_emb,
                        const float *pos_emb, float *output, int seq_len,
                        int hidden_dim, float dequant_scale, bool scaled) {
  if (scaled)
    dequant_scale *= sqrtf(hidden_dim);
  for (int q = 0; q < seq_len; q++) {
    float *outptr = output + q * hidden_dim;

    int word_index = tokens[q];

    const int8_t *em = token_emb + hidden_dim * word_index;

    for (int p = 0; p < hidden_dim; p += 8) {
      int8x8_t embs_int8 = vld1_s8(em + p);
      int16x8_t embs_int16 = vmovl_s8(embs_int8);
      int16x4_t low_int16 = vget_low_s16(embs_int16),
                high_int16 = vget_high_s16(embs_int16);
      int32x4_t low_int32 = vmovl_s16(low_int16),
                high_int32 = vmovl_s16(high_int16);
      float32x4_t low_f32 = vcvtq_f32_s32(low_int32),
                  high_f32 = vcvtq_f32_s32(high_int32);
      low_f32 = vmulq_n_f32(low_f32, dequant_scale);
      high_f32 = vmulq_n_f32(high_f32, dequant_scale);
      low_f32 = vaddq_f32(low_f32, vld1q_f32(pos_emb + q * hidden_dim + p));
      high_f32 =
          vaddq_f32(high_f32, vld1q_f32(pos_emb + q * hidden_dim + p + 4));
      vst1q_f32(outptr + p, low_f32);
      vst1q_f32(outptr + p + 4, high_f32);
    }
  }
}
#endif
} // namespace lightseq
