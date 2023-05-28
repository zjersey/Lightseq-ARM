#include "math.h"

namespace lightseq {

static const float epsilon = 0.000000000001;

static inline int8_t float2int8(float x, float quant_scale) {
  float i8_f = x * quant_scale;
  int32_t i8 = floorf(i8_f + 0.5);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

static inline int8_t posfloat2int8(float x, float quant_scale, float clip_max) {
  float i8_f = x * 2 * quant_scale - quant_scale * clip_max;
  int32_t i8 = floorf(i8_f + 0.5);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

static inline float gelu(float x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

static inline int flat_2dim(int id1, int id2, int dim2) {
  return id1 * dim2 + id2;
}

static inline int flat_3dim(int id1, int id2, int id3, int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

} // namespace lightseq
