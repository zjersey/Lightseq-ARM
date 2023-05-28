#include "ops.h"

namespace lightseq {

void quantize(const float *input, int8_t *output, float scale, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    output[i] = float2int8(input[i], scale);
  }
}

void quantize(const int32_t *input, int8_t *output, float scale, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    output[i] = float2int8(input[i], scale);
  }
}
} // namespace lightseq
