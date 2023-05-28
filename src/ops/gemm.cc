#include "ops.h"
extern "C" void MY_MMult(int, int, int, const int8_t *, int, const int8_t *,
                         int, int32_t *, int);

namespace lightseq {

void gemm_fp32(const float *A, const float *B, float *C, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    const float *pA = A + i * k;
    for (int j = 0; j < n; ++j) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p) {
        sum += pA[p] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

// [m, k]*[k, n]=[m, n]
void gemm_int8(const int8_t *A, const int8_t *B, int8_t *C, int m, int k, int n,
               float scale, int32_t *C_i32) {
  for (int i = 0; i < m; ++i) {
    const int8_t *pA = A + i * k;

    for (int j = 0; j < n; ++j) {
      int sum = 0;
      for (int p = 0; p < k; ++p) {
        sum += pA[p] * B[p * n + j];
      }

      if (C) {
        C[i * n + j] = float2int8(sum, scale);
      } else {
        C_i32[i * n + j] = sum;
      }
    }
  }
}

#if __ARM_NEON
void gemm_int8_arm_v0(const int8_t *A, const int8_t *B, int8_t *C, int m, int k,
                      int n, float scale, int32_t *C_i32) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const int8_t *pA = A + i * k;
      const int8_t *pB = B + j * k;
      int sum = 0;

      int p = 0;
      int32x4_t _sum0 = vdupq_n_s32(0);
      int32x4_t _sum1 = vdupq_n_s32(0);

      for (; p + 7 < k; p += 8) {
        int8x8_t _vA = vld1_s8(pA);
        int8x8_t _vB = vld1_s8(pB);

        int16x8_t _s0 = vmull_s8(_vA, _vB);
        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

        pA += 8;
        pB += 8;
      }

      _sum0 = vaddq_s32(_sum0, _sum1);
      sum = vaddvq_s32(_sum0);

      for (; p < k; ++p) {
        sum += *pA++ * *pB++;
      }

      if (C) {
        C[i * n + j] = float2int8(sum, scale);
      } else {
        C_i32[i * n + j] = sum;
      }
    }
  }
}

void gemm_int8_arm(const int8_t *A, const int8_t *B, int8_t *C, int m, int k,
                   int n, float scale, int32_t *C_i32) {
  memset(C_i32, 0, m * n * sizeof(int32_t));
  MY_MMult(m, n, k, A, k, B, n, C_i32, n);
  if (C) {
    quantize(C_i32, C, scale, m * n);
  }
}

void batch_gemm_int8_arm(const int8_t *A, const int8_t *B, int8_t *C, int m,
                         int k, int n, int sA, int sB, int sC, int num,
                         float scale, int32_t *C_i32) {
  for (int i = 0; i < num; ++i) {
    memset(C_i32 + i * sC, 0, m * n * sizeof(int32_t));
    MY_MMult(m, n, k, A + i * sA, k, B + i * sB, n, C_i32 + i * sC, n);
    if (C + i * sC) {
      quantize(C_i32 + i * sC, C + i * sC, scale, m * n);
    }
  }
}

#endif
} // namespace lightseq
