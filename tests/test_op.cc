
#include "../src/ops/ops.h"
#include "omp.h"
#include "util.h"

using namespace lightseq;

extern "C" void MY_MMult(int, int, int, int8_t *, int, int8_t *, int, int32_t *,
                         int);

void test_embedding_int8(int times = 10, int seq_len = 8,
                         int hidden_dim = 768) {
  int *tokens = RandomIntTensor(seq_len, 0, 10000);
  int8_t *token_emb = RandomInt8Tensor(30522 * hidden_dim);
  float *pos_emb = RandomFloatTensor(512 * hidden_dim);
  float *output = (float *)malloc(seq_len * hidden_dim * sizeof(float));
  float dequant_scale = RandomFloat(0., 2.);

  double omp_start, omp_end;
  double totaltime;

  embedding_int8(tokens, token_emb, pos_emb, output, seq_len, hidden_dim,
                 dequant_scale);

#if __ARM_NEON
  float *output2 = (float *)malloc(seq_len * hidden_dim * sizeof(float));
  embedding_int8_arm(tokens, token_emb, pos_emb, output2, seq_len, hidden_dim,
                     dequant_scale);
  printf("average diff: %f\n", compare(output, output2, seq_len * hidden_dim));

  omp_start = omp_get_wtime();
  for (int i = 0; i < times; ++i) {
    embedding_int8_arm(tokens, token_emb, pos_emb, output2, seq_len, hidden_dim,
                       dequant_scale);
  }
  omp_end = omp_get_wtime();
  totaltime = (omp_end - omp_start) * 1000;
  printf("average time of embedding_int8_arm: %lf\n", totaltime / times);
#endif

  omp_start = omp_get_wtime();
  for (int i = 0; i < times; ++i) {
    embedding_int8(tokens, token_emb, pos_emb, output, seq_len, hidden_dim,
                   dequant_scale);
  }
  omp_end = omp_get_wtime();
  totaltime = (omp_end - omp_start) * 1000;
  printf("average time of embedding_int8: %lf\n", totaltime / times);
}

void test_layernorm(int times = 10, int seq_len = 8, int hidden_dim = 768) {
  float *input = RandomFloatTensor(seq_len * hidden_dim);
  float *gamma = RandomFloatTensor(hidden_dim);
  float *beta = RandomFloatTensor(hidden_dim);
  float *input2 = RandomFloatTensor(seq_len * hidden_dim);
  memcpy(input2, input, seq_len * hidden_dim * sizeof(float));

  double omp_start, omp_end;
  double totaltime;

  layernorm(input, gamma, beta, hidden_dim, seq_len);

#if __ARM_NEON
  layernorm_arm(input2, gamma, beta, hidden_dim, seq_len);
  printf("average diff: %f\n", compare(input, input2, seq_len * hidden_dim));

  omp_start = omp_get_wtime();
  for (int i = 0; i < times; ++i) {
    layernorm_arm(input2, gamma, beta, hidden_dim, seq_len);
  }
  omp_end = omp_get_wtime();
  totaltime = (omp_end - omp_start) * 1000;
  printf("average time of layernorm_arm: %lf\n", totaltime / times);
#endif

  omp_start = omp_get_wtime();
  for (int i = 0; i < times; ++i) {
    layernorm(input, gamma, beta, hidden_dim, seq_len);
  }
  omp_end = omp_get_wtime();
  totaltime = (omp_end - omp_start) * 1000;
  printf("average time of layernorm: %lf\n", totaltime / times);
}

void test_res_ln_bias(int times = 10, int seq_len = 8, int hidden_size = 768) {
  int32_t *input = RandomIntTensor(seq_len * hidden_size);
  float *residual_bias = RandomFloatTensor(hidden_size);
  float *gamma = RandomFloatTensor(hidden_size);
  float *beta = RandomFloatTensor(hidden_size);
  float *output = RandomFloatTensor(seq_len * hidden_size);
  int8_t *output_i8 = RandomInt8Tensor(seq_len * hidden_size);
  float dequant_scale = RandomFloat(0.9, 20.);
  float quant_scale = RandomFloat(0.9, 20.);

  double omp_start, omp_end;
  double totaltime;
  omp_start = omp_get_wtime();
  for (int i = 0; i < times; ++i) {
    residual_bias_ln_i32I(input, residual_bias, gamma, beta, output, output_i8,
                          hidden_size, seq_len, dequant_scale, quant_scale,
                          true);
  }
  omp_end = omp_get_wtime();
  totaltime = (omp_end - omp_start) * 1000;
  printf("average time of residual_bias_ln_i32I: %lf ms.\n", totaltime / times);
}

void test_gemm_int8(int times = 10, int m = 8, int k = 768, int n = 3072) {

  int8_t *input = RandomInt8Tensor(m * k);
  int8_t *weight = RandomInt8Tensor(n * k);
  int8_t *output = RandomInt8Tensor(m * n);

  float quant_scale = RandomFloat(0.9, 20.);

  double omp_start, omp_end;
  double totaltime;

  gemm_int8(input, weight, output, m, k, n, quant_scale);

#if __ARM_NEON
  int32_t *buffer = (int32_t *)malloc(m * n * sizeof(int32_t));
  int8_t *output2 = RandomInt8Tensor(m * n);
  gemm_int8_arm(input, weight, output2, m, k, n, quant_scale, buffer);
  printf("average diff: %f\n", compare(output, output2, m * n));

  omp_start = omp_get_wtime();
  for (int i = 0; i < times; ++i) {
    gemm_int8_arm(input, weight, nullptr, m, k, n, quant_scale, buffer);
  }
  omp_end = omp_get_wtime();
  totaltime = (omp_end - omp_start) * 1000;
  printf("average time of MY_MMult: %lf ms.\n", totaltime / times);
#endif
}

int main(int argc, char **argv) {
#if __aarch64__
  printf("__aarch64__\n");
#endif
#if __ARM_FEATURE_DOTPROD
  printf("__ARM_FEATURE_DOTPROD\n");
#endif
#if __ARM_FEATURE_MATMUL_INT8
  printf("__ARM_FEATURE_MATMUL_INT8\n");
#endif
#if __ARM_NEON
  printf("__ARM_NEON\n");
#endif

  srand(time(NULL));
  int times = strtol(argv[1], nullptr, 10);
  int m = strtol(argv[2], nullptr, 10);
  int k = strtol(argv[3], nullptr, 10);
  int n = strtol(argv[4], nullptr, 10);
  test_gemm_int8(times, m, k, n);
  return 0;
}
