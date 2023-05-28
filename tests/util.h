#include <stdio.h>
#include <stdlib.h>

static float RandomFloat(float a = -1.2f, float b = 1.2f) {
  float random = ((float)rand()) / (float)RAND_MAX; // RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

static int RandomInt(int a = -10000, int b = 10000) {
  float random = ((float)rand()) / (float)RAND_MAX; // RAND_MAX;
  int diff = b - a;
  float r = random * diff;
  return a + (int)r;
}

static int8_t RandomS8() { return (int8_t)RandomInt(-127, 127); }

static float *RandomFloatTensor(size_t size, float a = -1.2f, float b = 1.2f) {
  float *data = (float *)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; ++i) {
    data[i] = RandomFloat(a, b);
  }
  return data;
}

static int *RandomIntTensor(size_t size, int a = -10000, int b = 10000) {
  int *data = (int *)malloc(size * sizeof(int));
  for (size_t i = 0; i < size; ++i) {
    data[i] = RandomInt(a, b);
  }
  return data;
}

static int8_t *RandomInt8Tensor(size_t size) {
  int8_t *data = (int8_t *)malloc(size * sizeof(int8_t));
  for (size_t i = 0; i < size; ++i) {
    data[i] = RandomS8();
  }
  return data;
}

template <typename T>
static double compare(const T *p1, const T *p2, size_t size) {
  double result = 0.f;
  for (int i = 0; i < size; ++i) {
    result += abs(p1[i] - p2[i]);
  }
  return result / size;
}

template <typename T>
static void print_vec(const T *vec, const char *name = "", size_t size = 0) {
  printf("%s:  ", name);
  for (int i = 0; i < size; ++i) {
    std::cout << vec[i] << ",  ";
  }
  printf("\n");
}

static void print_vec(const int8_t *vec, const char *name = "",
                      size_t size = 0) {
  printf("%s:  ", name);
  for (int i = 0; i < size; ++i) {
    printf("%d,  ", vec[i]);
  }
  printf("\n");
}

template <typename T>
static void print_sum(const T *vec, const char *name = "",
                      size_t size = 0) {
  float sum = 0.f;
  for (int i = 0; i < size; ++i) {
    sum += vec[i];
  }
  printf("sum of %s: %f\n", name, sum);
}