// Row-wise in-place L2 normalisation of an f32 row-major matrix on the GPU.
// Rows with zero (or non-finite) norm are left unchanged.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void row_normalize_f32(float* data, int64_t rows, int64_t dim) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  float* p = data + row * dim;
  float acc = 0.f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) acc += p[i] * p[i];
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  __shared__ float inv;
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < (blockDim.x >> 5); ++w) s += red[w];
    inv = (s > 0.f && isfinite(s)) ? rsqrtf(s) : 1.f;
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) p[i] *= inv;
}

extern "C" int lance_cuvs_row_normalize_f32(float* data, int64_t rows, int64_t dim, void* stream) {
  if (rows <= 0) return 0;
  row_normalize_f32<<<(unsigned)rows, 256, 0, (cudaStream_t)stream>>>(data, rows, dim);
  return (int)cudaGetLastError();
}
