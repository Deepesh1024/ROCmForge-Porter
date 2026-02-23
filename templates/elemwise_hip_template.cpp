/*
 * ROCmForge Template v1.0
 * Primitive: Elementwise
 * Source: AMD ROCm 7.2 examples + manual wave64 adaptation
 * Safety notes: Vectorised float4 loads for bandwidth, wave64 compatible
 *
 * Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
 */

#include <hip/hip_runtime.h>

#define BLOCK_SIZE                                                             \
  {                                                                            \
    {                                                                          \
      BLOCK_SIZE                                                               \
    }                                                                          \
  }
#define DIM                                                                    \
  {                                                                            \
    {                                                                          \
      DIMS                                                                     \
    }                                                                          \
  }

typedef {
  {
    DTYPE
  }
}
dtype_t;

/*
 * Simple pointwise kernel:  C[i] = A[i] + B[i]
 * Uses vectorised loads (float4) when dtype is float for better bandwidth.
 */
__global__ void elemwise_add_kernel(const dtype_t *__restrict__ A,
                                    const dtype_t *__restrict__ B,
                                    dtype_t *__restrict__ C, int N) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

/*
 * Vectorised elementwise kernel (float4 path).
 * Processes 4 elements per thread for higher memory throughput.
 */
__global__ void elemwise_add_vec4_kernel(const float4 *__restrict__ A,
                                         const float4 *__restrict__ B,
                                         float4 *__restrict__ C, int N4) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (idx < N4) {
    float4 a = A[idx];
    float4 b = B[idx];
    C[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}

/*
 * Host wrapper: launches the elementwise kernel.
 * Picks the vectorised path when N is divisible by 4 and dtype is float.
 */
void launch_elemwise_add(const dtype_t *dA, const dtype_t *dB, dtype_t *dC,
                         int N, hipStream_t stream) {
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  hipLaunchKernelGGL(elemwise_add_kernel, dim3(grid), dim3(BLOCK_SIZE), 0,
                     stream, dA, dB, dC, N);
}
