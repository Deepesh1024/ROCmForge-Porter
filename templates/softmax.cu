/*
 * TEMPLATE_METADATA:
 *   primitive: softmax
 *   pattern: fused_softmax_reduce
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: Standard DL Softmax
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void softmax_kernel(float *in, float *out, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  float max_val = -1e20f;
  for (int i = tid; i < N; i += blockDim.x) {
    if (in[row * N + i] > max_val)
      max_val = in[row * N + i];
  }

  // warp reduce max
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    max_val = max(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
  }

  float sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    sum += expf(in[row * N + i] - max_val);
  }

  // warp reduce sum
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  for (int i = tid; i < N; i += blockDim.x) {
    out[row * N + i] = expf(in[row * N + i] - max_val) / sum;
  }
}
