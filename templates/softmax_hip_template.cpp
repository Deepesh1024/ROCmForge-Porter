/*
 * TEMPLATE_METADATA:
 *   primitive: softmax
 *   pattern: fused_softmax_reduce
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: ROCm Optimized Softmax
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void softmax_kernel({{DTYPE}} * in, {{DTYPE}} * out, int N) {
  int row = hipBlockIdx_x;
  int tid = hipThreadIdx_x;

  {
    {
      DTYPE
    }
  }
  max_val = -1e20f;
  for (int i = tid; i < N; i += hipBlockDim_x) {
    if (in[row * N + i] > max_val)
      max_val = in[row * N + i];
  }

  // wave64 aware reduce max
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    max_val = max(max_val, __shfl_down(max_val, offset, warpSize));
  }

  {
    {
      DTYPE
    }
  }
  sum = 0.0f;
  for (int i = tid; i < N; i += hipBlockDim_x) {
    sum += expf(in[row * N + i] - max_val);
  }

  // wave64 aware reduce sum
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down(sum, offset, warpSize);
  }

  for (int i = tid; i < N; i += hipBlockDim_x) {
    out[row * N + i] = expf(in[row * N + i] - max_val) / sum;
  }
}
