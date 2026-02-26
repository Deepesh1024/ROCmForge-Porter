/*
 * TEMPLATE_METADATA:
 *   primitive: layernorm
 *   pattern: fused_layernorm
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: ROCm Optimized LayerNorm
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void layernorm_kernel({{DTYPE}} * in, {{DTYPE}} * out,
                                 {{DTYPE}} * gamma, {{DTYPE}} * beta, int N,
                                 float eps) {
  int row = hipBlockIdx_x;
  int tid = hipThreadIdx_x;

  float sum = 0.0f;
  for (int i = tid; i < N; i += hipBlockDim_x) {
    sum += (float)in[row * N + i];
  }

  // wave64 target
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down(sum, offset, warpSize);
  }
  float mean = sum / N;

  float sq_sum = 0.0f;
  for (int i = tid; i < N; i += hipBlockDim_x) {
    float val = (float)in[row * N + i] - mean;
    sq_sum += val * val;
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sq_sum += __shfl_down(sq_sum, offset, warpSize);
  }
  float var = sq_sum / N;

  for (int i = tid; i < N; i += hipBlockDim_x) {
    float normalized = ((float)in[row * N + i] - mean) * rsqrtf(var + eps);
    out[row * N + i] =
        ({{DTYPE}})(normalized * (float)gamma[i] + (float)beta[i]);
  }
}
