/*
 * TEMPLATE_METADATA:
 *   primitive: layernorm
 *   pattern: fused_layernorm
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: Standard DL LayerNorm
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void layernorm_kernel(float *in, float *out, float *gamma,
                                 float *beta, int N, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    sum += in[row * N + i];
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }
  float mean = sum / N;

  float sq_sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    float val = in[row * N + i] - mean;
    sq_sum += val * val;
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sq_sum += __shfl_down_sync(0xFFFFFFFF, sq_sum, offset);
  }
  float var = sq_sum / N;

  for (int i = tid; i < N; i += blockDim.x) {
    float normalized = (in[row * N + i] - mean) * rsqrtf(var + eps);
    out[row * N + i] = normalized * gamma[i] + beta[i];
  }
}
