/*
 * TEMPLATE_METADATA:
 *   primitive: dropout
 *   pattern: fused_dropout
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: Basic Dropout Implementation
 */
#include <cuda_runtime.h>

__global__ void dropout_kernel(float *in, float *out, float p_drop, int N,
                               unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    // High quality RNG placeholder
    float rand_val = (float)(idx * seed % 1000) / 1000.0f;
    if (rand_val < p_drop) {
      out[idx] = 0.0f;
    } else {
      out[idx] = in[idx] / (1.0f - p_drop);
    }
  }
}
