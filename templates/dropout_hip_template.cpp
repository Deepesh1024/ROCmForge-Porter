/*
 * TEMPLATE_METADATA:
 *   primitive: dropout
 *   pattern: fused_dropout
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: Basic Dropout Implementation
 */
#include <hip/hip_runtime.h>

__global__ void dropout_kernel({{DTYPE}} * in, {{DTYPE}} * out, float p_drop,
                               int N, unsigned long long seed) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (idx < N) {
    // High quality RNG placeholder
    float rand_val = (float)(idx * seed % 1000) / 1000.0f;
    if (rand_val < p_drop) {
      out[idx] = ({{DTYPE}})0.0f;
    } else {
      out[idx] = ({{DTYPE}})((float)in[idx] / (1.0f - p_drop));
    }
  }
}
