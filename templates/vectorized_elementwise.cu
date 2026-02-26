/*
 * TEMPLATE_METADATA:
 *   primitive: elementwise
 *   pattern: vectorized
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: ROCm 7.2 sample + manual adaptation
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorized_add_kernel(float4 *A, float4 *B, float4 *C,
                                      int N_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N_vec) {
    float4 a = A[idx];
    float4 b = B[idx];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    C[idx] = c;
  }
}
