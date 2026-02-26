/*
 * TEMPLATE_METADATA:
 *   primitive: fused_matmul
 *   pattern: fused_relu
 *   wave64_safe: true
 *   mfma_enabled: true
 *   attribution: ROCm 7.2 sample + manual adaptation
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void fused_matmul_relu_kernel({{DTYPE}} * A, {{DTYPE}} * B,
                                         {{DTYPE}} * C, int M, int N, int K) {
  int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (row < M && col < N) {
    {
      {
        DTYPE
      }
    }
    sum = 0;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    // Fused ReLU
    C[row * N + col] = sum > 0 ? sum : 0;
  }
}
