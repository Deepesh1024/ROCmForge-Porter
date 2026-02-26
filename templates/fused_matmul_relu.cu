/*
 * TEMPLATE_METADATA:
 *   primitive: fused_matmul
 *   pattern: fused_relu
 *   wave64_safe: true
 *   mfma_enabled: true
 *   attribution: ROCm 7.2 sample + manual adaptation
 */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fused_matmul_relu_kernel(float *A, float *B, float *C, int M,
                                         int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      sum += A[row * K + i] * B[i * N + col];
    }
    // Fused ReLU
    C[row * N + col] = sum > 0.0f ? sum : 0.0f;
  }
}
