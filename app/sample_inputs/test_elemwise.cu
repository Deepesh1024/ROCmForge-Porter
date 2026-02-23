/*
 * Sample CUDA input — Elementwise (vector add)
 * Used for testing the ROCmForge Studio /parse endpoint.
 */

#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void elementwise_add(const float *A, const float *B, float *C,
                                int n) {
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
  float *dA, *dB, *dC;
  cudaMalloc(&dA, N * sizeof(float));
  cudaMalloc(&dB, N * sizeof(float));
  cudaMalloc(&dC, N * sizeof(float));

  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  elementwise_add<<<grid, BLOCK_SIZE>>>(dA, dB, dC, N);

  cudaDeviceSynchronize();

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
