/*
 * Sample CUDA input — GEMM (matrix multiplication)
 * Used for testing the ROCmForge Studio /parse endpoint.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define K 1024

__global__ void matmul(const float *A, const float *B, float *C, int M, int N,
                       int K) {
  __shared__ float As[16][16];
  __shared__ float Bs[16][16];

  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + 15) / 16; ++t) {
    if (row < M && t * 16 + threadIdx.x < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (t * 16 + threadIdx.y < K && col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < 16; ++k)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
}

int main() {
  float *dA, *dB, *dC;
  cudaMalloc(&dA, M * K * sizeof(float));
  cudaMalloc(&dB, K * N * sizeof(float));
  cudaMalloc(&dC, M * N * sizeof(float));

  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  matmul<<<grid, block>>>(dA, dB, dC, M, N, K);

  cudaDeviceSynchronize();

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
