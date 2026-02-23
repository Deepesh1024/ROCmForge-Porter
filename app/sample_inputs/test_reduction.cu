/*
 * Sample CUDA input — Reduction (parallel sum)
 * Used for testing the ROCmForge Studio /parse endpoint.
 */

#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void reduction(const float *in, float *out, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * BLOCK_SIZE + tid;

  sdata[tid] = (idx < n) ? in[idx] : 0.0f;
  __syncthreads();

  /* Tree reduction in shared memory */
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out, sdata[0]);
  }
}

int main() {
  float *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, sizeof(float));
  cudaMemset(d_out, 0, sizeof(float));

  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  reduction<<<grid, BLOCK_SIZE>>>(d_in, d_out, N);

  cudaDeviceSynchronize();

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
