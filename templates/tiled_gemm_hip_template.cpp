/*
 * TEMPLATE_METADATA:
 *   primitive: gemm
 *   pattern: tiled_shared
 *   wave64_safe: true
 *   mfma_enabled: true
 *   attribution: ROCm 7.2 sample + manual adaptation
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void tiled_gemm_kernel({{DTYPE}} * A, {{DTYPE}} * B, {{DTYPE}} * C,
                                  int M, int N, int K) {
  __shared__ {
    {
      DTYPE
    }
  }
  As[{{TILE_SIZE}}][{{TILE_SIZE}}];
  __shared__ {
    {
      DTYPE
    }
  }
  Bs[{{TILE_SIZE}}][{{TILE_SIZE}}];

  int bx = hipBlockIdx_x;
  int by = hipBlockIdx_y;
  int tx = hipThreadIdx_x;
  int ty = hipThreadIdx_y;

  int row = by * {{TILE_SIZE}} + ty;
  int col = bx * {{TILE_SIZE}} + tx;

  {
    {
      DTYPE
    }
  }
  pvalue = 0;

  for (int p = 0; p < (K + {{TILE_SIZE}} - 1) / {{TILE_SIZE}}; ++p) {
    if (row < M && p * {{TILE_SIZE}} + tx < K)
      As[ty][tx] = A[row * K + p * {{TILE_SIZE}} + tx];
    else
      As[ty][tx] = 0;

    if (p * {{TILE_SIZE}} + ty < K && col < N)
      Bs[ty][tx] = B[(p * {{TILE_SIZE}} + ty) * N + col];
    else
      Bs[ty][tx] = 0;

    __syncthreads();

    for (int k = 0; k < {{TILE_SIZE}}; ++k) {
      pvalue += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = pvalue;
  }
}
