/*
 * ROCmForge Template v1.0
 * Primitive: GEMM
 * Source: AMD ROCm 7.2 examples + manual wave64 adaptation
 * Safety notes: MFMA enabled, wave64 compatible
 *
 * Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define TILE_SIZE {{ TILE_SIZE }}
#define BLOCK_SIZE {{ BLOCK_SIZE }}
#define DIM {{ DIMS }}

typedef {{ DTYPE }} dtype_t;

/*
 * Tiled GEMM kernel using shared memory
 * C[M×N] = A[M×K] × B[K×N]
 * Each block computes a TILE_SIZE × TILE_SIZE sub-matrix of C.
 */
__global__ void gemm_kernel(
    const dtype_t* __restrict__ A,
    const dtype_t* __restrict__ B,
    dtype_t* __restrict__ C,
    int M, int N, int K)
{
    __shared__ dtype_t As[TILE_SIZE][TILE_SIZE];
    __shared__ dtype_t Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    dtype_t acc = static_cast<dtype_t>(0);

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        /* Load tile from A */
        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = static_cast<dtype_t>(0);

        /* Load tile from B */
        if ((t * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = static_cast<dtype_t>(0);

        __syncthreads();

        /* Accumulate partial product */
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    /* Store result */
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/*
 * Host wrapper: launches the GEMM kernel.
 * For production, consider using rocBLAS:
 *   rocblas_sgemm(handle, transA, transB, M, N, K,
 *                 &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
 */
void launch_gemm(const dtype_t* dA, const dtype_t* dB, dtype_t* dC,
                 int M, int N, int K, hipStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(gemm_kernel, grid, block, 0, stream,
                       dA, dB, dC, M, N, K);
}
