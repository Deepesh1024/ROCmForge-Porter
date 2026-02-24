// clang-format off
/*
 * ROCmForge Template v2.0
 * Primitive: GEMM
 * Source: AMD ROCm 7.2 examples + manual wave64 adaptation
 * Safety notes: MFMA ready, wave64 compatible, tiled shared-mem
 *
 * Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define TILE_SIZE {{ TILE_SIZE }}
#define BLOCK_SIZE {{ BLOCK_SIZE }}
#define DIM {{ DIMS }}

typedef {{ DTYPE }} dtype_t;

/* ─── Tiled GEMM Kernel ───────────────────────────────────────────
 * C[M×N] = A[M×K] × B[K×N]
 * Each block computes a TILE_SIZE × TILE_SIZE sub-matrix of C.
 * Uses row-major shared-memory layout to avoid LDS bank conflicts.
 * Wave64 compatible — no warpSize==32 assumptions.
 */
__global__ void gemm_kernel(
    const dtype_t* __restrict__ A,
    const dtype_t* __restrict__ B,
    dtype_t*       __restrict__ C,
    int M, int N, int K)
{
    /* Row-major shared tiles — avoids LDS bank conflicts on AMD */
    __shared__ dtype_t As[TILE_SIZE][TILE_SIZE + 1]; // +1 padding
    __shared__ dtype_t Bs[TILE_SIZE][TILE_SIZE + 1]; // +1 padding

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    dtype_t acc = static_cast<dtype_t>(0);

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        /* Load tile from A (bounds-checked) */
        const int a_col = t * TILE_SIZE + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col]
                                             : static_cast<dtype_t>(0);

        /* Load tile from B (bounds-checked) */
        const int b_row = t * TILE_SIZE + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col]
                                             : static_cast<dtype_t>(0);

        __syncthreads();

        /* Accumulate partial product */
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    /* Store result (bounds-checked) */
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/* ─── MFMA-Accelerated GEMM Stub ────────────────────────────────
 * For MI200/MI300 GPUs, use MFMA (Matrix Fused Multiply-Add):
 *
 *   // Example: fp32 4×4×4 MFMA
 *   typedef float  __attribute__((ext_vector_type(4))) float4;
 *   float4 acc = {0};
 *   acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, acc, 0, 0, 0);
 *
 * For production GEMM, use rocBLAS directly:
 *   rocblas_sgemm(handle, transA, transB, M, N, K,
 *                 &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
 */

/* ─── Host Wrapper ───────────────────────────────────────────── */
void launch_gemm(const dtype_t* dA, const dtype_t* dB, dtype_t* dC,
                 int M, int N, int K, hipStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(gemm_kernel, grid, block, 0, stream,
                       dA, dB, dC, M, N, K);
}
// clang-format on
