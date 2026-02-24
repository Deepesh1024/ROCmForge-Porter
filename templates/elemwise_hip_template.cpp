// clang-format off
/*
 * ROCmForge Template v2.0
 * Primitive: Elementwise
 * Source: AMD ROCm 7.2 examples + vectorised memory access patterns
 * Safety notes: Vectorised float4 loads for bandwidth, wave64 compatible
 *
 * Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
 */

#include <hip/hip_runtime.h>

#define BLOCK_SIZE {{ BLOCK_SIZE }}
#define DIM {{ DIMS }}

typedef {{ DTYPE }} dtype_t;

/* ─── Scalar Elementwise Add ────────────────────────────────────
 * C[i] = A[i] + B[i]
 * Bounds-checked. Works for any dtype.
 */
__global__ void elemwise_add_kernel(
    const dtype_t* __restrict__ A,
    const dtype_t* __restrict__ B,
    dtype_t*       __restrict__ C,
    int N)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

/* ─── Scalar Elementwise Mul ────────────────────────────────────
 * C[i] = A[i] * B[i]
 */
__global__ void elemwise_mul_kernel(
    const dtype_t* __restrict__ A,
    const dtype_t* __restrict__ B,
    dtype_t*       __restrict__ C,
    int N)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

/* ─── Fused ReLU: C[i] = max(0, A[i]) ──────────────────────────
 * Common activation function — demonstrates fused elementwise op.
 */
__global__ void elemwise_relu_kernel(
    const dtype_t* __restrict__ A,
    dtype_t*       __restrict__ C,
    int N)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        C[idx] = (A[idx] > static_cast<dtype_t>(0)) ? A[idx]
                                                     : static_cast<dtype_t>(0);
    }
}

/* ─── Fused Scale + Bias: C[i] = A[i] * scale + bias ───────────
 * Common in normalization layers.
 */
__global__ void elemwise_scale_bias_kernel(
    const dtype_t* __restrict__ A,
    dtype_t*       __restrict__ C,
    dtype_t scale, dtype_t bias,
    int N)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * scale + bias;
    }
}

/* ─── Vectorised Add (float4) ───────────────────────────────────
 * Processes 4 elements per thread for higher memory throughput.
 * Requires N to be divisible by 4 and dtype to be float.
 */
__global__ void elemwise_add_vec4_kernel(
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    float4*       __restrict__ C,
    int N4)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N4) {
        float4 a = A[idx];
        float4 b = B[idx];
        C[idx] = make_float4(a.x + b.x, a.y + b.y,
                             a.z + b.z, a.w + b.w);
    }
}

/* ─── Host Wrappers ──────────────────────────────────────────── */
void launch_elemwise_add(const dtype_t* dA, const dtype_t* dB, dtype_t* dC,
                         int N, hipStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(elemwise_add_kernel, dim3(grid), dim3(BLOCK_SIZE),
                       0, stream, dA, dB, dC, N);
}

void launch_elemwise_mul(const dtype_t* dA, const dtype_t* dB, dtype_t* dC,
                         int N, hipStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(elemwise_mul_kernel, dim3(grid), dim3(BLOCK_SIZE),
                       0, stream, dA, dB, dC, N);
}

void launch_elemwise_relu(const dtype_t* dA, dtype_t* dC,
                          int N, hipStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(elemwise_relu_kernel, dim3(grid), dim3(BLOCK_SIZE),
                       0, stream, dA, dC, N);
}
// clang-format on
