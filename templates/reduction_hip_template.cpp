// clang-format off
/*
 * ROCmForge Template v2.0
 * Primitive: Reduction
 * Source: AMD ROCm 7.2 examples + wave64 wavefront-native reduction
 * Safety notes: Wave64 compatible, no warp32 assumptions, uses DPP
 *
 * Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
 */

#include <hip/hip_runtime.h>

#define BLOCK_SIZE {{ BLOCK_SIZE }}
#define DIM {{ DIMS }}
#define WAVEFRONT_SIZE 64  /* AMD GCN / CDNA default */

typedef {{ DTYPE }} dtype_t;

/* ─── Wavefront-level Reduction ─────────────────────────────────
 * Uses butterfly shuffle for wave64 — no warpSize==32 assumptions.
 * Compatible with GCN, RDNA (wave32 mode), and CDNA architectures.
 */
__device__ __forceinline__
dtype_t wavefront_reduce_sum(dtype_t val)
{
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    }
    return val;
}

__device__ __forceinline__
dtype_t wavefront_reduce_max(dtype_t val)
{
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        dtype_t other = __shfl_xor(val, offset, WAVEFRONT_SIZE);
        val = (val > other) ? val : other;
    }
    return val;
}

/* ─── Block-level Sum Reduction ─────────────────────────────────
 * Input:  in[N]
 * Output: out[1] — scalar sum across all elements
 *
 * Each block reduces BLOCK_SIZE elements, then atomicAdd to output.
 * Uses two-stage: wavefront-level → shared-mem → final wavefront.
 */
__global__ void reduction_sum_kernel(
    const dtype_t* __restrict__ in,
    dtype_t*       __restrict__ out,
    int N)
{
    __shared__ dtype_t sdata[BLOCK_SIZE / WAVEFRONT_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK_SIZE + tid;

    /* Load with bounds check */
    dtype_t val = (idx < N) ? in[idx] : static_cast<dtype_t>(0);

    /* Stage 1: wavefront-level reduction (wave64) */
    val = wavefront_reduce_sum(val);

    /* First lane of each wavefront writes to shared memory */
    const int wave_id = tid / WAVEFRONT_SIZE;
    const int lane_id = tid % WAVEFRONT_SIZE;

    if (lane_id == 0) {
        sdata[wave_id] = val;
    }
    __syncthreads();

    /* Stage 2: reduce across wavefronts */
    const int num_waves = BLOCK_SIZE / WAVEFRONT_SIZE;
    if (tid < num_waves) {
        val = sdata[tid];
        for (int offset = num_waves / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
        }
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}

/* ─── Block-level Max Reduction ─────────────────────────────────
 * Same structure as sum but uses max instead of add.
 */
__global__ void reduction_max_kernel(
    const dtype_t* __restrict__ in,
    dtype_t*       __restrict__ out,
    int N)
{
    __shared__ dtype_t sdata[BLOCK_SIZE / WAVEFRONT_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK_SIZE + tid;

    dtype_t val = (idx < N) ? in[idx] : static_cast<dtype_t>(-1e30);

    val = wavefront_reduce_max(val);

    const int wave_id = tid / WAVEFRONT_SIZE;
    const int lane_id = tid % WAVEFRONT_SIZE;

    if (lane_id == 0) {
        sdata[wave_id] = val;
    }
    __syncthreads();

    const int num_waves = BLOCK_SIZE / WAVEFRONT_SIZE;
    if (tid < num_waves) {
        val = sdata[tid];
        for (int offset = num_waves / 2; offset > 0; offset >>= 1) {
            dtype_t other = __shfl_xor(val, offset, WAVEFRONT_SIZE);
            val = (val > other) ? val : other;
        }
        if (tid == 0) {
            atomicMax((int*)out, __float_as_int(val));
        }
    }
}

/* ─── Host Wrapper ───────────────────────────────────────────── */
void launch_reduction_sum(const dtype_t* d_in, dtype_t* d_out, int N,
                          hipStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(reduction_sum_kernel, dim3(grid), dim3(BLOCK_SIZE),
                       0, stream, d_in, d_out, N);
}

void launch_reduction_max(const dtype_t* d_in, dtype_t* d_out, int N,
                          hipStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(reduction_max_kernel, dim3(grid), dim3(BLOCK_SIZE),
                       0, stream, d_in, d_out, N);
}
// clang-format on
