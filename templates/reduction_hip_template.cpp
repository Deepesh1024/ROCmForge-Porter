/*
 * ROCmForge Template v1.0
 * Primitive: Reduction
 * Source: AMD ROCm 7.2 examples + manual wave64 adaptation
 * Safety notes: Wave64 wavefront reduction, no warp32 assumptions
 *
 * Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
 */

#include <hip/hip_runtime.h>

#define BLOCK_SIZE                                                             \
  {                                                                            \
    {                                                                          \
      BLOCK_SIZE                                                               \
    }                                                                          \
  }
#define DIM                                                                    \
  {                                                                            \
    {                                                                          \
      DIMS                                                                     \
    }                                                                          \
  }
#define WAVEFRONT_SIZE 64

typedef {
  {
    DTYPE
  }
}
dtype_t;

/*
 * Wavefront-level reduction using DPP (Data-Parallel Primitives).
 * Compatible with wave64 — no warpSize==32 assumptions.
 */
__device__ dtype_t wavefront_reduce_sum(dtype_t val) {
  /* Wave64 reduction via butterfly shuffle */
  for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

/*
 * Block-level reduction kernel.
 * Input:  in[N]
 * Output: out[1] (scalar sum)
 */
__global__ void reduction_kernel(const dtype_t *__restrict__ in,
                                 dtype_t *__restrict__ out, int N) {
  __shared__ dtype_t sdata[BLOCK_SIZE / WAVEFRONT_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * BLOCK_SIZE + tid;

  /* Load element (or zero if out of bounds) */
  dtype_t val = (idx < N) ? in[idx] : static_cast<dtype_t>(0);

  /* Wavefront-level reduction (wave64) */
  val = wavefront_reduce_sum(val);

  /* First lane of each wavefront writes to shared memory */
  int wave_id = tid / WAVEFRONT_SIZE;
  int lane_id = tid % WAVEFRONT_SIZE;

  if (lane_id == 0) {
    sdata[wave_id] = val;
  }
  __syncthreads();

  /* Final reduction across wavefronts (single wavefront does it) */
  int num_waves = BLOCK_SIZE / WAVEFRONT_SIZE;
  if (tid < num_waves) {
    val = sdata[tid];
    /* Reduce across the wavefront leaders */
    for (int offset = num_waves / 2; offset > 0; offset >>= 1) {
      val += __shfl_xor(val, offset);
    }
    if (tid == 0) {
      atomicAdd(out, val);
    }
  }
}

/*
 * Host wrapper: launches the reduction kernel.
 */
void launch_reduction(const dtype_t *d_in, dtype_t *d_out, int N,
                      hipStream_t stream) {
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  hipLaunchKernelGGL(reduction_kernel, dim3(grid), dim3(BLOCK_SIZE), 0, stream,
                     d_in, d_out, N);
}
