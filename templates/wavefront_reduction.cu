/*
 * TEMPLATE_METADATA:
 *   primitive: reduction
 *   pattern: wavefront_reduce
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: ROCm 7.2 sample + manual adaptation
 */
#include <cuda_runtime.h>
#include <stdio.h>

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__global__ void wavefront_reduce_kernel(float *in, float *out, int size) {
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = warpReduceSum(sum);
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(out, sum);
  }
}
