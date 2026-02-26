/*
 * TEMPLATE_METADATA:
 *   primitive: reduction
 *   pattern: wavefront_reduce
 *   wave64_safe: true
 *   mfma_enabled: false
 *   attribution: ROCm 7.2 sample + manual adaptation
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

__inline__ __device__ {
  {
    DTYPE
  }
}
waveReduceSum({
  {
    DTYPE
  }
} val) {
  // Wave64 aware hardware intrinsic reduce
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset, warpSize);
  return val;
}

__global__ void wavefront_reduce_kernel({{DTYPE}} * in, {{DTYPE}} * out,
                                        int size) {
  {
    {
      DTYPE
    }
  }
  sum = 0;
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < size;
       i += hipBlockDim_x * hipGridDim_x) {
    sum += in[i];
  }
  sum = waveReduceSum(sum);
  // Explicit wave64 check for the first thread of each wavefront
  if ((hipThreadIdx_x & (warpSize - 1)) == 0) {
    atomicAdd(out, sum);
  }
}
