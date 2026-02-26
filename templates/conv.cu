/*
 * TEMPLATE_METADATA:
 *   primitive: conv
 *   pattern: direct_conv
 *   wave64_safe: true
 *   mfma_enabled: true
 *   attribution: Basic Direct Convolution
 */
#include <cuda_runtime.h>

__global__ void conv_kernel(float *in, float *filter, float *out, int batch,
                            int in_c, int in_h, int in_w, int out_c,
                            int k_size) {
  int c_out = blockIdx.z;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  int w_out = blockIdx.x * blockDim.x + threadIdx.x;

  if (h_out < in_h && w_out < in_w) {
    float sum = 0.0f;
    for (int c = 0; c < in_c; ++c) {
      for (int kh = 0; kh < k_size; ++kh) {
        for (int kw = 0; kw < k_size; ++kw) {
          int h_in = h_out + kh - k_size / 2;
          int w_in = w_out + kw - k_size / 2;
          if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            sum += in[((c * in_h) + h_in) * in_w + w_in] *
                   filter[(((c_out * in_c) + c) * k_size + kh) * k_size + kw];
          }
        }
      }
    }
    out[((c_out * in_h) + h_out) * in_w + w_out] = sum;
  }
}
