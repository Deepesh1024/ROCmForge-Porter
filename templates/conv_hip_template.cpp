/*
 * TEMPLATE_METADATA:
 *   primitive: conv
 *   pattern: direct_conv
 *   wave64_safe: true
 *   mfma_enabled: true
 *   attribution: ROCm Tuned Direct Convolution
 */
#include <hip/hip_runtime.h>

__global__ void conv_kernel({{DTYPE}} * in, {{DTYPE}} * filter, {{DTYPE}} * out,
                            int batch, int in_c, int in_h, int in_w, int out_c,
                            int k_size) {
  int c_out = hipBlockIdx_z;
  int h_out = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  int w_out = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (h_out < in_h && w_out < in_w) {
    float sum = 0.0f;
    for (int c = 0; c < in_c; ++c) {
      for (int kh = 0; kh < k_size; ++kh) {
        for (int kw = 0; kw < k_size; ++kw) {
          int h_in = h_out + kh - k_size / 2;
          int w_in = w_out + kw - k_size / 2;
          if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
            sum +=
                (float)in[((c * in_h) + h_in) * in_w + w_in] *
                (float)
                    filter[(((c_out * in_c) + c) * k_size + kh) * k_size + kw];
          }
        }
      }
    }
    out[((c_out * in_h) + h_out) * in_w + w_out] = ({{DTYPE}})sum;
  }
}
