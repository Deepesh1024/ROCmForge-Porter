"""
ROCmForge Template v1.0
Primitive: GEMM
Source: AMD ROCm 7.2 examples + Triton wave64 adaptation
Safety notes: Wave64 compatible, auto-tuned block sizes

Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
"""

import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = {{ TILE_SIZE }},
    BLOCK_N: tl.constexpr = {{ TILE_SIZE }},
    BLOCK_K: tl.constexpr = {{ TILE_SIZE }},
):
    """Tiled GEMM: C[M×N] = A[M×K] @ B[K×N]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block-level offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers into A and B
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype={{ DTYPE }})

    for k_start in range(0, K, BLOCK_K):
        # Load tiles with masking
        a_mask = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
        b_mask = ((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def launch_gemm(A, B, C, M: int, N: int, K: int):
    """Launch the Triton GEMM kernel."""
    grid = (
        (M + {{ TILE_SIZE }} - 1) // {{ TILE_SIZE }},
        (N + {{ TILE_SIZE }} - 1) // {{ TILE_SIZE }},
    )
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
