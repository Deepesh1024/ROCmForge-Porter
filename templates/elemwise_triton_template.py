"""
ROCmForge Template v1.0
Primitive: Elementwise
Source: AMD ROCm 7.2 examples + Triton wave64 adaptation
Safety notes: Wave64 compatible, vectorised access via BLOCK_SIZE

Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
"""

import triton
import triton.language as tl


@triton.jit
def elemwise_add_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """
    Pointwise addition: C[i] = A[i] + B[i]
    Each program processes BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})

    c = a + b

    tl.store(C_ptr + offsets, c, mask=mask)


def launch_elemwise_add(A, B, C, N: int):
    """Launch the Triton elementwise-add kernel."""
    grid = ((N + {{ BLOCK_SIZE }} - 1) // {{ BLOCK_SIZE }},)
    elemwise_add_kernel[grid](A, B, C, N)
