"""
ROCmForge Template v1.0
Primitive: Reduction
Source: AMD ROCm 7.2 examples + Triton wave64 adaptation
Safety notes: Wave64 compatible, uses tl.sum for wavefront-native reduction

Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
"""

import triton
import triton.language as tl


@triton.jit
def reduction_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """
    Parallel sum reduction.
    Each program reduces BLOCK_SIZE elements and atomically adds to out_ptr.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input values (masked)
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})

    # Block-level reduction via Triton built-in (wave64 native)
    block_sum = tl.sum(x, axis=0)

    # Atomic accumulate to output
    tl.atomic_add(out_ptr, block_sum)


def launch_reduction(x_ptr, out_ptr, N: int):
    """Launch the Triton reduction kernel."""
    grid = ((N + {{ BLOCK_SIZE }} - 1) // {{ BLOCK_SIZE }},)
    reduction_kernel[grid](x_ptr, out_ptr, N)
