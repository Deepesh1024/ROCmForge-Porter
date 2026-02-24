# fmt: off
"""
ROCmForge Template v2.0
Primitive: Elementwise
Source: AMD ROCm 7.2 examples + Triton wave64 adaptation
Safety notes: Wave64 compatible, vectorised via BLOCK_SIZE, fused ops

Placeholders: {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
"""

import triton
import triton.language as tl


@triton.jit
def elemwise_add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """C[i] = A[i] + B[i]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})

    tl.store(C_ptr + offsets, a + b, mask=mask)


@triton.jit
def elemwise_mul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """C[i] = A[i] * B[i]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})

    tl.store(C_ptr + offsets, a * b, mask=mask)


@triton.jit
def elemwise_relu_kernel(
    A_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """C[i] = max(0, A[i])"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})
    c = tl.maximum(a, 0.0)

    tl.store(C_ptr + offsets, c, mask=mask)


@triton.jit
def elemwise_sigmoid_kernel(
    A_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """C[i] = 1 / (1 + exp(-A[i]))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})
    c = 1.0 / (1.0 + tl.exp(-a))

    tl.store(C_ptr + offsets, c, mask=mask)


@triton.jit
def elemwise_fused_add_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = {{ BLOCK_SIZE }},
):
    """Fused: C[i] = max(0, A[i] + B[i])"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to({{ DTYPE }})

    c = tl.maximum(a + b, 0.0)
    tl.store(C_ptr + offsets, c, mask=mask)


def launch_elemwise_add(A, B, C, N: int):
    """Launch add kernel."""
    grid = ((N + {{ BLOCK_SIZE }} - 1) // {{ BLOCK_SIZE }},)
    elemwise_add_kernel[grid](A, B, C, N)


def launch_elemwise_relu(A, C, N: int):
    """Launch ReLU kernel."""
    grid = ((N + {{ BLOCK_SIZE }} - 1) // {{ BLOCK_SIZE }},)
    elemwise_relu_kernel[grid](A, C, N)
# fmt: on
