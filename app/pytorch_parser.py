"""
ROCmForge Studio — PyTorch Extension Parser

Detects and extracts embedded CUDA kernels from PyTorch C++ extensions.
Handles AT_DISPATCH_FLOATING_TYPES, torch::Tensor, and __global__ patterns.
"""

import re
from typing import Any, Dict, List


# ── Patterns for PyTorch extension detection ─────────────────────

_AT_DISPATCH_RE = re.compile(
    r'AT_DISPATCH_(?:FLOATING_TYPES|ALL_TYPES|INTEGRAL_TYPES)'
    r'(?:_AND_HALF)?'
    r'\s*\(\s*'
    r'([^,]+)'           # 1: tensor.scalar_type()
    r'\s*,\s*'
    r'"([^"]*)"'         # 2: op name string
    r'\s*,\s*'
    r'\[&\]\s*\{'        # lambda body start
)

_GLOBAL_KERNEL_RE = re.compile(
    r'(__global__\s+void\s+\w+\s*\([^)]*\)\s*\{)',
    re.DOTALL,
)

_TORCH_TENSOR_RE = re.compile(r'\btorch::Tensor\b')
_CUDA_ACCESSOR_RE = re.compile(r'\.packed_accessor\d*<')
_LAUNCH_RE = re.compile(r'<<<[^>]+>>>')


def parse_extension(code: str) -> Dict[str, Any]:
    """
    Parse a PyTorch C++ extension source and extract CUDA kernel info.

    Returns
    -------
    dict
        is_pytorch_extension : bool
        at_dispatch_calls    : list[dict] — detected AT_DISPATCH invocations
        embedded_kernels     : list[str]  — extracted __global__ kernel signatures
        cuda_launch_sites    : int        — count of <<<>>> launches
        torch_tensors        : int        — count of torch::Tensor references
        extracted_cuda_code  : str | None — the raw CUDA kernel code if found
    """
    result: Dict[str, Any] = {
        "is_pytorch_extension": False,
        "at_dispatch_calls": [],
        "embedded_kernels": [],
        "cuda_launch_sites": 0,
        "torch_tensors": 0,
        "extracted_cuda_code": None,
    }

    # Detect torch::Tensor usage
    tensor_matches = _TORCH_TENSOR_RE.findall(code)
    result["torch_tensors"] = len(tensor_matches)

    # Detect AT_DISPATCH calls
    for m in _AT_DISPATCH_RE.finditer(code):
        result["at_dispatch_calls"].append({
            "scalar_type_expr": m.group(1).strip(),
            "op_name": m.group(2),
            "position": m.start(),
        })

    # Detect __global__ kernels
    kernel_sigs: List[str] = []
    for m in _GLOBAL_KERNEL_RE.finditer(code):
        sig = m.group(1).strip()
        kernel_sigs.append(sig)
    result["embedded_kernels"] = kernel_sigs

    # Count kernel launches
    result["cuda_launch_sites"] = len(_LAUNCH_RE.findall(code))

    # Determine if it's a PyTorch extension
    result["is_pytorch_extension"] = (
        result["torch_tensors"] > 0
        or len(result["at_dispatch_calls"]) > 0
    )

    # Extract full CUDA kernel code blocks
    if kernel_sigs:
        cuda_blocks: List[str] = []
        for m in re.finditer(r'(__global__\s+void\s+\w+\s*\([^)]*\)\s*\{)', code):
            start = m.start()
            # Find matching closing brace
            depth = 0
            end = start
            for i in range(m.end() - 1, len(code)):
                if code[i] == '{':
                    depth += 1
                elif code[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            cuda_blocks.append(code[start:end])
        if cuda_blocks:
            result["extracted_cuda_code"] = "\n\n".join(cuda_blocks)

    return result
