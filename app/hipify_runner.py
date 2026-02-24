"""
ROCmForge Studio — Hipify Runner (Nationals Build)

Tries real hipify-clang subprocess first.
Falls back to safe mock hipify (regex) if not found.
"""

import re
import shutil
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Tuple


# ── CUDA → HIP token replacements ───────────────────────────────

_TOKEN_REPLACEMENTS: List[Tuple[str, str]] = [
    ("cudaMalloc",              "hipMalloc"),
    ("cudaMallocManaged",       "hipMallocManaged"),
    ("cudaFree",                "hipFree"),
    ("cudaMemcpy",              "hipMemcpy"),
    ("cudaMemcpyAsync",         "hipMemcpyAsync"),
    ("cudaMemcpyHostToDevice",  "hipMemcpyHostToDevice"),
    ("cudaMemcpyDeviceToHost",  "hipMemcpyDeviceToHost"),
    ("cudaMemcpyDeviceToDevice","hipMemcpyDeviceToDevice"),
    ("cudaMemset",              "hipMemset"),
    ("cudaMemsetAsync",         "hipMemsetAsync"),
    ("cudaStream_t",            "hipStream_t"),
    ("cudaStreamCreate",        "hipStreamCreate"),
    ("cudaStreamSynchronize",   "hipStreamSynchronize"),
    ("cudaStreamDestroy",       "hipStreamDestroy"),
    ("cudaStreamWaitEvent",     "hipStreamWaitEvent"),
    ("cudaEvent_t",             "hipEvent_t"),
    ("cudaEventCreate",         "hipEventCreate"),
    ("cudaEventRecord",         "hipEventRecord"),
    ("cudaEventSynchronize",    "hipEventSynchronize"),
    ("cudaEventElapsedTime",    "hipEventElapsedTime"),
    ("cudaEventDestroy",        "hipEventDestroy"),
    ("cudaDeviceSynchronize",   "hipDeviceSynchronize"),
    ("cudaSetDevice",           "hipSetDevice"),
    ("cudaGetDevice",           "hipGetDevice"),
    ("cudaGetDeviceCount",      "hipGetDeviceCount"),
    ("cudaGetDeviceProperties", "hipGetDeviceProperties"),
    ("cudaDeviceReset",         "hipDeviceReset"),
    ("cudaDeviceProp",          "hipDeviceProp_t"),
    ("cudaError_t",             "hipError_t"),
    ("cudaSuccess",             "hipSuccess"),
    ("cudaGetLastError",        "hipGetLastError"),
    ("cudaGetErrorString",      "hipGetErrorString"),
    ("cudaPeekAtLastError",     "hipPeekAtLastError"),
    ("cublasSgemm",             "rocblas_sgemm"),
    ("cublasDgemm",             "rocblas_dgemm"),
    ("cublasHgemm",             "rocblas_hgemm"),
    ("cublasCreate",            "rocblas_create_handle"),
    ("cublasDestroy",           "rocblas_destroy_handle"),
    ("cublasHandle_t",          "rocblas_handle"),
    ("cublasSetStream",         "rocblas_set_stream"),
    ("cudaCreateTextureObject", "hipCreateTextureObject"),
    ("cudaDestroyTextureObject","hipDestroyTextureObject"),
]

_HEADER_REPLACEMENTS: List[Tuple[str, str]] = [
    (r'#include\s*<cuda_runtime\.h>',      '#include <hip/hip_runtime.h>'),
    (r'#include\s*<cuda\.h>',               '#include <hip/hip_runtime.h>'),
    (r'#include\s*<cuda_runtime_api\.h>',  '#include <hip/hip_runtime_api.h>'),
    (r'#include\s*<cublas_v2\.h>',          '#include <rocblas/rocblas.h>'),
    (r'#include\s*<cublas\.h>',             '#include <rocblas/rocblas.h>'),
    (r'#include\s*<cufft\.h>',              '#include <rocfft/rocfft.h>'),
    (r'#include\s*<curand\.h>',             '#include <rocrand/rocrand.h>'),
    (r'#include\s*<curand_kernel\.h>',      '#include <rocrand/rocrand_kernel.h>'),
    (r'#include\s*<cusparse\.h>',           '#include <rocsparse/rocsparse.h>'),
    (r'#include\s*<cudnn\.h>',              '#include <miopen/miopen.h>'),
]

_KERNEL_LAUNCH_RE = re.compile(
    r'(\w+)'
    r'\s*<<<\s*'
    r'([^,>]+)'
    r'\s*,\s*'
    r'([^,>]+)'
    r'(?:\s*,\s*([^,>]+))?'
    r'(?:\s*,\s*([^>]+))?'
    r'\s*>>>\s*'
    r'\(([^)]*)\)'
)


def _transform_kernel_launch(line: str) -> Tuple[str, bool]:
    """Transform <<<>>> to hipLaunchKernelGGL."""
    match = _KERNEL_LAUNCH_RE.search(line)
    if not match:
        return line, False
    kernel = match.group(1).strip()
    grid   = match.group(2).strip()
    block  = match.group(3).strip()
    shared = (match.group(4) or "0").strip()
    stream = (match.group(5) or "0").strip()
    args   = match.group(6).strip()
    hip_call = f"hipLaunchKernelGGL({kernel}, {grid}, {block}, {shared}, {stream}, {args})"
    return line[:match.start()] + hip_call + line[match.end():], True


def _try_real_hipify(cuda_code: str) -> Dict[str, Any] | None:
    """
    Attempt to use the real hipify-clang binary.
    Returns result dict on success, None on failure.
    """
    if not shutil.which("hipify-clang"):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(cuda_code)
            tmp_path = f.name
        result = subprocess.run(
            ["hipify-clang", tmp_path, "--"],
            capture_output=True, timeout=15, text=True,
        )
        os.unlink(tmp_path)
        if result.returncode == 0:
            return {
                "hipified_code": result.stdout,
                "changes": ["hipify-clang applied all transformations"],
                "method": "hipify-clang (real)",
            }
    except Exception:
        pass
    return None


def _mock_hipify(cuda_code: str) -> Dict[str, Any]:
    """Apply regex-based mock hipify."""
    changes: List[str] = []
    hipified_lines: List[str] = []

    for line in cuda_code.splitlines():
        original = line
        changed = False

        for pattern, replacement in _HEADER_REPLACEMENTS:
            new_line, n = re.subn(pattern, replacement, line)
            if n > 0:
                old_h = re.search(r'<(.+?)>', pattern)
                new_h = re.search(r'<(.+?)>', replacement)
                if old_h and new_h:
                    changes.append(f"<{old_h.group(1)}> → <{new_h.group(1)}>")
                line = new_line
                changed = True

        line, did_launch = _transform_kernel_launch(line)
        if did_launch:
            changes.append("<<<grid, block>>>(args) → hipLaunchKernelGGL(kernel, grid, block, shared, stream, args)")
            changed = True

        for cuda_tok, hip_tok in _TOKEN_REPLACEMENTS:
            pat = rf'\b{re.escape(cuda_tok)}\b'
            new_line, n = re.subn(pat, hip_tok, line)
            if n > 0:
                changes.append(f"{cuda_tok} → {hip_tok}")
                line = new_line
                changed = True

        hipified_lines.append(f"// [hipified] {line}" if changed else line)

    seen: set = set()
    unique = [c for c in changes if not (c in seen or seen.add(c))]

    return {
        "hipified_code": "\n".join(hipified_lines),
        "changes": unique,
        "method": "mock hipify (regex)",
    }


def run_hipify(cuda_code: str) -> Dict[str, Any]:
    """
    Run hipify on CUDA code.
    Tries real hipify-clang first; falls back to mock.
    """
    real = _try_real_hipify(cuda_code)
    if real is not None:
        return real
    return _mock_hipify(cuda_code)
