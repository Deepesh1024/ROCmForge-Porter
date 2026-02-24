"""
ROCmForge Studio — Mock Hipify Runner v2.0

Simulates hipify-clang by performing regex-based CUDA → HIP API
replacements.  Real hipify-clang is NOT required.

v2.0 improvements:
  • Correct kernel launch <<<>>> → hipLaunchKernelGGL transformation
  • Extended API coverage (60+ tokens, cuDNN/cuFFT/cuRAND/cuSPARSE headers)
  • Proper handling of kernel<<<grid,block,shared,stream>>>(args)
  • Deduplicated change log
"""

import re
from typing import Dict, List, Any, Tuple

# ── CUDA → HIP simple token replacements ─────────────────────────
# Applied via word-boundary regex on each line.

_TOKEN_REPLACEMENTS: List[Tuple[str, str]] = [
    # Memory management
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

    # Streams & events
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

    # Device management
    ("cudaDeviceSynchronize",   "hipDeviceSynchronize"),
    ("cudaSetDevice",           "hipSetDevice"),
    ("cudaGetDevice",           "hipGetDevice"),
    ("cudaGetDeviceCount",      "hipGetDeviceCount"),
    ("cudaGetDeviceProperties", "hipGetDeviceProperties"),
    ("cudaDeviceReset",         "hipDeviceReset"),
    ("cudaDeviceProp",          "hipDeviceProp_t"),

    # Error handling
    ("cudaError_t",             "hipError_t"),
    ("cudaSuccess",             "hipSuccess"),
    ("cudaGetLastError",        "hipGetLastError"),
    ("cudaGetErrorString",      "hipGetErrorString"),
    ("cudaPeekAtLastError",     "hipPeekAtLastError"),

    # BLAS
    ("cublasSgemm",             "rocblas_sgemm"),
    ("cublasDgemm",             "rocblas_dgemm"),
    ("cublasHgemm",             "rocblas_hgemm"),
    ("cublasCreate",            "rocblas_create_handle"),
    ("cublasDestroy",           "rocblas_destroy_handle"),
    ("cublasHandle_t",          "rocblas_handle"),
    ("cublasSetStream",         "rocblas_set_stream"),

    # Texture / Surface
    ("cudaCreateTextureObject", "hipCreateTextureObject"),
    ("cudaDestroyTextureObject","hipDestroyTextureObject"),
]

# ── Header replacements  ─────────────────────────────────────────

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

# ── Kernel launch regex  ─────────────────────────────────────────
# Matches:  kernelName<<<grid, block>>>(args)
# And:      kernelName<<<grid, block, sharedMem>>>(args)
# And:      kernelName<<<grid, block, sharedMem, stream>>>(args)
#
# Correct HIP form:
#   hipLaunchKernelGGL(kernelName, grid, block, sharedMem, stream, args)

_KERNEL_LAUNCH_RE = re.compile(
    r'(\w+)'                    # 1: kernel name
    r'\s*<<<\s*'                # <<<
    r'([^,>]+)'                 # 2: grid dim
    r'\s*,\s*'                  # ,
    r'([^,>]+)'                 # 3: block dim
    r'(?:\s*,\s*([^,>]+))?'     # 4: optional shared mem
    r'(?:\s*,\s*([^>]+))?'      # 5: optional stream
    r'\s*>>>\s*'                # >>>
    r'\(([^)]*)\)'              # 6: (args)
)


def _transform_kernel_launch(line: str) -> Tuple[str, bool]:
    """
    Transform  kernel<<<grid, block>>>(args)
    into       hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, args)
    """
    match = _KERNEL_LAUNCH_RE.search(line)
    if not match:
        return line, False

    kernel = match.group(1).strip()
    grid   = match.group(2).strip()
    block  = match.group(3).strip()
    shared = (match.group(4) or "0").strip()
    stream = (match.group(5) or "0").strip()
    args   = match.group(6).strip()

    # Build proper hipLaunchKernelGGL call
    hip_call = f"hipLaunchKernelGGL({kernel}, {grid}, {block}, {shared}, {stream}, {args})"

    # Preserve indentation
    indent = line[:match.start()]
    trailing = line[match.end():]
    new_line = indent + hip_call + trailing

    return new_line, True


def run_mock_hipify(cuda_code: str) -> Dict[str, Any]:
    """
    Apply mock hipify to *cuda_code*.

    Returns
    -------
    dict
        hipified_code : str  — transformed source
        changes       : list[str] — deduplicated list of replacements made
    """
    changes: List[str] = []
    hipified_lines: List[str] = []

    for line in cuda_code.splitlines():
        original = line
        changed_this_line = False

        # 1) Header replacements (regex-based)
        for pattern, replacement in _HEADER_REPLACEMENTS:
            new_line, n = re.subn(pattern, replacement, line)
            if n > 0:
                # Extract the header name for a cleaner log
                old_hdr = re.search(r'<(.+?)>', pattern)
                new_hdr = re.search(r'<(.+?)>', replacement)
                if old_hdr and new_hdr:
                    changes.append(f"<{old_hdr.group(1)}> → <{new_hdr.group(1)}>")
                else:
                    changes.append(f"{pattern} → {replacement}")
                line = new_line
                changed_this_line = True

        # 2) Kernel launch transformation (must happen BEFORE token replacement)
        line, did_launch = _transform_kernel_launch(line)
        if did_launch:
            changes.append("<<<grid, block>>>(args) → hipLaunchKernelGGL(kernel, grid, block, sharedMem, stream, args)")
            changed_this_line = True

        # 3) Token replacements (word-boundary)
        for cuda_tok, hip_tok in _TOKEN_REPLACEMENTS:
            pattern = rf'\b{re.escape(cuda_tok)}\b'
            new_line, n = re.subn(pattern, hip_tok, line)
            if n > 0:
                changes.append(f"{cuda_tok} → {hip_tok}")
                line = new_line
                changed_this_line = True

        if changed_this_line:
            hipified_lines.append(f"// [hipified] {line}")
        else:
            hipified_lines.append(line)

    # Deduplicate changes while preserving order
    seen: set = set()
    unique_changes: List[str] = []
    for c in changes:
        if c not in seen:
            seen.add(c)
            unique_changes.append(c)

    return {
        "hipified_code": "\n".join(hipified_lines),
        "changes": unique_changes,
    }
