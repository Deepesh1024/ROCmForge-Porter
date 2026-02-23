"""
ROCmForge Studio — Mock Hipify Runner

Simulates hipify-clang by performing regex-based CUDA → HIP API
replacements.  Real hipify-clang is NOT required.
"""

import re
from typing import Dict, List, Any

# ── CUDA → HIP replacement map ──────────────────────────────────
_REPLACEMENTS: List[tuple] = [
    # Memory management
    (r'\bcudaMalloc\b',           'hipMalloc'),
    (r'\bcudaFree\b',             'hipFree'),
    (r'\bcudaMemcpy\b',           'hipMemcpy'),
    (r'\bcudaMemcpyHostToDevice\b', 'hipMemcpyHostToDevice'),
    (r'\bcudaMemcpyDeviceToHost\b', 'hipMemcpyDeviceToHost'),
    (r'\bcudaMemset\b',           'hipMemset'),

    # Streams & events
    (r'\bcudaStream_t\b',         'hipStream_t'),
    (r'\bcudaStreamCreate\b',     'hipStreamCreate'),
    (r'\bcudaStreamSynchronize\b','hipStreamSynchronize'),
    (r'\bcudaStreamDestroy\b',    'hipStreamDestroy'),
    (r'\bcudaEvent_t\b',          'hipEvent_t'),
    (r'\bcudaEventCreate\b',      'hipEventCreate'),
    (r'\bcudaEventRecord\b',      'hipEventRecord'),
    (r'\bcudaEventSynchronize\b', 'hipEventSynchronize'),
    (r'\bcudaEventElapsedTime\b', 'hipEventElapsedTime'),
    (r'\bcudaEventDestroy\b',     'hipEventDestroy'),

    # Device management
    (r'\bcudaDeviceSynchronize\b','hipDeviceSynchronize'),
    (r'\bcudaSetDevice\b',        'hipSetDevice'),
    (r'\bcudaGetDevice\b',        'hipGetDevice'),
    (r'\bcudaGetDeviceProperties\b', 'hipGetDeviceProperties'),

    # Error handling
    (r'\bcudaError_t\b',          'hipError_t'),
    (r'\bcudaSuccess\b',          'hipSuccess'),
    (r'\bcudaGetLastError\b',     'hipGetLastError'),
    (r'\bcudaGetErrorString\b',   'hipGetErrorString'),

    # BLAS
    (r'\bcublasSgemm\b',          'rocblas_sgemm'),
    (r'\bcublasCreate\b',         'rocblas_create_handle'),
    (r'\bcublasDestroy\b',        'rocblas_destroy_handle'),
    (r'\bcublasHandle_t\b',       'rocblas_handle'),

    # Kernel launch  <<<grid, block>>>  →  hipLaunchKernelGGL
    (r'<<<\s*(.+?)\s*,\s*(.+?)\s*>>>', r'hipLaunchKernelGGL(/* \1, \2 */)'),

    # Headers
    (r'#include\s*<cuda_runtime\.h>', '#include <hip/hip_runtime.h>'),
    (r'#include\s*<cuda\.h>',          '#include <hip/hip_runtime.h>'),
    (r'#include\s*<cublas_v2\.h>',     '#include <rocblas/rocblas.h>'),

    # Qualifiers (these stay the same in HIP, but we still log them)
    # __syncthreads() is identical in HIP — no replacement needed.
]


def run_mock_hipify(cuda_code: str) -> Dict[str, Any]:
    """
    Apply mock hipify to *cuda_code*.

    Returns
    -------
    dict
        hipified_code : str — transformed source
        changes       : list[str] — human-readable list of replacements made
    """
    changes: List[str] = []
    hipified_lines: List[str] = []

    for line in cuda_code.splitlines():
        original = line
        for pattern, replacement in _REPLACEMENTS:
            new_line, n = re.subn(pattern, replacement, line)
            if n > 0:
                changes.append(f"{pattern} → {replacement} ({n}×)")
                line = new_line

        if line != original:
            hipified_lines.append(f"// hipified: {line}")
        else:
            hipified_lines.append(line)

    return {
        "hipified_code": "\n".join(hipified_lines),
        "changes": changes,
    }
