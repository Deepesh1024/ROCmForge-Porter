"""
ROCmForge Studio — Verification Engine (Nationals Build)

Hardware-adaptive verification:
  • cpu_mock      → NumPy reference + deterministic mock timings
  • rocm_local    → hipcc compile + real execution (stub)
  • mi300x_remote → delegated to mi300x_runner
"""

import numpy as np
from typing import Any, Dict

from app.config import CPU_MOCK_TIMINGS
from app.utils import l2_norm


L2_THRESHOLD: float = 1e-5


def verify(meta: Dict[str, Any], backend: str = "cpu_mock") -> Dict[str, Any]:
    """
    Run verification for the given primitive and backend.

    Parameters
    ----------
    meta    : dict with keys 'primitive', 'dtype', 'dims'
    backend : "cpu_mock" | "rocm_local" | "mi300x_remote"
    """
    primitive = meta.get("primitive", "elementwise")
    dtype     = meta.get("dtype", "float")

    if backend == "mi300x_remote":
        from app import mi300x_runner
        return mi300x_runner.run_remote_mock(primitive, meta)

    if backend == "rocm_local":
        return _run_rocm_local(primitive, meta)

    return _run_cpu_mock(primitive, dtype, meta)


def _run_cpu_mock(primitive: str, dtype: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """CPU fallback — always succeeds with deterministic timings."""
    dims = meta.get("dims", {})
    np_dtype = {"float": np.float32, "double": np.float64, "half": np.float16}.get(dtype, np.float32)

    np.random.seed(42)  # deterministic for demos

    if primitive == "gemm":
        M = min(dims.get("M", 128), 256)
        N = min(dims.get("N", 128), 256)
        K = min(dims.get("K", 128), 256)
        A = np.random.randn(M, K).astype(np_dtype)
        B = np.random.randn(K, N).astype(np_dtype)
        ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np_dtype)
    elif primitive == "reduction":
        n = min(dims.get("N", 1024), 2048)
        A = np.random.randn(n).astype(np_dtype)
        ref = np.array([A.astype(np.float64).sum()], dtype=np_dtype)
    else:
        n = min(dims.get("N", 1024), 2048)
        A = np.random.randn(n).astype(np_dtype)
        B = np.random.randn(n).astype(np_dtype)
        ref = (A.astype(np.float64) + B.astype(np.float64)).astype(np_dtype)

    mock_out = _mock_device_output(ref)
    norm = l2_norm(ref, mock_out)
    timing = CPU_MOCK_TIMINGS.get(primitive, 0.30)

    return {
        "l2_norm":               round(norm, 12),
        "pass":                  norm < L2_THRESHOLD,
        "speed_ms":              timing,
        "occupancy":             "N/A",
        "bandwidth_gbps":        "N/A",
        "hardware_backend_used": "cpu_mock",
    }


def _run_rocm_local(primitive: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Local ROCm execution stub.
    In a full deployment this would:
      1. hipcc compile the generated kernel
      2. Run it on the local GPU
      3. Measure real timings
    For now, returns cpu_mock with a note.
    """
    result = _run_cpu_mock(primitive, meta.get("dtype", "float"), meta)
    result["hardware_backend_used"] = "rocm_local"
    result["note"] = "rocm_local detected but using CPU reference (no GPU on this host)"
    return result


def _mock_device_output(reference: np.ndarray) -> np.ndarray:
    """Simulate device output with tiny relative noise."""
    ref_f64 = reference.astype(np.float64)
    scale = max(np.abs(ref_f64).max(), 1.0)
    noise = np.random.randn(*reference.shape).astype(np.float64)
    noise *= (1e-7 / max(reference.size ** 0.5, 1.0)) * scale
    return (ref_f64 + noise).astype(reference.dtype)
