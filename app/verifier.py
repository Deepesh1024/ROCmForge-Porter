"""
ROCmForge Studio — Verification Engine

Development-mode verification using NumPy:
  1. Generate random test tensors from metadata
  2. Compute reference output on CPU
  3. Mock ROCm output = reference + small noise (1e-6) to simulate drift
  4. Compute L2 norm between reference and mock
  5. Pass if L2 < 1e-5
"""

from typing import Any, Dict

import numpy as np

from app.utils import l2_norm


# ── Dtype mapping for NumPy ──────────────────────────────────────

_NP_DTYPE = {
    "float":  np.float32,
    "double": np.float64,
    "half":   np.float16,
    "int":    np.int32,
}

L2_THRESHOLD = 1e-5


def _make_tensors(primitive: str, dims: Dict[str, int], dtype: str):
    """Generate random input tensors appropriate for *primitive*."""
    np_dtype = _NP_DTYPE.get(dtype, np.float32)

    if primitive == "gemm":
        M = dims.get("M", 1024)
        N = dims.get("N", 1024)
        K = dims.get("K", 1024)
        A = np.random.randn(M, K).astype(np_dtype)
        B = np.random.randn(K, N).astype(np_dtype)
        return A, B
    elif primitive == "reduction":
        N = dims.get("N", 1024)
        A = np.random.randn(N).astype(np_dtype)
        return (A,)
    else:  # elementwise
        N = dims.get("N", 1024)
        A = np.random.randn(N).astype(np_dtype)
        B = np.random.randn(N).astype(np_dtype)
        return A, B


def _cpu_reference(primitive: str, tensors: tuple) -> np.ndarray:
    """Compute the reference result on CPU using NumPy."""
    if primitive == "gemm":
        A, B = tensors
        return (A.astype(np.float64) @ B.astype(np.float64)).astype(A.dtype)
    elif primitive == "reduction":
        (A,) = tensors
        return np.array([A.astype(np.float64).sum()], dtype=A.dtype)
    else:  # elementwise (a + b)
        A, B = tensors
        return (A.astype(np.float64) + B.astype(np.float64)).astype(A.dtype)


def _mock_rocm_output(reference: np.ndarray) -> np.ndarray:
    """Simulate ROCm device output by adding tiny noise to the reference."""
    ref_f64 = reference.astype(np.float64)
    # Scale noise relative to the reference magnitude to stay under L2 threshold
    scale = max(np.abs(ref_f64).max(), 1.0)
    noise = np.random.randn(*reference.shape).astype(np.float64) * (1e-7 / max(reference.size ** 0.5, 1.0)) * scale
    return (ref_f64 + noise).astype(reference.dtype)


def verify(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run development-mode verification.

    Parameters
    ----------
    meta : dict
        Must contain 'primitive', 'dtype'.  May contain 'dims'.

    Returns
    -------
    dict
        l2_norm        : float
        pass           : bool  (L2 < 1e-5)
        speed_ms       : float (placeholder)
        occupancy      : float (placeholder)
        bandwidth_gbps : float (placeholder)
    """
    primitive = meta.get("primitive", "elementwise")
    dtype     = meta.get("dtype", "float")
    dims      = meta.get("dims", {"M": 128, "N": 128, "K": 128})

    # Use smaller dims for verification to keep it fast
    capped = {k: min(v, 256) for k, v in dims.items()}

    tensors   = _make_tensors(primitive, capped, dtype)
    reference = _cpu_reference(primitive, tensors)
    mock_out  = _mock_rocm_output(reference)

    norm = l2_norm(reference, mock_out)

    return {
        "l2_norm":        round(norm, 10),
        "pass":           norm < L2_THRESHOLD,
        "speed_ms":       42.0,
        "occupancy":      72.0,
        "bandwidth_gbps": 1800.0,
    }
