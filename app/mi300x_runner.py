"""
ROCmForge Studio — MI300X Remote Runner

Handles MI300X droplet registration and remote mock execution.
In DEV MODE, returns deterministic mock timings.
"""

import json
import os
from typing import Any, Dict

from app.config import MI300X_CONFIG_PATH, MI300X_MOCK_TIMINGS


def register_droplet(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save MI300X droplet configuration to mi300x_config.json.

    Returns status dict.
    """
    os.makedirs(os.path.dirname(MI300X_CONFIG_PATH) or ".", exist_ok=True)

    with open(MI300X_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    return {
        "registered": True,
        "config_path": MI300X_CONFIG_PATH,
        "size": config.get("size", "unknown"),
        "image": config.get("image", "unknown"),
    }


def is_registered() -> bool:
    """Check if an MI300X droplet is registered."""
    if not os.path.isfile(MI300X_CONFIG_PATH):
        return False
    try:
        with open(MI300X_CONFIG_PATH) as f:
            data = json.load(f)
        return isinstance(data, dict) and data.get("size", "") != ""
    except Exception:
        return False


def get_config() -> Dict[str, Any] | None:
    """Read the registered MI300X config, or None."""
    if not is_registered():
        return None
    with open(MI300X_CONFIG_PATH) as f:
        return json.load(f)


def run_remote_mock(primitive: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEV MODE: simulate MI300X remote execution with deterministic timings.

    Returns verification-compatible result dict.
    """
    import numpy as np
    from app.utils import l2_norm

    dtype = meta.get("dtype", "float")
    dims = meta.get("dims", {"M": 128, "N": 128, "K": 128})

    # Generate reference on CPU
    np_dtype = {"float": np.float32, "double": np.float64, "half": np.float16}.get(dtype, np.float32)

    if primitive == "gemm":
        M, N, K = dims.get("M", 128), dims.get("N", 128), dims.get("K", 128)
        M, N, K = min(M, 256), min(N, 256), min(K, 256)
        A = np.random.randn(M, K).astype(np_dtype)
        B = np.random.randn(K, N).astype(np_dtype)
        ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np_dtype)
    elif primitive == "reduction":
        n = min(dims.get("N", 1024), 1024)
        A = np.random.randn(n).astype(np_dtype)
        ref = np.array([A.astype(np.float64).sum()], dtype=np_dtype)
    else:
        n = min(dims.get("N", 1024), 1024)
        A = np.random.randn(n).astype(np_dtype)
        B = np.random.randn(n).astype(np_dtype)
        ref = (A.astype(np.float64) + B.astype(np.float64)).astype(np_dtype)

    # Mock MI300X output with very tiny noise
    ref_f64 = ref.astype(np.float64)
    scale = max(np.abs(ref_f64).max(), 1.0)
    noise = np.random.randn(*ref.shape) * (1e-8 / max(ref.size ** 0.5, 1.0)) * scale
    mock_out = (ref_f64 + noise).astype(ref.dtype)

    norm = l2_norm(ref, mock_out)
    timing = MI300X_MOCK_TIMINGS.get(primitive, 0.12)

    return {
        "l2_norm": round(norm, 12),
        "pass": norm < 1e-5,
        "speed_ms": timing,
        "occupancy": "82% (estimated)",
        "bandwidth_gbps": "3200 (estimated)",
        "hardware_backend_used": "mi300x_remote",
    }
