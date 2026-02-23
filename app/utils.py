"""
ROCmForge Studio — Utility Helpers
"""

import json
import tempfile
import os
from typing import Any

import numpy as np


def l2_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the L2 (Euclidean) norm between two arrays."""
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def safe_json_dump(obj: Any) -> str:
    """Serialise *obj* to a JSON string, converting non-serialisable types."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return str(o)

    return json.dumps(obj, indent=2, default=_default)


def write_temp_file(content: str, suffix: str = ".cu") -> str:
    """Write *content* to a temporary file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
    except Exception:
        os.close(fd)
        raise
    return path
