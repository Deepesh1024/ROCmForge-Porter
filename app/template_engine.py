"""
ROCmForge Studio — Template Engine

Loads HIP / Triton templates from the templates/ directory and fills
Jinja2-style placeholders:  {{ DTYPE }}, {{ DIMS }}, {{ TILE_SIZE }}, {{ BLOCK_SIZE }}
"""

import os
from typing import Any, Dict

from app.config import TEMPLATE_DIR

# Map primitive → (HIP template, Triton template)
_TEMPLATE_MAP: Dict[str, Dict[str, str]] = {
    "gemm": {
        "hip":    "gemm_hip_template.cpp",
        "triton": "gemm_triton_template.py",
    },
    "reduction": {
        "hip":    "reduction_hip_template.cpp",
        "triton": "reduction_triton_template.py",
    },
    "elementwise": {
        "hip":    "elemwise_hip_template.cpp",
        "triton": "elemwise_triton_template.py",
    },
}

# Default placeholder values
_DEFAULTS: Dict[str, str] = {
    "DTYPE":      "float",
    "DIMS":       "1024",
    "TILE_SIZE":  "16",
    "BLOCK_SIZE": "256",
}

# Dtype → C++ type mapping
_DTYPE_CPP: Dict[str, str] = {
    "float":  "float",
    "double": "double",
    "half":   "__half",
    "int":    "int",
}

# Dtype → Triton type mapping
_DTYPE_TRITON: Dict[str, str] = {
    "float":  "tl.float32",
    "double": "tl.float64",
    "half":   "tl.float16",
    "int":    "tl.int32",
}


def _load_template(filename: str) -> str:
    """Read a template file from disk."""
    path = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path, "r") as f:
        return f.read()


def _fill(template: str, replacements: Dict[str, str]) -> str:
    """Replace {{ KEY }} placeholders in *template*."""
    result = template
    for key, value in replacements.items():
        # Support both {{ KEY }} and {{KEY}}
        result = result.replace(f"{{{{ {key} }}}}", value)
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


def generate(primitive: str, meta: Dict[str, Any], backend: str = "hip") -> Dict[str, Any]:
    """
    Generate ROCm code for the given *primitive* using templates.

    Parameters
    ----------
    primitive : str
        "gemm", "reduction", or "elementwise"
    meta : dict
        Must contain at least "dtype".  May contain "dims".
    backend : str
        "hip" (default) or "triton".

    Returns
    -------
    dict
        rocm_code     : str — generated source code
        template_used : str — filename of the template used
    """
    if primitive not in _TEMPLATE_MAP:
        raise ValueError(f"Unknown primitive: {primitive}")

    if backend not in _TEMPLATE_MAP[primitive]:
        raise ValueError(f"No {backend} template for primitive: {primitive}")

    template_file = _TEMPLATE_MAP[primitive][backend]
    template_src  = _load_template(template_file)

    dtype_raw = meta.get("dtype", "float")
    dims      = meta.get("dims", {})

    # Build replacement map
    is_triton = backend == "triton"
    dtype_mapped = (_DTYPE_TRITON if is_triton else _DTYPE_CPP).get(dtype_raw, dtype_raw)

    replacements: Dict[str, str] = {
        "DTYPE":      dtype_mapped,
        "DIMS":       str(dims.get("M", dims.get("N", _DEFAULTS["DIMS"]))),
        "TILE_SIZE":  str(meta.get("tile_size", _DEFAULTS["TILE_SIZE"])),
        "BLOCK_SIZE": str(meta.get("block_size", _DEFAULTS["BLOCK_SIZE"])),
    }

    rocm_code = _fill(template_src, replacements)

    return {
        "rocm_code":     rocm_code,
        "template_used": template_file,
    }
