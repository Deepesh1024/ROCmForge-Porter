"""
ROCmForge Studio — Template Engine (Nationals Build)

Loads HIP C++ and Triton Python templates from the templates/ directory.
Extracts YAML-style metadata headers from template comments.
Fills placeholders ({{ DTYPE }}, {{ DIMS }}, etc.) deterministically.
NEVER uses LLM for code generation.
"""

import os
import re
from typing import Any, Dict, List, Tuple

from app.config import TEMPLATE_DIR

# ── Template filename map ────────────────────────────────────────

_TEMPLATE_MAP: Dict[str, str] = {
    "gemm":        "gemm_hip_template.cpp",
    "reduction":   "reduction_hip_template.cpp",
    "elementwise": "elemwise_hip_template.cpp",
}

_TRITON_TEMPLATE_MAP: Dict[str, str] = {
    "gemm":        "gemm_triton_template.py",
    "reduction":   "reduction_triton_template.py",
    "elementwise": "elemwise_triton_template.py",
}

# ── Placeholder defaults ─────────────────────────────────────────

_DTYPE_MAP = {
    "float":  "float",
    "double": "double",
    "half":   "__half",
    "int":    "int",
}

_TRITON_DTYPE_MAP = {
    "float":  "tl.float32",
    "double": "tl.float64",
    "half":   "tl.float16",
}


def _extract_metadata(template_text: str) -> Dict[str, str]:
    """
    Extract YAML-style metadata from the template header comment block.

    Looks for lines like:
        * Primitive: GEMM
        * Source: AMD ROCm 7.2 ...
        * Safety notes: ...

    Returns dict of key→value.
    """
    metadata: Dict[str, str] = {}
    header_match = re.search(r'/\*(.+?)\*/', template_text, re.DOTALL)
    if not header_match:
        return metadata

    for line in header_match.group(1).splitlines():
        line = line.strip().lstrip("*").strip()
        if ":" in line and not line.startswith("Placeholders"):
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key and value:
                metadata[key] = value

    return metadata


def _fill_placeholders(template: str, meta: Dict[str, Any], dtype_map: Dict[str, str]) -> str:
    """Replace all {{ PLACEHOLDER }} tokens."""
    dtype = meta.get("dtype", "float")
    dims = meta.get("dims", {})

    dim_val = dims.get("M", dims.get("N", 1024))
    tile_size = meta.get("tile_size", 16)
    block_size = meta.get("block_size", 256)

    replacements = {
        "{{ DTYPE }}":      dtype_map.get(dtype, dtype),
        "{{ DIMS }}":       str(dim_val),
        "{{ TILE_SIZE }}":  str(tile_size),
        "{{ BLOCK_SIZE }}": str(block_size),
    }

    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result


def generate(primitive: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate ROCm code from templates.

    Returns dict with:
        rocm_code      — filled HIP C++ template
        triton_code    — filled Triton Python template (if available)
        template_used  — filename of the HIP template
        metadata       — extracted template metadata dict
    """
    # HIP template
    hip_filename = _TEMPLATE_MAP.get(primitive, _TEMPLATE_MAP["elementwise"])
    hip_path = os.path.join(TEMPLATE_DIR, hip_filename)

    with open(hip_path) as f:
        hip_template = f.read()

    metadata = _extract_metadata(hip_template)
    rocm_code = _fill_placeholders(hip_template, meta, _DTYPE_MAP)

    # Triton template (optional)
    triton_code = None
    triton_filename = _TRITON_TEMPLATE_MAP.get(primitive)
    if triton_filename:
        triton_path = os.path.join(TEMPLATE_DIR, triton_filename)
        if os.path.isfile(triton_path):
            with open(triton_path) as f:
                triton_template = f.read()
            triton_code = _fill_placeholders(triton_template, meta, _TRITON_DTYPE_MAP)

    return {
        "rocm_code":     rocm_code,
        "triton_code":   triton_code,
        "template_used": hip_filename,
        "metadata":      metadata,
    }
