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

TEMPLATE_MAP: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("gemm", "tiled_shared"): {
        "hip_template": "tiled_gemm_hip_template.cpp",
        "triton_template": "tiled_gemm_triton_template.py",
        "changes": [
            "Applied shared memory tiling for improved cache locality",
            "Configured block sizes for wave64 occupancy",
            "Swapped cuBLAS calls for rocBLAS equivalents"
        ]
    },
    ("fused_matmul", "fused_relu"): {
        "hip_template": "fused_matmul_relu_hip_template.cpp",
        "triton_template": "fused_matmul_relu_triton_template.py",
        "changes": [
            "Fused ReLU activation into GEMM epilogue",
            "Eliminated intermediate global memory allocation",
            "Optimized memory alignment for vector operations"
        ]
    },
    ("elementwise", "vectorized"): {
        "hip_template": "vectorized_elementwise_hip_template.cpp",
        "triton_template": "vectorized_elementwise_triton_template.py",
        "changes": [
            "Vectorized global memory load/store operations (float4/double2)",
            "Ensured continuous memory access patterns",
            "Added robust boundary checks for unaligned tails"
        ]
    },
    ("reduction", "wavefront_reduce"): {
        "hip_template": "wavefront_reduction_hip_template.cpp",
        "triton_template": "wavefront_reduction_triton_template.py",
        "changes": [
            "Replaced shared array syncs with native wave64 shuffle instructions",
            "Optimized multi-block reduce logic for hardware execution",
            "Mitigated LDS bank conflicts in hierarchical sum"
        ]
    },
    ("softmax", "fused_softmax_reduce"): {
        "hip_template": "softmax_hip_template.cpp",
        "triton_template": "softmax_triton_template.py",
        "changes": [
            "Fused exponentiation and normalization stages",
            "Implemented cross-lane Wave64 intrinsic reductions",
            "Replaced global memory syncs with warp primitives"
        ]
    },
    ("layernorm", "fused_layernorm"): {
        "hip_template": "layernorm_hip_template.cpp",
        "triton_template": "layernorm_triton_template.py",
        "changes": [
            "Fused mean and variance accumulation in single-pass",
            "Substituted software math with native rsqrtf",
            "Applied vector-width memory alignments"
        ]
    },
    ("conv", "direct_conv"): {
        "hip_template": "conv_hip_template.cpp",
        "triton_template": "conv_triton_template.py",
        "changes": [
            "Mapped explicit filter operations into MFMA core loops",
            "Optimized halo regions for padding constraints",
            "Scheduled register allocation for maximum wave occupancy"
        ]
    },
    ("attention", "flash_attention"): {
        "hip_template": "attention_hip_template.cpp",
        "triton_template": "attention_triton_template.py",
        "changes": [
            "Deployed FlashAttention structural optimizations",
            "Blocked SRAM logic for Q, K, V independent tiles",
            "Aligned scale factor mathematics to fp16 fast-math routines"
        ]
    },
    ("dropout", "fused_dropout"): {
        "hip_template": "dropout_hip_template.cpp",
        "triton_template": "dropout_triton_template.py",
        "changes": [
            "Applied deterministic hardware-level RNG seeding logic",
            "Scaled scaling-factors aggressively within register space",
            "Eliminated branching divergence in evaluation stages"
        ]
    }
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


def _fill_placeholders(template: str, meta: Dict[str, Any], dtype_map: Dict[str, str], primitive: str, pattern: str) -> str:
    """Replace all {{ PLACEHOLDER }} tokens."""
    dtype = meta.get("dtype", "float")
    dims = meta.get("dims", {})

    dim_val = dims.get("M", dims.get("N", 1024))
    
    # Auto-Variant Selection Logic based on dimensions
    tile_size = 16
    block_size = 256
    variant_note = "Using generic baseline."
    
    if primitive == "gemm":
        m, n, k = dims.get("M", 1024), dims.get("N", 1024), dims.get("K", 1024)
        if m >= 4096 and n >= 4096:
            block_size = 256
            tile_size = 32 # switch to large tile configuration
            variant_note = "Switched to 32x32 TILE_SIZE due to massive M/N boundaries."
        elif k < 256:
            block_size = 64
            tile_size = 8  # skinny matrix configuration
            variant_note = "Skinny matrix detected, swapped to 64/8 register allocation limits."
    elif primitive == "reduction" or primitive == "softmax":
        n = dims.get("N", 1024)
        if n <= 64:
            block_size = 64
            variant_note = "Small target dimensions mapped natively to single wave64."
        else:
            block_size = 256
            variant_note = "Hierarchical sum deployed across 256 thread block limits."

    meta["tuner_variants_applied"] = variant_note

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
        rocm_code       — filled HIP C++ template
        triton_code     — filled Triton Python template (if available)
        template_used   — filename of the HIP template
        metadata        — extracted template metadata dict
        changes_applied — specific changes made based on pattern
    """
    pattern = meta.get("pattern")
    # Provide default patterns if None 
    if not pattern:
        if primitive == "gemm": pattern = "tiled_shared"
        elif primitive == "fused_matmul": pattern = "fused_relu"
        elif primitive == "reduction": pattern = "wavefront_reduce"
        elif primitive == "softmax": pattern = "fused_softmax_reduce"
        elif primitive == "layernorm": pattern = "fused_layernorm"
        elif primitive == "conv": pattern = "direct_conv"
        elif primitive == "attention": pattern = "flash_attention"
        elif primitive == "dropout": pattern = "fused_dropout"
        else: pattern = "vectorized"
        
    mapping = TEMPLATE_MAP.get((primitive, pattern))
    if not mapping:
        mapping = TEMPLATE_MAP[("elementwise", "vectorized")]
        
    hip_filename = mapping["hip_template"]
    hip_path = os.path.join(TEMPLATE_DIR, hip_filename)

    with open(hip_path) as f:
        hip_template = f.read()

    metadata = _extract_metadata(hip_template)
    rocm_code = _fill_placeholders(hip_template, meta, _DTYPE_MAP, primitive, pattern)
    
    # Inject auto tuning note into changes applied
    changes_arr = mapping["changes"].copy()
    if "tuner_variants_applied" in meta:
        changes_arr.append(f"Auto-Tuner: {meta['tuner_variants_applied']}")

    # Triton template (optional)
    triton_code = None
    triton_filename = mapping.get("triton_template")
    if triton_filename:
        triton_path = os.path.join(TEMPLATE_DIR, triton_filename)
        if os.path.isfile(triton_path):
            with open(triton_path) as f:
                triton_template = f.read()
            triton_code = _fill_placeholders(triton_template, meta, _TRITON_DTYPE_MAP, primitive, pattern)

    return {
        "rocm_code":       rocm_code,
        "triton_code":     triton_code,
        "template_used":   hip_filename,
        "metadata":        metadata,
        "changes_applied": changes_arr,
    }
