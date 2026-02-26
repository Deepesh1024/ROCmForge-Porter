"""
ROCmForge Studio — CUDA Primitive Classifier

Deterministic, regex-based classifier that detects:
  • GEMM (matmul patterns, cuBLAS calls)
  • Reduction (sum, max, atomicAdd, warp-level reductions)
  • Elementwise (pointwise / element-wise ops)
"""

import re
from typing import Any, Dict, List, Optional

# ── Regex pattern banks ──────────────────────────────────────────

_GEMM_PATTERNS: List[str] = [
    r'\bmatmul\b',
    r'\bgemm\b',
    r'\bcublasSgemm\b',
    r'\bcublasDgemm\b',
    r'\bcublasHgemm\b',
    r'\brocblas_sgemm\b',
    r'\brocblas_dgemm\b',
    r'\b__global__\s+void\s+matmul\b',
    r'\bshared\s+\w+\s+\w+\s*\[',       # shared-memory tiling (heuristic)
    r'\bfor\s*\(.*\bk\b.*\)',             # k-loop in tiled matmul (heuristic)
]

_REDUCTION_PATTERNS: List[str] = [
    r'\batomicAdd\b',
    r'\b__reduce_add_sync\b',
    r'\b__shfl_down_sync\b',
    r'\b__shfl_xor_sync\b',
    r'\breduction\b',
    r'\breduce\b',
    r'\bsum\b',
    r'\bmax\b',
    r'\bmin\b',
    r'\bwarpReduce\b',
    r'\bblockReduce\b',
]

_ELEMENTWISE_PATTERNS: List[str] = [
    r'\belementwise\b',
    r'\bpointwise\b',
    r'\b__global__\s+void\s+\w*(add|mul|sub|div|relu|sigmoid|tanh|scale|bias)\w*\b',
    r'\bout\s*\[\s*idx\s*\]\s*=',        # out[idx] = … pattern
    r'\bc\s*\[\s*i\s*\]\s*=\s*a\s*\[\s*i\s*\]',  # c[i] = a[i] …
]

# ── Data-type detection ──────────────────────────────────────────

_DTYPE_PATTERNS: Dict[str, str] = {
    r'\bhalf\b':    "half",
    r'\b__half\b':  "half",
    r'\bfloat16\b': "half",
    r'\bfp16\b':    "half",
    r'\bfloat\b':   "float",
    r'\bfp32\b':    "float",
    r'\bdouble\b':  "double",
    r'\bfp64\b':    "double",
    r'\bint\b':     "int",
    r'\bint32_t\b': "int",
}

# ── Dimension / shape extraction (best effort) ──────────────────

_DIM_PATTERN = re.compile(
    r'(?:M|N|K|rows|cols|width|height|dim|size)\s*=\s*(\d+)', re.IGNORECASE
)


def _detect_dtype(code: str) -> str:
    """Return the first matching dtype found in *code*, default 'float'."""
    for pattern, dtype in _DTYPE_PATTERNS.items():
        if re.search(pattern, code):
            return dtype
    return "float"


def _extract_dims(code: str) -> Dict[str, int]:
    """Best-effort extraction of named dimensions from *code*."""
    dims: Dict[str, int] = {}
    for m in _DIM_PATTERN.finditer(code):
        name = m.group(0).split("=")[0].strip().upper()
        dims[name] = int(m.group(1))
    return dims


def _score_patterns(code: str, patterns: List[str]) -> int:
    """Count how many patterns from *patterns* match in *code*."""
    return sum(1 for p in patterns if re.search(p, code, re.IGNORECASE))


def classify(code: str) -> Dict[str, Any]:
    """
    Classify *code* as gemm / reduction / elementwise / fused_matmul.

    Returns
    -------
    dict
        primitive : str   — "gemm", "reduction", "elementwise", or "fused_matmul"
        dtype     : str   — detected data type
        shape     : str   — shape format (e.g. "M=1024, N=1024, K=1024")
        pattern   : str   — detected semantic pattern
        meta      : dict  — additional metadata (dims, scores)
    """
    gemm_score = _score_patterns(code, _GEMM_PATTERNS)
    red_score  = _score_patterns(code, _REDUCTION_PATTERNS)
    elem_score = _score_patterns(code, _ELEMENTWISE_PATTERNS)

    scores = {"gemm": gemm_score, "reduction": red_score, "elementwise": elem_score}

    # Pick the primitive with the highest score; default to elementwise
    primitive = max(scores, key=scores.get)  # type: ignore[arg-type]
    if all(v == 0 for v in scores.values()):
        primitive = "elementwise"

    dtype = _detect_dtype(code)
    dims  = _extract_dims(code)

    # Detect fused_matmul if gemm has a relu
    if primitive == "gemm" and re.search(r'\brelu\b', code, re.IGNORECASE):
        primitive = "fused_matmul"

    # Fallback dims when none detected
    if not dims:
        if primitive in ("gemm", "fused_matmul"):
            dims = {"M": 1024, "N": 1024, "K": 1024}
        elif primitive == "reduction":
            dims = {"N": 1024}
        else:
            dims = {"N": 1024}

    shape = ", ".join(f"{k}={v}" for k, v in dims.items())

    # Detect pattern based on primitive
    pattern = ""
    if primitive == "fused_matmul":
        pattern = "fused_relu"
    elif primitive == "gemm":
        if re.search(r'\bshared\b|__shared__', code):
            pattern = "tiled_shared"
        else:
            pattern = "tiled_shared"
    elif primitive == "elementwise":
        if re.search(r'float4|float2|double2|int4', code):
            pattern = "vectorized"
        else:
            pattern = "vectorized"
    elif primitive == "reduction":
        if re.search(r'__shfl_down_sync|warpReduce', code):
            pattern = "wavefront_reduce"
        else:
            pattern = "wavefront_reduce"

    return {
        "primitive": primitive,
        "dtype": dtype,
        "shape": shape,
        "pattern": pattern,
        "meta": {
            "dims": dims,
            "pattern_scores": scores,
            "pattern": pattern
        },
    }

