"""
ROCmForge Studio — LLM-Based Semantic CUDA Primitive Classifier

Replaces AST/regex classification with an LLM call that extracts
mathematical intent and memory patterns from CUDA code.
Falls back to 'unknown' if the LLM call fails.
"""

import json
import re
from typing import Any, Dict

from openai import OpenAI

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL


_SYSTEM_PROMPT = (
    "You are an abstract mathematician. Read this CUDA code. "
    "Strip all hardware syntax. Return a JSON object defining the "
    "mathematical intent and memory patterns. Schema: "
    '{ "primitive": "gemm|reduction|elementwise|fused_matmul|softmax|layernorm|conv|attention|dropout|unknown", '
    '"pattern": "string", '
    '"memory_bound": boolean, '
    '"shared_memory_used": boolean }. '
    "Return ONLY the JSON object, no markdown fences, no explanation."
)

# Fallback dims by primitive
_DEFAULT_DIMS: Dict[str, Dict[str, int]] = {
    "gemm":        {"M": 1024, "N": 1024, "K": 1024},
    "fused_matmul": {"M": 1024, "N": 1024, "K": 1024},
    "reduction":   {"N": 1024},
    "softmax":     {"N": 1024},
    "layernorm":   {"N": 1024},
    "conv":        {"N": 1024, "C": 64, "H": 32, "W": 32},
    "attention":   {"N": 1024, "D": 64},
    "dropout":     {"N": 1024},
    "elementwise": {"N": 1024},
    "unknown":     {"N": 1024},
}

# Default pattern by primitive (must align with template_engine.TEMPLATE_MAP keys)
_DEFAULT_PATTERN: Dict[str, str] = {
    "gemm":        "tiled_shared",
    "fused_matmul": "fused_relu",
    "reduction":   "wavefront_reduce",
    "softmax":     "fused_softmax_reduce",
    "layernorm":   "fused_layernorm",
    "conv":        "direct_conv",
    "attention":   "flash_attention",
    "dropout":     "fused_dropout",
    "elementwise": "vectorized",
    "unknown":     "unknown",
}


def _get_client() -> OpenAI:
    """Create an OpenAI-compatible client pointing at Groq."""
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )


def _extract_dims_from_code(code: str) -> Dict[str, int]:
    """Best-effort regex extraction of dimension literals from code."""
    dims: Dict[str, int] = {}
    dim_pattern = re.compile(
        r'(?:M|N|K|rows|cols|width|height|dim|size)\s*=\s*(\d+)',
        re.IGNORECASE,
    )
    for m in dim_pattern.finditer(code):
        name = m.group(0).split("=")[0].strip().upper()
        dims[name] = int(m.group(1))
    return dims


def _is_cuda_code(code: str) -> bool:
    """Check if input looks like actual CUDA/C++ code (not random text)."""
    indicators = [
        r'__global__', r'__device__', r'__host__',
        r'__shared__', r'__constant__',
        r'\bblockIdx\b', r'\bthreadIdx\b', r'\bblockDim\b', r'\bgridDim\b',
        r'\bcudaMalloc\b', r'\bcudaMemcpy\b', r'\bcudaFree\b',
        r'\bhipMalloc\b', r'\bhipMemcpy\b',
        r'\bvoid\b\s+\w+\s*\(', r'\bfloat\b', r'\bdouble\b', r'\bint\b',
        r'#include', r'\bfor\s*\(', r'\bwhile\s*\(',
        r'\*\s*\w+', r'\w+\s*\[',  # pointer or array access
        r'<<<',  # kernel launch syntax
    ]
    matches = sum(1 for pat in indicators if re.search(pat, code))
    # Need at least 2 indicators to be considered code
    return matches >= 2


def _keyword_fallback_classify(code: str) -> Dict[str, Any]:
    """
    Deterministic keyword-based fallback classifier.
    Used when the LLM is unavailable (bad API key, network error, etc.).
    Mirrors the old AST classifier logic using simple keyword matching.
    """
    # Pre-check: if input doesn't look like code at all, return unknown
    if not _is_cuda_code(code):
        return {"primitive": "unknown", "pattern": "unknown",
                "memory_bound": False, "shared_memory_used": False}

    code_lower = code.lower()

    has_shared = "__shared__" in code
    has_shfl = any(k in code for k in ["__shfl", "warp_reduce"])
    has_atomic = any(k in code for k in ["atomicAdd", "atomicMax"])
    has_cublas = any(k in code_lower for k in ["cublas", "rocblas"])
    has_matmul_name = bool(re.search(r'\b(matmul|gemm|sgemm|dgemm)\b', code_lower))
    has_2d_grid = "blockIdx.y" in code or "threadIdx.y" in code
    has_conv = any(k in code_lower for k in ["conv", "cudnn", "kernel_h", "kernel_w"])
    has_exp = any(k in code_lower for k in ["expf", "exp(", "exp "])
    has_rsqrt = any(k in code_lower for k in ["rsqrtf", "rsqrt", "sqrt"])
    has_relu = bool(re.search(r'\brelu\b|>\s*0\s*\?', code_lower))
    has_dropout = any(k in code_lower for k in ["drop", "rand", "seed"])
    has_attention = sum(1 for k in ["float* Q", "float* K", "float* V", "* Q,", "* K,", "* V,"] if k in code) >= 2

    # Count for loops
    for_count = len(re.findall(r'\bfor\s*\(', code))

    # Classification priority (most specific first)
    if has_cublas or has_matmul_name:
        return {"primitive": "gemm", "pattern": "tiled_shared",
                "memory_bound": False, "shared_memory_used": has_shared}

    if has_attention and has_exp:
        return {"primitive": "attention", "pattern": "flash_attention",
                "memory_bound": True, "shared_memory_used": has_shared}

    if has_exp and has_shfl and for_count >= 2:
        return {"primitive": "softmax", "pattern": "fused_softmax_reduce",
                "memory_bound": True, "shared_memory_used": has_shared}

    if has_rsqrt and for_count >= 2 and has_shfl:
        return {"primitive": "layernorm", "pattern": "fused_layernorm",
                "memory_bound": True, "shared_memory_used": has_shared}

    if has_conv or for_count >= 4:
        return {"primitive": "conv", "pattern": "direct_conv",
                "memory_bound": False, "shared_memory_used": has_shared}

    if has_dropout and ("rand" in code_lower or "seed" in code_lower):
        return {"primitive": "dropout", "pattern": "fused_dropout",
                "memory_bound": False, "shared_memory_used": has_shared}

    if has_shfl or has_atomic:
        pattern = "wavefront_reduce" if has_shfl else "atomic_reduce"
        return {"primitive": "reduction", "pattern": pattern,
                "memory_bound": True, "shared_memory_used": has_shared}

    # 2D grid + for loop = strong GEMM signal (row/col matrix access)
    if has_2d_grid and for_count >= 1:
        if has_relu:
            return {"primitive": "fused_matmul", "pattern": "fused_relu",
                    "memory_bound": False, "shared_memory_used": has_shared}
        return {"primitive": "gemm", "pattern": "tiled_shared" if has_shared else "tiled_shared",
                "memory_bound": False, "shared_memory_used": has_shared}

    if for_count >= 1 and (has_shared or for_count >= 2):
        if has_relu:
            return {"primitive": "fused_matmul", "pattern": "fused_relu",
                    "memory_bound": False, "shared_memory_used": has_shared}
        return {"primitive": "gemm", "pattern": "tiled_shared" if has_shared else "tiled_shared",
                "memory_bound": False, "shared_memory_used": has_shared}

    # Default: elementwise
    has_vec = bool(re.search(r'float4|float2|double2|int4', code))
    return {"primitive": "elementwise", "pattern": "vectorized" if has_vec else "scalar",
            "memory_bound": True, "shared_memory_used": has_shared}


def _call_llm(cuda_code: str) -> Dict[str, Any]:
    """
    Call LLM to extract semantic primitives from CUDA code.
    Returns parsed JSON dict or raises on failure.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": cuda_code},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content or ""
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    return json.loads(raw)


def classify(code: str) -> Dict[str, Any]:
    """
    Classify CUDA code via LLM semantic extraction.
    Falls back to deterministic keyword classifier if LLM unavailable.

    Returns
    -------
    dict
        primitive          : str   — detected primitive type
        dtype              : str   — "float" (default)
        shape              : str   — shape string
        pattern            : str   — detected semantic pattern
        meta               : dict  — dims + pattern
        semantic_extraction: dict  — raw LLM/fallback JSON output
    """
    semantic_result: Dict[str, Any] = {}
    llm_used = False

    # Try LLM first
    try:
        semantic_result = _call_llm(code)
        primitive = semantic_result.get("primitive", "unknown")
        valid_primitives = {
            "gemm", "reduction", "elementwise", "fused_matmul",
            "softmax", "layernorm", "conv", "attention", "dropout", "unknown",
        }
        if primitive not in valid_primitives:
            primitive = "unknown"
        llm_used = True
    except Exception:
        # LLM unavailable — use deterministic keyword fallback
        semantic_result = _keyword_fallback_classify(code)
        semantic_result["_classifier"] = "keyword_fallback"
        primitive = semantic_result["primitive"]

    # Use LLM-returned pattern or fall back to default
    pattern = semantic_result.get("pattern")
    if not pattern or pattern == "string":
        pattern = _DEFAULT_PATTERN.get(primitive, "vectorized")

    # Extract dimensions from code (best-effort regex)
    dims = _extract_dims_from_code(code)
    if not dims:
        dims = _DEFAULT_DIMS.get(primitive, {"N": 1024}).copy()

    shape = ", ".join(f"{k}={v}" for k, v in dims.items())

    return {
        "primitive": primitive,
        "dtype": "float",
        "shape": shape,
        "pattern": pattern,
        "meta": {
            "dims": dims,
            "pattern": pattern,
        },
        "semantic_extraction": semantic_result,
    }
