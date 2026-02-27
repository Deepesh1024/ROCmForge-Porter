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
    Classify CUDA code by calling LLM for semantic extraction.

    Returns
    -------
    dict
        primitive          : str   — detected primitive type
        dtype              : str   — "float" (default)
        shape              : str   — shape string
        pattern            : str   — detected semantic pattern
        meta               : dict  — dims + pattern + LLM extraction result
        semantic_extraction: dict  — raw LLM JSON output
    """
    # Default fallback result
    fallback_primitive = "unknown"
    semantic_result: Dict[str, Any] = {}

    try:
        semantic_result = _call_llm(code)
        primitive = semantic_result.get("primitive", fallback_primitive)
        # Validate primitive value
        valid_primitives = {
            "gemm", "reduction", "elementwise", "fused_matmul",
            "softmax", "layernorm", "conv", "attention", "dropout", "unknown",
        }
        if primitive not in valid_primitives:
            primitive = fallback_primitive
    except Exception:
        primitive = fallback_primitive
        semantic_result = {
            "primitive": "unknown",
            "pattern": "unknown",
            "memory_bound": False,
            "shared_memory_used": False,
        }

    # Use LLM-returned pattern or fall back to default
    pattern = semantic_result.get("pattern")
    if not pattern or pattern == "string":
        pattern = _DEFAULT_PATTERN.get(primitive, "vectorized")

    # Extract dimensions from code (best-effort regex fallback)
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
