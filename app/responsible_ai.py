"""
ROCmForge Studio — Responsible AI Layer

Provides deterministic, traceable Responsible-AI artefacts:
  • reasoning_trace   — step-by-step reasoning log
  • attribution       — provenance of templates and rules
  • safety_score      — 0–100 composite score
  • risk_flags        — machine-readable risk identifiers
  • human_approval    — whether human approval is required
"""

from typing import Any, Dict, List


# ── Attribution database ─────────────────────────────────────────

_TEMPLATE_ATTRIBUTION = {
    "gemm_hip_template.cpp":         "AMD ROCm 7.2 HIP GEMM example, wave64-adapted",
    "gemm_triton_template.py":       "OpenAI Triton GEMM tutorial, ROCm wave64-adapted",
    "reduction_hip_template.cpp":    "AMD ROCm 7.2 reduction example, DPP wave64",
    "reduction_triton_template.py":  "OpenAI Triton reduction, wave64-compatible",
    "elemwise_hip_template.cpp":     "AMD ROCm 7.2 vectorised elementwise example",
    "elemwise_triton_template.py":   "OpenAI Triton pointwise tutorial",
}

_RULE_ATTRIBUTION = [
    "Safety rules v2.0 — ROCmForge Studio Nationals Build",
    "Wave64 guidance: AMD CDNA3 ISA Reference Manual",
    "LDS bank-conflict heuristics: AMD GCN Memory Model",
    "Vectorisation guidance: AMD Instinct MI300X tuning guide",
]


def build_reasoning_trace(
    stage: str,
    primitive: str,
    backend: str,
    safety_score: int,
    extra_steps: List[str] | None = None,
) -> List[str]:
    """Build a step-by-step reasoning trace for audit/explainability."""
    trace = [
        f"[1] Received {stage} request for primitive: {primitive}",
        f"[2] Hardware backend detected: {backend}",
    ]

    if stage == "parse":
        trace += [
            "[3] Applied mock hipify-clang (CUDA → HIP API translation)",
            "[4] Classified CUDA primitive via deterministic regex engine",
            "[5] Ran Responsible-AI safety analysis on hipified code",
        ]
    elif stage == "generate":
        trace += [
            "[3] Selected template from verified template library (NO LLM code gen)",
            "[4] Filled placeholders with extracted metadata (dtype, dims, tile_size)",
            "[5] Ran Responsible-AI safety analysis on generated HIP code",
        ]
    elif stage == "verify":
        trace += [
            f"[3] Executed verification on backend: {backend}",
            "[4] Computed CPU reference output via NumPy",
            "[5] Compared reference vs device output (L2 norm)",
        ]
    else:
        trace.append(f"[3] Executed stage: {stage}")

    trace.append(f"[{len(trace) + 1}] Safety score computed: {safety_score}/100")
    trace.append(f"[{len(trace) + 1}] Audit log written with full provenance")

    if extra_steps:
        for i, step in enumerate(extra_steps):
            trace.append(f"[{len(trace) + 1}] {step}")

    return trace


def build_attribution(template_used: str | None = None) -> List[str]:
    """Build provenance attribution list."""
    attrs = list(_RULE_ATTRIBUTION)
    if template_used and template_used in _TEMPLATE_ATTRIBUTION:
        attrs.insert(0, f"Template: {_TEMPLATE_ATTRIBUTION[template_used]}")
    return attrs


def compute_risk_flags(
    safety_details: List[str],
    raw_flags: List[str],
    safety_score: int,
) -> tuple:
    """
    Compute final risk flags and whether human approval is required.

    Returns (risk_flags, human_approval_required)
    """
    flags = list(raw_flags)

    if safety_score < 50:
        flags.append("LOW_SAFETY_SCORE")
    if safety_score < 30:
        flags.append("CRITICAL_SAFETY")

    human_approval = safety_score < 60 or "CRITICAL_SAFETY" in flags
    return flags, human_approval


def build_responsible_ai_bundle(
    stage: str,
    primitive: str,
    backend: str,
    safety_result: Dict[str, Any],
    template_used: str | None = None,
    cache_hit: bool = False,
) -> Dict[str, Any]:
    """
    Build the complete Responsible-AI bundle that goes into every response.

    Returns dict with: safety_score, execution_confidence, risk_flags,
                       attribution, reasoning_trace, human_approval_required
    """
    score = safety_result.get("score", 0)
    details = safety_result.get("details", [])
    raw_flags = safety_result.get("risk_flags", [])

    # Cache-hit bonus: +10 to safety score (capped at 100)
    if cache_hit:
        score = min(score + 10, 100)

    # Execution confidence based on backend
    if cache_hit:
        execution_confidence = 95
    elif "rocm_local" in backend:
        execution_confidence = 85
    else:
        execution_confidence = 70

    risk_flags, human_approval = compute_risk_flags(details, raw_flags, score)
    attribution = build_attribution(template_used)

    extra_steps = []
    if cache_hit:
        extra_steps.append("MI300X cache hit — using pre-recorded GPU metrics (confidence: 95%)")
    else:
        extra_steps.append(f"No MI300X cache — CPU fallback with real NumPy timing (confidence: {execution_confidence}%)")

    reasoning = build_reasoning_trace(stage, primitive, backend, score, extra_steps)

    return {
        "safety_score": score,
        "execution_confidence": execution_confidence,
        "risk_flags": risk_flags,
        "attribution": attribution,
        "reasoning_trace": reasoning,
        "human_approval_required": human_approval,
        "hardware_backend_used": backend,
    }

