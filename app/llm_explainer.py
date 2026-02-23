"""
ROCmForge Studio — LLM Explainer (Groq)

Uses ChatGroq (OpenAI-compatible) for generating human-readable
explanations about the CUDA-to-ROCm porting process.

IMPORTANT: The LLM is used ONLY for explanations and reasoning logs,
NEVER for generating kernel code.  All code comes from templates.
"""

import asyncio
from typing import Any, Dict, Optional

from openai import OpenAI

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL


def _get_client() -> OpenAI:
    """Create an OpenAI-compatible client pointing at Groq."""
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )


# ── Prompt templates (LLM ONLY for explanations, NOT code gen) ──

_PARSE_EXPLANATION_PROMPT = """You are ROCmForge Studio, a CUDA-to-ROCm porting assistant.

A user submitted CUDA code that was:
1. Hipified (CUDA APIs replaced with HIP equivalents)
2. Classified as primitive type: {primitive}
3. Safety-analyzed with score: {safety_score}/100

Changes made during hipification:
{changes}

Safety findings:
{safety_details}

Risk flags: {risk_flags}

Write a concise, technical explanation (3-5 sentences) of:
- What the original CUDA code does
- What changes were made during hipification
- Key safety observations for ROCm/AMD GPU execution
- Any recommendations for the developer

Be precise and technical. Do NOT generate any code."""

_GENERATE_EXPLANATION_PROMPT = """You are ROCmForge Studio, a CUDA-to-ROCm porting assistant.

A HIP kernel was generated from template: {template_used}
Primitive type: {primitive}
Data type: {dtype}
Dimensions: {dims}
Safety score: {safety_score}/100

Safety findings:
{safety_details}

Write a concise, technical explanation (3-5 sentences) of:
- What the generated HIP kernel does
- Why this template was chosen
- Key safety/performance notes for AMD GPUs (wave64, vectorization, etc.)
- Any tuning recommendations

Be precise and technical. Do NOT generate any code."""

_VERIFY_EXPLANATION_PROMPT = """You are ROCmForge Studio, a CUDA-to-ROCm porting assistant.

Verification results for a {primitive} kernel:
- L2 norm (ref vs mock ROCm): {l2_norm}
- Pass: {passed}
- Estimated speed: {speed_ms} ms
- Occupancy: {occupancy}%
- Bandwidth: {bandwidth_gbps} GB/s
- Safety score: {safety_score}/100

Write a concise, technical explanation (2-4 sentences) of:
- Whether the verification passed and what the L2 norm means
- Performance placeholder interpretation
- Any concerns or next steps

Be precise and technical. Do NOT generate any code."""


async def explain_parse(parse_result: Dict[str, Any]) -> str:
    """Generate an LLM explanation for a /parse result."""
    classification = parse_result.get("classification", {})
    safety = parse_result.get("safety", {})
    hipify = parse_result.get("hipify", {})

    prompt = _PARSE_EXPLANATION_PROMPT.format(
        primitive=classification.get("primitive", "unknown"),
        safety_score=safety.get("score", "N/A"),
        changes="\n".join(f"  - {c}" for c in hipify.get("changes", [])),
        safety_details="\n".join(f"  - {d}" for d in safety.get("details", [])),
        risk_flags=", ".join(safety.get("risk_flags", [])) or "None",
    )

    return await _call_llm(prompt)


async def explain_generate(generate_result: Dict[str, Any], primitive: str, meta: Dict[str, Any]) -> str:
    """Generate an LLM explanation for a /generate result."""
    generation = generate_result.get("generation", {})
    safety = generate_result.get("safety", {})

    prompt = _GENERATE_EXPLANATION_PROMPT.format(
        template_used=generation.get("template_used", "unknown"),
        primitive=primitive,
        dtype=meta.get("dtype", "float"),
        dims=meta.get("dims", {}),
        safety_score=safety.get("score", "N/A"),
        safety_details="\n".join(f"  - {d}" for d in safety.get("details", [])),
    )

    return await _call_llm(prompt)


async def explain_verify(verify_result: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """Generate an LLM explanation for a /verify result."""
    verification = verify_result.get("verification", {})
    safety = verify_result.get("safety", {})

    prompt = _VERIFY_EXPLANATION_PROMPT.format(
        primitive=meta.get("primitive", "unknown"),
        l2_norm=verification.get("l2_norm", "N/A"),
        passed=verification.get("pass", "N/A"),
        speed_ms=verification.get("speed_ms", "N/A"),
        occupancy=verification.get("occupancy", "N/A"),
        bandwidth_gbps=verification.get("bandwidth_gbps", "N/A"),
        safety_score=safety.get("score", "N/A"),
    )

    return await _call_llm(prompt)


async def _call_llm(prompt: str) -> str:
    """Call the Groq LLM and return the response text."""
    try:
        client = _get_client()

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are ROCmForge Studio, a technical CUDA-to-ROCm porting assistant. Be concise and precise. Never generate code."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            ),
        )

        return response.choices[0].message.content or "No explanation generated."

    except Exception as exc:
        return f"[LLM explanation unavailable: {exc}]"
