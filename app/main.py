"""
ROCmForge Studio — FastAPI Application

Endpoints
---------
POST /parse    — mock hipify + classification
POST /generate — deterministic template-based ROCm code generation
POST /verify   — CPU-reference verification with mock ROCm output

All endpoints require ``Authorization: Bearer test-token``.
"""

import asyncio
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import BEARER_TOKEN, TIMEOUT_SECONDS
from app.models import (
    APIResponse,
    GenerateRequest,
    ParseRequest,
    VerifyRequest,
)
from app import audit_logger, hipify_runner, llm_explainer, primitive_classifier, safety_engine, template_engine, verifier

# ── App setup ────────────────────────────────────────────────────

app = FastAPI(
    title="ROCmForge Studio",
    description="Responsible-AI CUDA-to-ROCm porting engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helper ──────────────────────────────────────────────────

def _check_auth(authorization: Optional[str]) -> None:
    """Raise 401 if the bearer token is missing or wrong."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# ── Endpoints ────────────────────────────────────────────────────

@app.post("/parse", response_model=APIResponse)
async def parse_endpoint(
    body: ParseRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Run mock hipify and classify the CUDA primitive."""
    _check_auth(authorization)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _do_parse, body.cuda_code),
            timeout=TIMEOUT_SECONDS,
        )
        # LLM explanation (non-blocking, best-effort)
        explanation = await llm_explainer.explain_parse(result)
        result["explanation"] = explanation

        audit_id = audit_logger.log("parse", {"cuda_code": body.cuda_code}, result)
        return APIResponse(status="success", data=result, audit_id=audit_id)
    except asyncio.TimeoutError:
        return APIResponse(status="error", data=None, audit_id=None, message="Operation timed out (30s)")
    except Exception as exc:
        return APIResponse(status="error", data=None, audit_id=None, message=str(exc))


@app.post("/generate", response_model=APIResponse)
async def generate_endpoint(
    body: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Generate ROCm code from templates."""
    _check_auth(authorization)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _do_generate, body.primitive, body.meta),
            timeout=TIMEOUT_SECONDS,
        )
        # LLM explanation (non-blocking, best-effort)
        explanation = await llm_explainer.explain_generate(result, body.primitive, body.meta)
        result["explanation"] = explanation

        audit_id = audit_logger.log("generate", {"primitive": body.primitive, "meta": body.meta}, result)
        return APIResponse(status="success", data=result, audit_id=audit_id)
    except asyncio.TimeoutError:
        return APIResponse(status="error", data=None, audit_id=None, message="Operation timed out (30s)")
    except Exception as exc:
        return APIResponse(status="error", data=None, audit_id=None, message=str(exc))


@app.post("/verify", response_model=APIResponse)
async def verify_endpoint(
    body: VerifyRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Verify generated ROCm code via CPU reference + mock ROCm output."""
    _check_auth(authorization)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _do_verify, body.rocm_code, body.meta),
            timeout=TIMEOUT_SECONDS,
        )
        # LLM explanation (non-blocking, best-effort)
        explanation = await llm_explainer.explain_verify(result, body.meta)
        result["explanation"] = explanation

        audit_id = audit_logger.log("verify", {"rocm_code": body.rocm_code, "meta": body.meta}, result)
        return APIResponse(status="success", data=result, audit_id=audit_id)
    except asyncio.TimeoutError:
        return APIResponse(status="error", data=None, audit_id=None, message="Operation timed out (30s)")
    except Exception as exc:
        return APIResponse(status="error", data=None, audit_id=None, message=str(exc))


# ── Internal pipeline functions ──────────────────────────────────

def _do_parse(cuda_code: str) -> dict:
    """Hipify → classify → safety check pipeline."""
    hipified    = hipify_runner.run_mock_hipify(cuda_code)
    classified  = primitive_classifier.classify(cuda_code)
    safety      = safety_engine.analyse(hipified["hipified_code"])
    return {
        "hipify":         hipified,
        "classification": classified,
        "safety":         safety,
    }


def _do_generate(primitive: str, meta: dict) -> dict:
    """Template generation → safety check pipeline."""
    generated = template_engine.generate(primitive, meta)
    safety    = safety_engine.analyse(generated["rocm_code"])
    return {
        "generation": generated,
        "safety":     safety,
    }


def _do_verify(rocm_code: str, meta: dict) -> dict:
    """Verification + safety check pipeline."""
    verification = verifier.verify(meta)
    safety       = safety_engine.analyse(rocm_code)
    return {
        "verification": verification,
        "safety":       safety,
    }


# ── Health check ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ROCmForge Studio", "version": "1.0.0"}
