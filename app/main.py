"""
ROCmForge Studio — FastAPI Application (Nationals Build v2.0)

Endpoints:
  POST /parse             — hipify + classify + safety
  POST /generate          — template-based ROCm code gen
  POST /verify            — hardware-adaptive verification
  POST /verify_remote     — MI300X remote verification
  POST /parse_extension   — PyTorch extension parser
  POST /register_mi300x_droplet — register MI300X config
  GET  /health            — healthcheck + backend info

All endpoints:
  • Require Authorization: Bearer dev-token
  • Return structured JSON with safety_score, risk_flags,
    attribution, reasoning_trace, hardware_backend_used
  • Never crash — always return valid JSON
  • Are async with 30s timeout
"""

import asyncio
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import BEARER_TOKEN, TIMEOUT_SECONDS, VERSION
from app.models import (
    APIResponse,
    GenerateRequest,
    ParseRequest,
    ParseExtensionRequest,
    RegisterMI300XRequest,
    VerifyRequest,
)
from app import (
    audit_logger,
    hardware_detector,
    hipify_runner,
    mi300x_runner,
    primitive_classifier,
    pytorch_parser,
    responsible_ai,
    safety_engine,
    template_engine,
    verifier,
)

# ── App setup ────────────────────────────────────────────────────

app = FastAPI(
    title="ROCmForge Studio",
    description="Responsible-AI CUDA-to-ROCm porting engine — Nationals Build",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helper ──────────────────────────────────────────────────

def _check_auth(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _detect() -> str:
    """Detect the hardware backend."""
    return hardware_detector.detect_backend()


# ── POST /parse ──────────────────────────────────────────────────

@app.post("/parse", response_model=APIResponse)
async def parse_endpoint(
    body: ParseRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Run mock hipify, classify CUDA primitive, and safety-analyse."""
    _check_auth(authorization)
    backend = _detect()

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _do_parse, body.cuda_code),
            timeout=TIMEOUT_SECONDS,
        )

        primitive = result["classification"]["primitive"]
        safety = result["safety"]
        rai = responsible_ai.build_responsible_ai_bundle(
            "parse", primitive, backend, safety,
        )

        audit_id = audit_logger.log(
            "parse", {"cuda_code": body.cuda_code[:500]}, result,
            hardware_backend_used=backend,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
        )

        return APIResponse(
            status="success",
            data=result,
            audit_id=audit_id,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
            hardware_backend_used=backend,
        )
    except asyncio.TimeoutError:
        return _error_response("Operation timed out (30s)", backend)
    except Exception as exc:
        return _error_response(str(exc), backend)


# ── POST /generate ───────────────────────────────────────────────

@app.post("/generate", response_model=APIResponse)
async def generate_endpoint(
    body: GenerateRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Generate ROCm code from templates (NEVER LLM)."""
    _check_auth(authorization)
    backend = _detect()

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, _do_generate, body.primitive, body.meta
            ),
            timeout=TIMEOUT_SECONDS,
        )

        safety = result["safety"]
        template_used = result["generation"].get("template_used")
        rai = responsible_ai.build_responsible_ai_bundle(
            "generate", body.primitive, backend, safety, template_used,
        )

        audit_id = audit_logger.log(
            "generate",
            {"primitive": body.primitive, "meta": body.meta},
            {"template_used": template_used, "safety_score": safety.get("score")},
            hardware_backend_used=backend,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
        )

        return APIResponse(
            status="success",
            data=result,
            audit_id=audit_id,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
            hardware_backend_used=backend,
        )
    except asyncio.TimeoutError:
        return _error_response("Operation timed out (30s)", backend)
    except Exception as exc:
        return _error_response(str(exc), backend)


# ── POST /verify ─────────────────────────────────────────────────

@app.post("/verify", response_model=APIResponse)
async def verify_endpoint(
    body: VerifyRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Verify generated ROCm code via CPU reference + mock device output."""
    _check_auth(authorization)
    backend = _detect()

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, _do_verify, body.rocm_code, body.meta, backend
            ),
            timeout=TIMEOUT_SECONDS,
        )

        primitive = body.meta.get("primitive", "elementwise")
        safety = result["safety"]
        rai = responsible_ai.build_responsible_ai_bundle(
            "verify", primitive, backend, safety,
        )

        audit_id = audit_logger.log(
            "verify",
            {"rocm_code": body.rocm_code[:300], "meta": body.meta},
            result.get("verification", {}),
            hardware_backend_used=backend,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
        )

        return APIResponse(
            status="success",
            data=result,
            audit_id=audit_id,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
            hardware_backend_used=backend,
        )
    except asyncio.TimeoutError:
        return _error_response("Operation timed out (30s)", backend)
    except Exception as exc:
        return _error_response(str(exc), backend)


# ── POST /verify_remote ──────────────────────────────────────────

@app.post("/verify_remote", response_model=APIResponse)
async def verify_remote_endpoint(
    body: VerifyRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Verify using MI300X remote (DEV MODE: mock execution)."""
    _check_auth(authorization)

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, _do_verify, body.rocm_code, body.meta, "mi300x_remote"
            ),
            timeout=TIMEOUT_SECONDS,
        )

        primitive = body.meta.get("primitive", "elementwise")
        safety = result["safety"]
        rai = responsible_ai.build_responsible_ai_bundle(
            "verify", primitive, "mi300x_remote", safety,
        )

        audit_id = audit_logger.log(
            "verify_remote",
            {"rocm_code": body.rocm_code[:300], "meta": body.meta},
            result.get("verification", {}),
            hardware_backend_used="mi300x_remote",
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
        )

        return APIResponse(
            status="success",
            data=result,
            audit_id=audit_id,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
            hardware_backend_used="mi300x_remote",
        )
    except asyncio.TimeoutError:
        return _error_response("Operation timed out (30s)", "mi300x_remote")
    except Exception as exc:
        return _error_response(str(exc), "mi300x_remote")


# ── POST /parse_extension ────────────────────────────────────────

@app.post("/parse_extension", response_model=APIResponse)
async def parse_extension_endpoint(
    body: ParseExtensionRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Parse a PyTorch C++ extension and extract CUDA kernels."""
    _check_auth(authorization)
    backend = _detect()

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, pytorch_parser.parse_extension, body.extension_code
            ),
            timeout=TIMEOUT_SECONDS,
        )

        # If embedded CUDA found, classify it
        classification = None
        safety = {"score": 100, "details": ["No embedded kernels to analyse"], "risk_flags": []}
        if result.get("extracted_cuda_code"):
            classification = primitive_classifier.classify(result["extracted_cuda_code"])
            safety = safety_engine.analyse(result["extracted_cuda_code"])

        data = {
            "extension_analysis": result,
            "classification": classification,
            "safety": safety,
        }

        rai = responsible_ai.build_responsible_ai_bundle(
            "parse_extension",
            classification["primitive"] if classification else "unknown",
            backend, safety,
        )

        audit_id = audit_logger.log(
            "parse_extension",
            {"extension_code": body.extension_code[:500]},
            data,
            hardware_backend_used=backend,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
        )

        return APIResponse(
            status="success",
            data=data,
            audit_id=audit_id,
            safety_score=rai["safety_score"],
            risk_flags=rai["risk_flags"],
            attribution=rai["attribution"],
            reasoning_trace=rai["reasoning_trace"],
            hardware_backend_used=backend,
        )
    except asyncio.TimeoutError:
        return _error_response("Operation timed out (30s)", backend)
    except Exception as exc:
        return _error_response(str(exc), backend)


# ── POST /register_mi300x_droplet ────────────────────────────────

@app.post("/register_mi300x_droplet", response_model=APIResponse)
async def register_mi300x_endpoint(
    body: RegisterMI300XRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Register MI300X droplet configuration."""
    _check_auth(authorization)

    try:
        config = body.model_dump()
        result = mi300x_runner.register_droplet(config)

        audit_id = audit_logger.log(
            "register_mi300x",
            config,
            result,
            hardware_backend_used="mi300x_remote",
        )

        return APIResponse(
            status="success",
            data=result,
            audit_id=audit_id,
            hardware_backend_used="mi300x_remote",
        )
    except Exception as exc:
        return _error_response(str(exc), "unknown")


# ── Internal pipeline functions ──────────────────────────────────

def _do_parse(cuda_code: str) -> dict:
    hipified   = hipify_runner.run_hipify(cuda_code)
    classified = primitive_classifier.classify(cuda_code)
    safety     = safety_engine.analyse(hipified["hipified_code"])
    return {
        "hipify":         hipified,
        "classification": classified,
        "safety":         safety,
    }


def _do_generate(primitive: str, meta: dict) -> dict:
    generated = template_engine.generate(primitive, meta)
    safety    = safety_engine.analyse(generated["rocm_code"])
    return {
        "generation": generated,
        "safety":     safety,
    }


def _do_verify(rocm_code: str, meta: dict, backend: str) -> dict:
    verification = verifier.verify(meta, backend)
    safety       = safety_engine.analyse(rocm_code)
    return {
        "verification": verification,
        "safety":       safety,
    }


def _error_response(message: str, backend: str) -> APIResponse:
    """Build a safe error response — never crashes."""
    return APIResponse(
        status="error",
        data=None,
        audit_id=None,
        message=message,
        hardware_backend_used=backend,
    )


# ── Health check ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    info = hardware_detector.get_backend_info()
    return {
        "status": "ok",
        "service": "ROCmForge Studio",
        "version": VERSION,
        "hardware": info,
    }
