"""
ROCmForge Studio — Pydantic Request / Response Models
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────

class ParseRequest(BaseModel):
    """Input for /parse — raw CUDA source code."""
    cuda_code: str = Field(..., description="Raw CUDA source code to parse and classify")


class GenerateRequest(BaseModel):
    """Input for /generate — primitive type and metadata."""
    primitive: str = Field(..., description="Primitive type: gemm | reduction | elementwise")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata (dtype, dims, etc.)")


class VerifyRequest(BaseModel):
    """Input for /verify — generated ROCm code and metadata."""
    rocm_code: str = Field(..., description="Generated ROCm / HIP code to verify")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata (primitive, dtype, dims, etc.)")


# ── Response Model ───────────────────────────────────────────────

class APIResponse(BaseModel):
    """Standardised envelope returned by every endpoint."""
    status: str = Field(..., description="'success' or 'error'")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Payload (when success)")
    audit_id: Optional[str] = Field(default=None, description="UUID of the audit-log entry")
    message: Optional[str] = Field(default=None, description="Human-readable message (esp. on error)")
