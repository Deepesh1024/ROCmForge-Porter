"""
ROCmForge Studio — Pydantic Models (Nationals Build)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────

class ParseRequest(BaseModel):
    cuda_code: str = Field(..., description="Raw CUDA source code to parse and classify")


class GenerateRequest(BaseModel):
    primitive: str = Field(..., description="Primitive type: gemm | reduction | elementwise")
    meta: Dict[str, Any] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    rocm_code: str = Field(..., description="Generated ROCm / HIP code to verify")
    meta: Dict[str, Any] = Field(default_factory=dict)


class ParseExtensionRequest(BaseModel):
    extension_code: str = Field(..., description="PyTorch C++ extension source code")


class RegisterMI300XRequest(BaseModel):
    region: str = Field(default="")
    size: str = Field(default="gpu-mi300x8-1536gb-devcloud")
    image: str = Field(default="rocm-7-1-software")
    ssh_keys: List[str] = Field(default_factory=list)
    backups: bool = Field(default=False)
    ipv6: bool = Field(default=False)
    monitoring: bool = Field(default=False)
    tags: List[str] = Field(default_factory=list)
    user_data: str = Field(default="")
    vpc_uuid: str = Field(default="")


# ── Response Model ───────────────────────────────────────────────

class APIResponse(BaseModel):
    status: str = Field(..., description="'success' or 'error'")
    data: Optional[Dict[str, Any]] = None
    audit_id: Optional[str] = None
    message: Optional[str] = None
    safety_score: Optional[int] = None
    execution_confidence: Optional[int] = None
    risk_flags: Optional[List[str]] = None
    attribution: Optional[List[str]] = None
    reasoning_trace: Optional[List[str]] = None
    hardware_backend_used: Optional[str] = None
