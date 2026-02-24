"""
ROCmForge Studio — Audit Logger (Nationals Build)

Extended audit log with:
  stage, request, response, hardware_backend_used,
  safety_score, risk_flags, attribution, reasoning_trace, timestamp
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import AUDIT_DIR


def log(
    stage: str,
    request_data: Dict[str, Any],
    response_data: Dict[str, Any],
    *,
    hardware_backend_used: str = "cpu_mock",
    safety_score: Optional[int] = None,
    risk_flags: Optional[List[str]] = None,
    attribution: Optional[List[str]] = None,
    reasoning_trace: Optional[List[str]] = None,
) -> str:
    """
    Write a structured audit log entry.

    Returns the audit_id (short UUID).
    """
    os.makedirs(AUDIT_DIR, exist_ok=True)

    audit_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    ts_file = now.strftime("%Y-%m-%dT%H-%M-%S")
    ts_iso = now.isoformat()

    entry = {
        "audit_id":               audit_id,
        "stage":                  stage,
        "timestamp":              ts_iso,
        "hardware_backend_used":  hardware_backend_used,
        "safety_score":           safety_score,
        "risk_flags":             risk_flags or [],
        "attribution":            attribution or [],
        "reasoning_trace":        reasoning_trace or [],
        "request":                _safe_serialize(request_data),
        "response":               _safe_serialize(response_data),
    }

    filename = f"{ts_file}_{audit_id}.json"
    filepath = os.path.join(AUDIT_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2, default=str)

    return audit_id


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-safe by converting non-serialisable types."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
