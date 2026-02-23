"""
ROCmForge Studio — Audit Logger

Writes structured JSON audit logs to the audit_logs/ directory.
Each log file is named with an ISO timestamp + short UUID.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from app.config import AUDIT_DIR


def _ensure_dir() -> None:
    """Create the audit-log directory if it does not exist."""
    os.makedirs(AUDIT_DIR, exist_ok=True)


def log(stage: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
    """
    Write an audit log entry and return its unique ID.

    Parameters
    ----------
    stage   : str  — "parse", "generate", or "verify"
    inputs  : dict — the request payload
    outputs : dict — the response payload

    Returns
    -------
    str — audit_id (UUID)
    """
    _ensure_dir()

    audit_id  = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    filename  = f"{timestamp}_{audit_id}.json"
    filepath  = os.path.join(AUDIT_DIR, filename)

    entry: Dict[str, Any] = {
        "audit_id":  audit_id,
        "stage":     stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs":    _sanitise(inputs),
        "outputs":   _sanitise(outputs),
    }

    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2, default=str)

    return audit_id


def _sanitise(obj: Any) -> Any:
    """Make *obj* JSON-serialisable (truncate very long strings)."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, str) and len(obj) > 5000:
        return obj[:5000] + "… [truncated]"
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
