"""
ROCmForge Studio — Hardware Detection

Detects the available execution backend:
  • "rocm_local"    — hipcc found AND rocminfo shows a GPU
  • "mi300x_remote" — mi300x_config.json exists with valid data
  • "cpu_mock"      — fallback (always works)
"""

import json
import os
import shutil
import subprocess
from typing import Dict, Any

from app.config import MI300X_CONFIG_PATH


def _check_hipcc() -> bool:
    """Return True if hipcc is on PATH and responds."""
    if not shutil.which("hipcc"):
        return False
    try:
        result = subprocess.run(
            ["hipcc", "--version"],
            capture_output=True, timeout=5, text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_rocminfo() -> bool:
    """Return True if rocminfo lists at least one agent with 'gfx'."""
    if not shutil.which("rocminfo"):
        return False
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True, timeout=10, text=True,
        )
        return "gfx" in result.stdout.lower()
    except Exception:
        return False


def _check_mi300x_config() -> bool:
    """Return True if mi300x_config.json exists and contains valid JSON."""
    if not os.path.isfile(MI300X_CONFIG_PATH):
        return False
    try:
        with open(MI300X_CONFIG_PATH) as f:
            data = json.load(f)
        return isinstance(data, dict) and data.get("size", "") != ""
    except Exception:
        return False


def detect_backend() -> str:
    """
    Detect the best available execution backend.

    Returns
    -------
    str — "rocm_local" | "mi300x_remote" | "cpu_mock"
    """
    if _check_hipcc() and _check_rocminfo():
        return "rocm_local"
    if _check_mi300x_config():
        return "mi300x_remote"
    return "cpu_mock"


def get_backend_info() -> Dict[str, Any]:
    """Return detailed info about the detected backend."""
    backend = detect_backend()
    info: Dict[str, Any] = {
        "backend": backend,
        "hipcc_available": _check_hipcc(),
        "rocminfo_gpu": _check_rocminfo(),
        "mi300x_config_exists": _check_mi300x_config(),
    }
    if backend == "mi300x_remote":
        try:
            with open(MI300X_CONFIG_PATH) as f:
                info["mi300x_config"] = json.load(f)
        except Exception:
            pass
    return info
