"""
ROCmForge Studio — MI300X Remote Runner (Nationals Final)

Handles MI300X droplet registration.
Remote execution now uses the cache-first verifier.
"""

import json
import os
from typing import Any, Dict, Optional

from app.config import MI300X_CONFIG_PATH


def register_droplet(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save MI300X droplet configuration to mi300x_config.json.

    Returns status dict.
    """
    os.makedirs(os.path.dirname(MI300X_CONFIG_PATH) or ".", exist_ok=True)

    with open(MI300X_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    return {
        "registered": True,
        "config_path": MI300X_CONFIG_PATH,
        "size": config.get("size", "unknown"),
        "image": config.get("image", "unknown"),
    }


def is_registered() -> bool:
    """Check if an MI300X droplet is registered."""
    if not os.path.isfile(MI300X_CONFIG_PATH):
        return False
    try:
        with open(MI300X_CONFIG_PATH) as f:
            data = json.load(f)
        return isinstance(data, dict) and data.get("size", "") != ""
    except Exception:
        return False


def get_config() -> Optional[Dict[str, Any]]:
    """Read the registered MI300X config, or None."""
    if not is_registered():
        return None
    with open(MI300X_CONFIG_PATH) as f:
        return json.load(f)
