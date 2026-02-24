"""
ROCmForge Studio — Configuration Constants (Nationals Build)
"""

import os

# ── Paths ────────────────────────────────────────────────────────
TEMPLATE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
AUDIT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audit_logs")
CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
MI300X_CONFIG_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mi300x_config.json")

# ── Auth ─────────────────────────────────────────────────────────
BEARER_TOKEN: str = os.getenv("BEARER_TOKEN", "dev-token")

# ── Timeouts ─────────────────────────────────────────────────────
TIMEOUT_SECONDS: int = 30

# ── Groq LLM (explanations only, NEVER code generation) ─────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# ── Version ──────────────────────────────────────────────────────
VERSION: str = "2.1.0-nationals-final"
