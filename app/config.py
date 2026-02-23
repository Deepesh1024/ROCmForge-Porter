"""
ROCmForge Studio — Configuration Constants
"""

import os

# Path to the templates directory (relative to backend/)
TEMPLATE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

# Path to the audit logs directory (relative to backend/)
AUDIT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "audit_logs")

# Bearer token required for all API calls
BEARER_TOKEN: str = "test-token"

# Timeout in seconds for all operations
TIMEOUT_SECONDS: int = 30

# ── Groq LLM Configuration (for explanations only, NOT code gen) ─
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
