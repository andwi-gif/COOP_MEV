"""OpenRouter planner configuration. Credentials are environment-only and never stored."""
from __future__ import annotations
import os

OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")
OPENROUTER_MAX_COMPLETION_TOKENS = int(os.environ.get("OPENROUTER_MAX_COMPLETION_TOKENS", "512"))
OPENROUTER_REASONING_EFFORT = os.environ.get("OPENROUTER_REASONING_EFFORT", "low").strip().lower()
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
# Optional attribution headers recommended by OpenRouter. They are not credentials.
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "CoopMEV-Lite").strip()


def get_api_key() -> str:
    key = os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Create your own OpenRouter API key and export it in the shell; "
            "do not put credentials in source files, plans, logs, or Git."
        )
    return key
