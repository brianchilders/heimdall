"""
Pipeline worker configuration loaded from environment / .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Look for .env next to this file first, then fall back to cwd (project root).
_MODULE_ENV = Path(__file__).parent / ".env"
_ENV_FILE = str(_MODULE_ENV) if _MODULE_ENV.exists() else ".env"


class Settings(BaseSettings):
    """All configuration for the pipeline worker.

    Values are read from environment variables or a .env file.
    Looks for pipeline_worker/.env first, then falls back to .env in cwd.
    See .env.example for documentation on each field.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # memory-mcp
    memory_mcp_url: str = "http://localhost:8900"
    memory_mcp_token: str = ""

    # Service
    pipeline_port: int = 8001

    # Voiceprint matching
    voiceprint_db: str = "./voiceprints.sqlite"
    voiceprint_confident_threshold: float = 0.85
    voiceprint_probable_threshold: float = 0.70

    # Whisper confidence below this → attach audio clip for large-v3 fallback
    whisper_confidence_threshold: float = 0.85

    # Ollama base URL (used by memory-mcp for LLM calls; referenced here for documentation)
    ollama_base_url: str = "http://blackmagic.lan:11434/v1"

    # HuggingFace token for pyannote model (optional; diarization skipped if absent)
    hf_token: str = ""

    # Home Assistant webhook URL for probable-match notifications (optional)
    ha_webhook_url: str = ""

    # Logging
    log_level: str = "INFO"
