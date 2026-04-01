"""Muninn configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_MODULE_ENV = Path(__file__).parent.parent / ".env"
_ENV_FILE = str(_MODULE_ENV) if _MODULE_ENV.exists() else ".env"


class MuninnConfig(BaseSettings):
    """All configuration for the Muninn memory server."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    muninn_db_path: str = "./muninn/data/muninn.db"

    # Service
    muninn_port: int = 8900
    muninn_workers: int = 2

    # Embedding model — must match what Verdandi uses
    muninn_embed_model: str = "nomic-embed-text"
    muninn_embed_dim: int = 768

    # Ollama for embedding (Muninn embeds on write via MCP tool path)
    ollama_url: str = "http://blackmagic.lan:11434"

    # TTL defaults
    muninn_episodic_ttl_days: int = 90
    muninn_timeseries_ttl_days: int = 180
    muninn_followup_ttl_hours: int = 4

    # Logging
    log_level: str = "INFO"
