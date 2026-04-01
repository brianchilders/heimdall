"""Verdandi configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_MODULE_ENV = Path(__file__).parent.parent / ".env"
_ENV_FILE = str(_MODULE_ENV) if _MODULE_ENV.exists() else ".env"


class VerdandiConfig(BaseSettings):
    """All configuration for the Verdandi recommender engine."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Upstream services
    ollama_url: str = "http://blackmagic.lan:11434"
    muninn_url: str = "http://blackmagic.lan:8900"

    # Embedding model — must match what Muninn stores
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # Service
    verdandi_port: int = 8901

    # Scoring weights (must sum to 1.0)
    verdandi_w_sim: float = 0.60   # semantic similarity
    verdandi_w_rec: float = 0.25   # recency decay
    verdandi_w_urg: float = 0.15   # deadline urgency

    # Recommendation defaults
    verdandi_min_score: float = 0.35
    verdandi_top_k: int = 5

    # Recency decay half-life
    verdandi_recency_days: float = 7.0

    # Urgency window — deadlines within this many hours get a boost
    verdandi_urgency_window_hours: float = 2.0

    # Logging
    log_level: str = "INFO"
