"""
Environment variable name constants shared across services.

Using constants instead of bare strings prevents typos and makes
refactoring env var names a single-file change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------------

ENV_OLLAMA_URL = "OLLAMA_URL"
ENV_MUNINN_URL = "MUNINN_URL"
ENV_VERDANDI_URL = "VERDANDI_URL"
ENV_MIMIR_URL = "MIMIR_URL"

# ---------------------------------------------------------------------------
# Muninn
# ---------------------------------------------------------------------------

ENV_MUNINN_DB_PATH = "MUNINN_DB_PATH"
ENV_MUNINN_PORT = "MUNINN_PORT"
ENV_MUNINN_WORKERS = "MUNINN_WORKERS"
ENV_MUNINN_EMBED_MODEL = "MUNINN_EMBED_MODEL"
ENV_MUNINN_EMBED_DIM = "MUNINN_EMBED_DIM"
ENV_MUNINN_EPISODIC_TTL_DAYS = "MUNINN_EPISODIC_TTL_DAYS"
ENV_MUNINN_TIMESERIES_TTL_DAYS = "MUNINN_TIMESERIES_TTL_DAYS"
ENV_MUNINN_FOLLOWUP_TTL_HOURS = "MUNINN_FOLLOWUP_TTL_HOURS"

# ---------------------------------------------------------------------------
# Verdandi
# ---------------------------------------------------------------------------

ENV_VERDANDI_PORT = "VERDANDI_PORT"
ENV_EMBED_MODEL = "EMBED_MODEL"
ENV_VERDANDI_W_SIM = "VERDANDI_W_SIM"
ENV_VERDANDI_W_REC = "VERDANDI_W_REC"
ENV_VERDANDI_W_URG = "VERDANDI_W_URG"
ENV_VERDANDI_MIN_SCORE = "VERDANDI_MIN_SCORE"
ENV_VERDANDI_TOP_K = "VERDANDI_TOP_K"
ENV_VERDANDI_RECENCY_DAYS = "VERDANDI_RECENCY_DAYS"
ENV_VERDANDI_URGENCY_WINDOW_HOURS = "VERDANDI_URGENCY_WINDOW_HOURS"

# ---------------------------------------------------------------------------
# Mimir
# ---------------------------------------------------------------------------

ENV_MIMIR_PORT = "MIMIR_PORT"
ENV_MIMIR_LLM_MODEL = "MIMIR_LLM_MODEL"
ENV_MIMIR_LLM_MAX_TOKENS = "MIMIR_LLM_MAX_TOKENS"
ENV_MIMIR_LLM_TEMPERATURE = "MIMIR_LLM_TEMPERATURE"
ENV_RELAY_HOST = "RELAY_HOST"
ENV_RELAY_PORT = "RELAY_PORT"
ENV_AVATAR_ROOMS = "AVATAR_ROOMS"
ENV_SILENCE_COOLDOWN_SECONDS = "SILENCE_COOLDOWN_SECONDS"
ENV_GREETING_COOLDOWN_MINUTES = "GREETING_COOLDOWN_MINUTES"
