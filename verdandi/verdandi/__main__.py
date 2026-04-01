"""Entrypoint: python -m verdandi"""

from __future__ import annotations

import logging

import uvicorn

from verdandi.config import VerdandiConfig

config = VerdandiConfig()

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

uvicorn.run(
    "verdandi.api.app:app",
    host="0.0.0.0",
    port=config.verdandi_port,
    log_level=config.log_level.lower(),
)
