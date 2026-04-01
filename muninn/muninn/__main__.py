"""Entrypoint: python -m muninn"""

from __future__ import annotations

import logging

import uvicorn

from muninn.config import MuninnConfig

config = MuninnConfig()

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

uvicorn.run(
    "muninn.api.app:app",
    host="0.0.0.0",
    port=config.muninn_port,
    workers=config.muninn_workers,
    log_level=config.log_level.lower(),
)
