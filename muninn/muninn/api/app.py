"""
Muninn FastAPI application.

Mounts all route groups and manages the database connection pool
via the lifespan context.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool
from muninn.api.routes import followups, maintenance, memories, search

logger = logging.getLogger(__name__)


def _make_lifespan(config: MuninnConfig):
    """Return a lifespan context manager bound to the given config.

    Args:
        config: MuninnConfig instance to use for this app.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Muninn starting — db=%s", config.muninn_db_path)
        pool = await ConnectionPool.create(config.muninn_db_path)
        app.state.pool = pool
        app.state.config = config
        yield
        await pool.close()
        logger.info("Muninn stopped")

    return lifespan


def create_app(config: MuninnConfig | None = None) -> FastAPI:
    """Build and return the configured FastAPI application.

    Args:
        config: Optional MuninnConfig.  Defaults to a fresh MuninnConfig()
                (reads from env / .env file).

    Returns:
        FastAPI instance with all routes mounted.
    """
    cfg = config or MuninnConfig()

    app = FastAPI(
        title="Muninn Memory Server",
        version="0.1.0",
        description="Persistent memory store for the Heimdall ambient intelligence system.",
        lifespan=_make_lifespan(cfg),
    )

    app.include_router(memories.router)
    app.include_router(search.router)
    app.include_router(followups.router)
    app.include_router(maintenance.router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        """Return service health and database statistics.

        Returns:
            Dict with status, db_path, and embed_model.
        """
        return {
            "status": "ok",
            "db_path": cfg.muninn_db_path,
            "embed_model": cfg.muninn_embed_model,
        }

    return app


# Module-level app instance for uvicorn (reads from env)
app = create_app()
