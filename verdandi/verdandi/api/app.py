"""
Verdandi FastAPI application.

Routes:
  POST /recommend   — embed ContextEvent, query Muninn, return scored memories
  POST /embed       — embed ContextEvent, return raw vector (debug / inspection)
  GET  /health      — liveness + upstream connectivity check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nornir.models import ContextEvent, ScoredMemory
from verdandi.config import VerdandiConfig
from verdandi.embedder import embed_context
from verdandi.memory_client import active_model
from verdandi.recommender import get_recommendations

logger = logging.getLogger(__name__)

_config = VerdandiConfig()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ContextEventRequest(BaseModel):
    """Pydantic mirror of nornir.ContextEvent for HTTP deserialization."""

    who: str
    transcript: str
    emotion: str
    location: str
    local_time: str
    speaker_confidence: float = 0.0
    doa_degrees: Optional[int] = None
    objects_visible: list[str] = []
    people_detected: list[str] = []
    activity: Optional[str] = None

    def to_event(self) -> ContextEvent:
        """Convert to the nornir ContextEvent dataclass.

        Returns:
            ContextEvent populated from this request.
        """
        return ContextEvent(
            who=self.who,
            transcript=self.transcript,
            emotion=self.emotion,
            location=self.location,
            local_time=self.local_time,
            speaker_confidence=self.speaker_confidence,
            doa_degrees=self.doa_degrees,
            objects_visible=self.objects_visible,
            people_detected=self.people_detected,
            activity=self.activity,
        )


class RecommendRequest(BaseModel):
    """Body for POST /recommend."""

    event: ContextEventRequest
    top_k: Optional[int] = None
    min_score: Optional[float] = None


class ScoredMemoryOut(BaseModel):
    """Pydantic output model for ScoredMemory."""

    id: str
    content: str
    score: float
    similarity: float
    recency: float
    urgency: float
    meta: dict


class RecommendResponse(BaseModel):
    """Response from POST /recommend."""

    recommendations: list[ScoredMemoryOut]
    count: int


class EmbedResponse(BaseModel):
    """Response from POST /embed."""

    embedding: list[float]
    model: str
    dim: int


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def _make_lifespan(config: VerdandiConfig):
    """Return a lifespan context for this config instance.

    Args:
        config: VerdandiConfig to bind.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Shared HTTP client for Ollama + Muninn calls
        app.state.client = httpx.AsyncClient()
        app.state.config = config

        # Assert embedding model matches what Muninn has stored
        try:
            muninn_model = await active_model(config, client=app.state.client)
            if muninn_model is not None:
                if muninn_model["model_name"] != config.embed_model:
                    logger.warning(
                        "Embed model mismatch — Verdandi uses %r but Muninn has %r. "
                        "Run POST /maintenance/reembed on Muninn to re-embed.",
                        config.embed_model,
                        muninn_model["model_name"],
                    )
                else:
                    logger.info(
                        "Embed model confirmed: %s (%d dims, %d memories)",
                        muninn_model["model_name"],
                        muninn_model["embed_dim"],
                        muninn_model["memory_count"],
                    )
            else:
                logger.info("Muninn has no embeddings yet — proceeding")
        except Exception as exc:
            logger.warning("Could not reach Muninn at startup: %s", exc)

        logger.info("Verdandi ready (muninn=%s, ollama=%s)", config.muninn_url, config.ollama_url)
        yield
        await app.state.client.aclose()
        logger.info("Verdandi stopped")

    return lifespan


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: VerdandiConfig | None = None) -> FastAPI:
    """Build and return the configured FastAPI application.

    Args:
        config: Optional VerdandiConfig.  Reads from env if omitted.

    Returns:
        FastAPI instance.
    """
    cfg = config or VerdandiConfig()

    app = FastAPI(
        title="Verdandi Recommender",
        version="0.1.0",
        description="Surfaces relevant memories from Muninn for each ContextEvent.",
        lifespan=_make_lifespan(cfg),
    )

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(body: RecommendRequest) -> RecommendResponse:
        """Embed a ContextEvent and return scored memory recommendations.

        Args:
            body: Event + optional parameter overrides.

        Returns:
            RecommendResponse with ranked ScoredMemory list.

        Raises:
            HTTPException 502: If Ollama or Muninn is unreachable.
        """
        event = body.event.to_event()
        try:
            memories = await get_recommendations(
                event,
                cfg,
                top_k=body.top_k,
                min_score=body.min_score,
                client=app.state.client,
            )
        except httpx.HTTPError as exc:
            logger.error("Upstream error in /recommend: %s", exc)
            raise HTTPException(502, detail=f"Upstream error: {exc}") from exc

        out = [
            ScoredMemoryOut(
                id=m.id,
                content=m.content,
                score=m.score,
                similarity=m.similarity,
                recency=m.recency,
                urgency=m.urgency,
                meta=m.meta,
            )
            for m in memories
        ]
        return RecommendResponse(recommendations=out, count=len(out))

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(body: ContextEventRequest) -> EmbedResponse:
        """Embed a ContextEvent and return the raw vector.

        Useful for debugging and for pre-computing vectors to store
        as memory embeddings.

        Args:
            body: ContextEvent request.

        Returns:
            EmbedResponse with float vector, model name, and dimension.

        Raises:
            HTTPException 502: If Ollama is unreachable.
        """
        event = body.to_event()
        try:
            vec = await embed_context(event, cfg, client=app.state.client)
        except httpx.HTTPError as exc:
            logger.error("Ollama error in /embed: %s", exc)
            raise HTTPException(502, detail=f"Ollama error: {exc}") from exc

        return EmbedResponse(embedding=vec, model=cfg.embed_model, dim=len(vec))

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        """Liveness + upstream connectivity check.

        Returns:
            Dict with ``status``, ``muninn``, ``ollama`` fields.
        """
        muninn_ok = False
        ollama_ok = False
        muninn_detail = "unknown"
        ollama_detail = "unknown"

        try:
            resp = await app.state.client.get(
                f"{cfg.muninn_url}/health", timeout=3.0
            )
            muninn_ok = resp.status_code == 200
            muninn_detail = "connected" if muninn_ok else f"http {resp.status_code}"
        except Exception as exc:
            muninn_detail = str(exc)

        try:
            resp = await app.state.client.get(
                f"{cfg.ollama_url}/api/tags", timeout=3.0
            )
            ollama_ok = resp.status_code == 200
            ollama_detail = "connected" if ollama_ok else f"http {resp.status_code}"
        except Exception as exc:
            ollama_detail = str(exc)

        return {
            "status": "ok" if (muninn_ok and ollama_ok) else "degraded",
            "muninn": muninn_detail,
            "ollama": ollama_detail,
            "embed_model": cfg.embed_model,
        }

    return app


app = create_app()
