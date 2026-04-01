"""
Semantic search routes.

Prefix: /search

Two endpoints:
  GET  /search?q=...    — text query, embeds via Ollama internally
  POST /search          — pre-computed vector (Verdandi fast path, avoids double-embedding)
  GET  /embed-model/active — active embedding model metadata
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from muninn.store.embeddings import get_active_embed_model, knn_search
from muninn.store.memories import get_memory

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


class SearchResult(BaseModel):
    """Single result from a KNN search."""

    memory_id: str
    distance: float
    tier: str
    content: str
    metadata: dict
    created_at: str


class VectorSearchRequest(BaseModel):
    """Body for POST /search — pre-computed embedding from Verdandi."""

    embedding: list[float]
    top_k: int = 20
    tier: Optional[str] = None


# ---------------------------------------------------------------------------
# Embed-model endpoint (used by Verdandi at startup to assert model match)
# ---------------------------------------------------------------------------


@router.get("/embed-model/active", tags=["ops"])
async def active_embed_model(request: Request) -> dict:
    """Return the most recently used embedding model.

    Verdandi calls this at startup to verify the embedding model matches
    what is stored in Muninn.  Returns HTTP 404 if no embeddings exist yet.

    Returns:
        Dict with model_name, embed_dim, memory_count.

    Raises:
        HTTPException 404: If no embeddings have been stored yet.
    """
    model = await get_active_embed_model(request.app.state.pool)
    if model is None:
        raise HTTPException(404, detail="No embeddings stored yet")
    return model


# ---------------------------------------------------------------------------
# Vector search (Verdandi fast path)
# ---------------------------------------------------------------------------


@router.post("/search", tags=["search"])
async def vector_search(body: VectorSearchRequest, request: Request) -> list[SearchResult]:
    """KNN search using a pre-computed embedding vector.

    Called by Verdandi after it has already embedded the ContextEvent via
    Ollama.  Avoids re-embedding on the Muninn side.

    Args:
        body: Pre-computed embedding + search parameters.
        request: FastAPI request.

    Returns:
        List of SearchResult ordered by distance ascending.
    """
    pool = request.app.state.pool
    hits = await knn_search(pool, body.embedding, top_k=body.top_k * 3)

    results: list[SearchResult] = []
    for hit in hits:
        mem = await get_memory(pool, hit["memory_id"])
        if mem is None:
            continue
        if body.tier and mem.get("tier") != body.tier:
            continue
        results.append(
            SearchResult(
                memory_id=mem["id"],
                distance=hit["distance"],
                tier=mem["tier"],
                content=mem["content"],
                metadata=mem.get("metadata") or {},
                created_at=mem.get("created_at", ""),
            )
        )
        if len(results) >= body.top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Text search (embeds via Ollama)
# ---------------------------------------------------------------------------


@router.get("/search", tags=["search"])
async def text_search(
    request: Request,
    q: str = Query(..., description="Natural-language search query"),
    top_k: int = Query(10, ge=1, le=100),
    tier: Optional[str] = Query(None),
) -> list[SearchResult]:
    """Semantic KNN search — embeds query text via Ollama.

    Args:
        request: FastAPI request.
        q: Query string.
        top_k: Number of results to return.
        tier: Optional tier filter applied after vector search.

    Returns:
        List of SearchResult ordered by distance ascending.

    Raises:
        HTTPException 502: If Ollama is unreachable.
    """
    pool = request.app.state.pool
    config = request.app.state.config

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{config.ollama_url}/api/embeddings",
                json={"model": config.muninn_embed_model, "prompt": q},
                timeout=30.0,
            )
            resp.raise_for_status()
            query_vec = resp.json()["embedding"]
    except httpx.HTTPError as exc:
        logger.error("Ollama embedding error: %s", exc)
        raise HTTPException(502, detail=f"Embedding service unavailable: {exc}") from exc

    hits = await knn_search(pool, query_vec, top_k=top_k * 3)

    results: list[SearchResult] = []
    for hit in hits:
        mem = await get_memory(pool, hit["memory_id"])
        if mem is None:
            continue
        if tier and mem.get("tier") != tier:
            continue
        results.append(
            SearchResult(
                memory_id=mem["id"],
                distance=hit["distance"],
                tier=mem["tier"],
                content=mem["content"],
                metadata=mem.get("metadata") or {},
                created_at=mem.get("created_at", ""),
            )
        )
        if len(results) >= top_k:
            break

    return results
