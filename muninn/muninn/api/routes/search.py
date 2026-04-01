"""
Semantic search route.

Prefix: /search
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from muninn.store.embeddings import knn_search
from muninn.store.memories import get_memory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


class SearchResult(BaseModel):
    """Single result from a KNN search."""

    memory_id: str
    distance: float
    tier: str
    content: str
    metadata: dict


@router.get("")
async def semantic_search(
    request: Request,
    q: str = Query(..., description="Natural-language search query"),
    top_k: int = Query(10, ge=1, le=100),
    tier: Optional[str] = Query(None),
) -> list[SearchResult]:
    """Semantic KNN search over embedded memories.

    Embeds the query via Ollama and returns the nearest memories by
    cosine distance.

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
            )
        )
        if len(results) >= top_k:
            break

    return results
