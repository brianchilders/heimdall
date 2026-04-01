"""
HTTP client for the Muninn memory server.

Provides two operations used by the recommender:
  - vector_search()  — POST /search with a pre-computed embedding
  - active_model()   — GET /embed-model/active (startup model assertion)

Uses a single shared AsyncClient per app instance (injected via
app.state) to reuse the underlying TCP connection pool.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from verdandi.config import VerdandiConfig

logger = logging.getLogger(__name__)

# Shape of a single result row returned by Muninn's POST /search
MuninnHit = dict  # keys: memory_id, distance, tier, content, metadata, created_at


async def vector_search(
    embedding: list[float],
    config: VerdandiConfig,
    top_k: int = 20,
    tier: Optional[str] = None,
    client: httpx.AsyncClient | None = None,
) -> list[MuninnHit]:
    """POST a pre-computed vector to Muninn and return KNN results.

    Args:
        embedding: Query vector (must match Muninn's stored embed_dim).
        config: Verdandi config (supplies muninn_url).
        top_k: Number of candidates to request from Muninn.
        tier: Optional tier filter applied server-side.
        client: Optional existing AsyncClient.

    Returns:
        List of hit dicts, each with keys:
        ``memory_id``, ``distance``, ``tier``, ``content``,
        ``metadata``, ``created_at``.

    Raises:
        httpx.HTTPStatusError: On non-2xx from Muninn.
        httpx.TimeoutException: If Muninn does not respond within 10s.
    """
    body: dict = {"embedding": embedding, "top_k": top_k}
    if tier:
        body["tier"] = tier

    async def _call(c: httpx.AsyncClient) -> list[MuninnHit]:
        resp = await c.post(
            f"{config.muninn_url}/search",
            json=body,
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    if client is not None:
        return await _call(client)

    async with httpx.AsyncClient() as c:
        return await _call(c)


async def active_model(
    config: VerdandiConfig,
    client: httpx.AsyncClient | None = None,
) -> dict | None:
    """Fetch the active embedding model metadata from Muninn.

    Returns None if Muninn has no embeddings yet (fresh install).

    Args:
        config: Verdandi config.
        client: Optional existing AsyncClient.

    Returns:
        Dict with ``model_name``, ``embed_dim``, ``memory_count``,
        or None if 404 (no embeddings stored yet).

    Raises:
        httpx.HTTPStatusError: On non-404 error responses.
    """
    async def _call(c: httpx.AsyncClient) -> dict | None:
        resp = await c.get(
            f"{config.muninn_url}/embed-model/active",
            timeout=10.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    if client is not None:
        return await _call(client)

    async with httpx.AsyncClient() as c:
        return await _call(c)
