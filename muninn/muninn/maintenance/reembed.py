"""
Background re-embedding job.

When the active embedding model changes (MUNINN_EMBED_MODEL env var),
existing memories need new vectors.  This module:

1. Queries all memories that have no embedding for the current model.
2. Calls Ollama to compute embeddings in small batches.
3. Stores the new vectors via store_embedding().

Run manually or as a one-shot async task after changing the model.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool
from muninn.store.embeddings import get_active_embed_model, store_embedding
from muninn.store.memories import list_memories

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32


async def _embed_text(client: httpx.AsyncClient, text: str, model: str, ollama_url: str) -> list[float]:
    """Call Ollama /api/embeddings and return the embedding vector.

    Args:
        client: Shared async HTTP client.
        text: Text to embed.
        model: Ollama model name.
        ollama_url: Base URL for Ollama.

    Returns:
        Float list (dimension depends on model).

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    resp = await client.post(
        f"{ollama_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def reembed_all(
    pool: ConnectionPool,
    config: MuninnConfig,
    force: bool = False,
) -> dict[str, int]:
    """Embed any memories that lack a vector for the current model.

    Args:
        pool: Connection pool.
        config: Muninn config (supplies embed model + Ollama URL).
        force: If True, re-embed even memories that already have a vector
               for this model (useful after a model update).

    Returns:
        Dict with keys ``total``, ``embedded``, ``failed``.
    """
    target_model = config.muninn_embed_model

    # Fetch all memories (no limit — this is a maintenance job)
    memories = await list_memories(pool, limit=100_000)
    logger.info("reembed_all: %d memories total, model=%s", len(memories), target_model)

    if not memories:
        return {"total": 0, "embedded": 0, "failed": 0}

    if not force:
        # Filter to only those without an embedding for the current model
        async with pool.read() as conn:
            cursor = await conn.execute(
                "SELECT memory_id FROM memory_embeddings WHERE embed_model = ?",
                (target_model,),
            )
            rows = await cursor.fetchall()
        already_embedded = {r["memory_id"] for r in rows}
        memories = [m for m in memories if m["id"] not in already_embedded]
        logger.info(
            "reembed_all: %d memories need embedding for model=%s",
            len(memories),
            target_model,
        )

    embedded = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i in range(0, len(memories), _BATCH_SIZE):
            batch = memories[i : i + _BATCH_SIZE]
            tasks = [
                _embed_text(client, m["content"], target_model, config.ollama_url)
                for m in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for memory, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to embed memory %s: %s", memory["id"], result
                    )
                    failed += 1
                    continue
                try:
                    await store_embedding(pool, memory["id"], result, target_model)
                    embedded += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to store embedding for %s: %s", memory["id"], exc
                    )
                    failed += 1

    logger.info(
        "reembed_all done: %d embedded, %d failed (model=%s)",
        embedded,
        failed,
        target_model,
    )
    return {"total": len(memories), "embedded": embedded, "failed": failed}
