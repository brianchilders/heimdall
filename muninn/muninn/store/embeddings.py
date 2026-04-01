"""
Embedding store — write to memory_embeddings + vec_memories, KNN search.
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime, timezone
from typing import Optional

from muninn.db.connection import ConnectionPool

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack(embedding: list[float]) -> bytes:
    """Pack a float list to little-endian float32 bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


async def store_embedding(
    pool: ConnectionPool,
    memory_id: str,
    embedding: list[float],
    embed_model: str,
) -> None:
    """Store a precomputed embedding for a memory.

    Writes to both memory_embeddings (raw blob) and vec_memories (KNN index).
    Updates the embed_models registry row.

    Args:
        pool: Connection pool.
        memory_id: UUID of the parent memory.
        embedding: Float list from the embedding model.
        embed_model: Model name, e.g. 'nomic-embed-text'.
    """
    embed_dim = len(embedding)
    blob = _pack(embedding)
    now = _utcnow()

    async with pool.write() as conn:
        # Upsert into raw embedding store
        await conn.execute(
            """
            INSERT INTO memory_embeddings (memory_id, embedding, embed_model, embed_dim, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, embed_model) DO UPDATE SET
                embedding  = excluded.embedding,
                created_at = excluded.created_at
            """,
            (memory_id, blob, embed_model, embed_dim, now),
        )

        # Upsert into vec virtual table (sqlite-vec KNN index)
        await conn.execute(
            "INSERT OR REPLACE INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
            (memory_id, blob),
        )

        # Update embed_models registry
        await conn.execute(
            """
            INSERT INTO embed_models (model_name, embed_dim, first_used, last_used, memory_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(model_name) DO UPDATE SET
                last_used    = excluded.last_used,
                memory_count = memory_count + 1
            """,
            (embed_model, embed_dim, now, now),
        )

        await conn.commit()

    logger.debug("Stored embedding for memory %s [model=%s dim=%d]", memory_id, embed_model, embed_dim)


async def knn_search(
    pool: ConnectionPool,
    query_embedding: list[float],
    top_k: int = 20,
    embed_model: Optional[str] = None,
) -> list[dict]:
    """KNN vector search using sqlite-vec.

    Args:
        pool: Connection pool.
        query_embedding: Query vector as float list.
        top_k: Number of nearest neighbours to return.
        embed_model: Optional model filter (for multi-model scenarios).

    Returns:
        List of dicts with keys: memory_id, distance.
        Ordered by distance ascending (closest first).
    """
    blob = _pack(query_embedding)

    async with pool.read() as conn:
        cursor = await conn.execute(
            """
            SELECT memory_id, distance
            FROM vec_memories
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (blob, top_k),
        )
        rows = await cursor.fetchall()

    return [{"memory_id": row["memory_id"], "distance": row["distance"]} for row in rows]


async def get_active_embed_model(pool: ConnectionPool) -> Optional[dict]:
    """Return the most recently used embedding model, or None if no embeddings exist.

    Args:
        pool: Connection pool.

    Returns:
        Dict with model_name, embed_dim, memory_count, or None.
    """
    async with pool.read() as conn:
        cursor = await conn.execute(
            "SELECT model_name, embed_dim, memory_count FROM embed_models "
            "ORDER BY last_used DESC LIMIT 1"
        )
        row = await cursor.fetchone()

    if row is None:
        return None
    return dict(row)


async def delete_embeddings_for_model(pool: ConnectionPool, embed_model: str) -> int:
    """Delete all embeddings for a given model (used during migration).

    Args:
        pool: Connection pool.
        embed_model: Model name to purge.

    Returns:
        Number of rows deleted from memory_embeddings.
    """
    async with pool.write() as conn:
        cursor = await conn.execute(
            "DELETE FROM memory_embeddings WHERE embed_model = ?", (embed_model,)
        )
        # Also clear vec_memories (rebuild via re-embed job)
        await conn.execute("DELETE FROM vec_memories")
        await conn.execute(
            "DELETE FROM embed_models WHERE model_name = ?", (embed_model,)
        )
        await conn.commit()
        return cursor.rowcount
