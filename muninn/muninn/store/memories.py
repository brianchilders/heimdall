"""
Memory CRUD operations against the memories table.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from muninn.db.connection import ConnectionPool
from nornir.schema import ALL_TIERS

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_at(tier: str, meta: dict, episodic_ttl_days: int, timeseries_ttl_days: int) -> Optional[str]:
    """Compute expires_at from explicit TTL or tier default."""
    if "ttl_hours" in meta:
        return (datetime.now(timezone.utc) + timedelta(hours=meta["ttl_hours"])).isoformat()
    if tier == "episodic":
        return (datetime.now(timezone.utc) + timedelta(days=episodic_ttl_days)).isoformat()
    if tier == "timeseries":
        return (datetime.now(timezone.utc) + timedelta(days=timeseries_ttl_days)).isoformat()
    return None


async def create_memory(
    pool: ConnectionPool,
    tier: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    source: str | None = None,
    episodic_ttl_days: int = 90,
    timeseries_ttl_days: int = 180,
) -> dict:
    """Insert a new memory row.

    Args:
        pool: Connection pool.
        tier: One of the TIER_* constants.
        content: Natural-language memory text.
        metadata: Optional JSON-serialisable dict (tags, deadline_utc, etc.).
        source: Optional provenance tag.
        episodic_ttl_days: Default TTL for episodic memories.
        timeseries_ttl_days: Default TTL for timeseries memories.

    Returns:
        The newly created memory row as a dict.

    Raises:
        ValueError: If tier is not valid.
    """
    if tier not in ALL_TIERS:
        raise ValueError(f"Invalid tier: {tier!r}. Must be one of {ALL_TIERS}")

    meta = metadata or {}
    memory_id = str(uuid.uuid4())
    now = _utcnow()
    expires = _expires_at(tier, meta, episodic_ttl_days, timeseries_ttl_days)

    async with pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memories (id, tier, content, metadata, created_at, updated_at, expires_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, tier, content, json.dumps(meta), now, now, expires, source),
        )
        await conn.commit()

    logger.debug("Created memory %s [%s]", memory_id, tier)
    return await get_memory(pool, memory_id)


async def get_memory(pool: ConnectionPool, memory_id: str) -> Optional[dict]:
    """Retrieve a memory by ID, or None if not found.

    Args:
        pool: Connection pool.
        memory_id: UUID string.

    Returns:
        Memory dict or None.
    """
    async with pool.read() as conn:
        cursor = await conn.execute(
            "SELECT id, tier, content, metadata, created_at, updated_at, expires_at, source "
            "FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        return None
    return _row_to_dict(row)


async def update_memory(
    pool: ConnectionPool,
    memory_id: str,
    content: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """Update content and/or metadata for an existing memory.

    Args:
        pool: Connection pool.
        memory_id: UUID of the memory to update.
        content: New content string, or None to leave unchanged.
        metadata: New metadata dict, or None to leave unchanged.

    Returns:
        Updated memory dict, or None if not found.
    """
    existing = await get_memory(pool, memory_id)
    if existing is None:
        return None

    new_content = content if content is not None else existing["content"]
    new_meta = metadata if metadata is not None else existing["metadata"]
    now = _utcnow()

    async with pool.write() as conn:
        await conn.execute(
            "UPDATE memories SET content = ?, metadata = ?, updated_at = ? WHERE id = ?",
            (new_content, json.dumps(new_meta), now, memory_id),
        )
        await conn.commit()

    return await get_memory(pool, memory_id)


async def delete_memory(pool: ConnectionPool, memory_id: str) -> bool:
    """Hard-delete a memory and its embeddings (CASCADE).

    Args:
        pool: Connection pool.
        memory_id: UUID of the memory to delete.

    Returns:
        True if deleted, False if not found.
    """
    async with pool.write() as conn:
        cursor = await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await conn.commit()
        return cursor.rowcount > 0


async def list_memories(
    pool: ConnectionPool,
    tier: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    include_expired: bool = False,
) -> list[dict]:
    """List memories with optional tier filter.

    Args:
        pool: Connection pool.
        tier: Filter by tier, or None for all tiers.
        limit: Maximum rows to return.
        offset: Pagination offset.
        include_expired: Include memories past their expires_at timestamp.

    Returns:
        List of memory dicts ordered by created_at descending.
    """
    clauses = []
    params: list[Any] = []

    if tier is not None:
        clauses.append("tier = ?")
        params.append(tier)

    if not include_expired:
        clauses.append("(expires_at IS NULL OR expires_at > datetime('now'))")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]

    async with pool.read() as conn:
        cursor = await conn.execute(
            f"SELECT id, tier, content, metadata, created_at, updated_at, expires_at, source "
            f"FROM memories {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()

    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    """Convert an aiosqlite Row to a plain dict with parsed metadata."""
    d = dict(row)
    if isinstance(d.get("metadata"), str):
        d["metadata"] = json.loads(d["metadata"])
    return d
