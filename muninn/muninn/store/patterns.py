"""
Pattern-tier memory helpers.

Pattern memories are LLM-generated behavioural summaries — e.g.
"Brian typically leaves for school dropoff at 7:45am on weekdays."
They are written by a scheduled job that analyses episodic and timeseries
memories, not by the real-time pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from muninn.db.connection import ConnectionPool
from muninn.store.memories import create_memory, list_memories
from nornir.schema import TIER_PATTERN

logger = logging.getLogger(__name__)


async def store_pattern(
    pool: ConnectionPool,
    content: str,
    who: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source: str = "pattern_job",
) -> dict:
    """Store a pattern-tier memory.

    Args:
        pool: Connection pool.
        content: Natural-language pattern description.
        who: Entity this pattern is about.
        tags: Optional tags for filtering.
        source: Provenance tag.

    Returns:
        The created memory dict.
    """
    meta: dict = {}
    if who:
        meta["who"] = who
    if tags:
        meta["tags"] = tags

    return await create_memory(
        pool=pool,
        tier=TIER_PATTERN,
        content=content,
        metadata=meta,
        source=source,
    )


async def list_patterns(
    pool: ConnectionPool,
    who: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """List pattern-tier memories, optionally filtered by entity.

    Args:
        pool: Connection pool.
        who: Filter by entity name, or None for all patterns.
        limit: Maximum rows.

    Returns:
        List of pattern memory dicts.
    """
    all_patterns = await list_memories(pool, tier=TIER_PATTERN, limit=limit * 2)

    if who is None:
        return all_patterns[:limit]

    return [
        p for p in all_patterns
        if p.get("metadata", {}).get("who") == who
    ][:limit]
