"""
TTL-based memory and followup expiry.

Deletes rows whose expires_at has elapsed.  Intended to be called
from a periodic background task (e.g. every hour).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from muninn.db.connection import ConnectionPool
from muninn.store.followups import delete_expired_followups

logger = logging.getLogger(__name__)


async def expire_memories(pool: ConnectionPool) -> int:
    """Delete all memories whose expires_at timestamp has elapsed.

    Args:
        pool: Connection pool.

    Returns:
        Number of memory rows deleted.
    """
    async with pool.write() as conn:
        cursor = await conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= datetime('now')"
        )
        await conn.commit()
        count = cursor.rowcount

    if count:
        logger.info("Expired %d memories", count)
    return count


async def run_expiry(pool: ConnectionPool) -> dict[str, int]:
    """Run all expiry jobs and return a summary.

    Args:
        pool: Connection pool.

    Returns:
        Dict with keys ``memories`` and ``followups`` — rows deleted each.
    """
    memories_deleted = await expire_memories(pool)
    followups_deleted = await delete_expired_followups(pool)

    logger.info(
        "Expiry run complete: %d memories, %d followups removed",
        memories_deleted,
        followups_deleted,
    )
    return {"memories": memories_deleted, "followups": followups_deleted}
