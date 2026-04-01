"""
Pending followup CRUD — Mimir writes after speaking; reads on next ContextEvent.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from muninn.db.connection import ConnectionPool

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_followup(
    pool: ConnectionPool,
    who: str,
    spoken_text: str,
    location: Optional[str] = None,
    memory_id: Optional[str] = None,
    ttl_hours: int = 4,
) -> dict:
    """Store a pending followup after Mimir speaks.

    Args:
        pool: Connection pool.
        who: Speaker name this followup is for.
        spoken_text: What was said — for context when it fires again.
        location: Room where it was spoken (optional).
        memory_id: Source memory ID (optional).
        ttl_hours: Hours until this followup expires.

    Returns:
        The created followup row as a dict.
    """
    followup_id = str(uuid.uuid4())
    now = _utcnow()
    expires = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()

    async with pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO pending_followups
                (id, memory_id, who, location, spoken_text, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (followup_id, memory_id, who, location, spoken_text, now, expires),
        )
        await conn.commit()

    logger.debug("Created followup %s for %s (expires %s)", followup_id, who, expires)
    return await get_followup(pool, followup_id)


async def get_followup(pool: ConnectionPool, followup_id: str) -> Optional[dict]:
    """Retrieve a followup by ID."""
    async with pool.read() as conn:
        cursor = await conn.execute(
            "SELECT id, memory_id, who, location, spoken_text, created_at, expires_at "
            "FROM pending_followups WHERE id = ?",
            (followup_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def list_followups_for(
    pool: ConnectionPool,
    who: str,
    include_expired: bool = False,
) -> list[dict]:
    """List active pending followups for a speaker.

    Args:
        pool: Connection pool.
        who: Speaker name.
        include_expired: Include followups past their expires_at.

    Returns:
        List of followup dicts ordered by created_at descending.
    """
    if include_expired:
        clause = "WHERE who = ?"
        params = [who]
    else:
        clause = "WHERE who = ? AND expires_at > datetime('now')"
        params = [who]

    async with pool.read() as conn:
        cursor = await conn.execute(
            f"SELECT id, memory_id, who, location, spoken_text, created_at, expires_at "
            f"FROM pending_followups {clause} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def delete_followup(pool: ConnectionPool, followup_id: str) -> bool:
    """Delete a single followup by ID.

    Returns:
        True if deleted, False if not found.
    """
    async with pool.write() as conn:
        cursor = await conn.execute(
            "DELETE FROM pending_followups WHERE id = ?", (followup_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0


async def delete_expired_followups(pool: ConnectionPool) -> int:
    """Delete all followups past their expires_at timestamp.

    Returns:
        Number of rows deleted.
    """
    async with pool.write() as conn:
        cursor = await conn.execute(
            "DELETE FROM pending_followups WHERE expires_at <= datetime('now')"
        )
        await conn.commit()
        count = cursor.rowcount
    if count:
        logger.info("Deleted %d expired followups", count)
    return count
