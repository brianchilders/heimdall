"""
SQLite VACUUM and WAL checkpoint maintenance.

Run periodically (e.g. daily) to reclaim disk space and keep the
WAL file from growing unbounded.
"""

from __future__ import annotations

import logging

from muninn.db.connection import ConnectionPool

logger = logging.getLogger(__name__)


async def vacuum_db(pool: ConnectionPool) -> None:
    """Run VACUUM on the database to reclaim freed pages.

    VACUUM rewrites the entire database file; it cannot run inside a
    transaction and requires exclusive access.  Only call during a
    maintenance window.

    Args:
        pool: Connection pool.
    """
    async with pool.write() as conn:
        # Must issue outside an active transaction
        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await conn.execute("VACUUM")
    logger.info("VACUUM complete")


async def wal_checkpoint(pool: ConnectionPool) -> dict[str, int]:
    """Issue a WAL checkpoint and return page counts.

    Args:
        pool: Connection pool.

    Returns:
        Dict with keys ``busy``, ``log``, ``checkpointed`` from
        ``PRAGMA wal_checkpoint(PASSIVE)``.
    """
    async with pool.read() as conn:
        cursor = await conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        row = await cursor.fetchone()

    result = {
        "busy": row[0] if row else -1,
        "log": row[1] if row else -1,
        "checkpointed": row[2] if row else -1,
    }
    logger.debug("WAL checkpoint: %s", result)
    return result
