"""
SQLite connection pool with sqlite-vec extension and WAL mode.

Architecture
------------
SQLite supports one writer + multiple concurrent readers under WAL mode.
We use a single async write connection (protected by asyncio.Lock) and a
pool of read connections for concurrent GET requests.

sqlite-vec is loaded at every connection via the load_extension() call.
The extension must be installed: pip install sqlite-vec.

Usage::

    pool = await ConnectionPool.create(db_path)
    async with pool.write() as conn:
        await conn.execute("INSERT ...")
    async with pool.read() as conn:
        rows = await conn.execute_fetchall("SELECT ...")
    await pool.close()
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
import sqlite_vec

logger = logging.getLogger(__name__)

READ_POOL_SIZE = 3


def _apply_pragmas(conn: aiosqlite.Connection) -> None:
    """Register sqlite-vec and apply required pragmas synchronously.

    Called once per connection after open.
    """
    # sqlite-vec must be loaded before any vec0 table is accessed
    conn._conn.enable_load_extension(True)
    conn._conn.load_extension(sqlite_vec.loadable_path())
    conn._conn.enable_load_extension(False)


async def _open(db_path: str) -> aiosqlite.Connection:
    """Open an aiosqlite connection with pragmas applied."""
    conn = await aiosqlite.connect(db_path, check_same_thread=False)
    conn.row_factory = aiosqlite.Row
    _apply_pragmas(conn)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA cache_size=-64000")    # 64 MB
    await conn.execute("PRAGMA mmap_size=268435456")  # 256 MB
    await conn.execute("PRAGMA temp_store=MEMORY")
    await conn.execute("PRAGMA foreign_keys=ON")
    return conn


class ConnectionPool:
    """Async SQLite connection pool — 1 write + N read connections.

    Attributes:
        db_path: Absolute path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._write_conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._read_conns: list[aiosqlite.Connection] = []
        self._read_semaphore = asyncio.Semaphore(READ_POOL_SIZE)
        self._read_index = 0

    @classmethod
    async def create(cls, db_path: str | Path) -> "ConnectionPool":
        """Open the database, run migrations, and return a ready pool.

        Args:
            db_path: Path to the SQLite file (created if absent).

        Returns:
            Initialised ConnectionPool.
        """
        path = str(Path(db_path).expanduser().resolve())
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        pool = cls(path)
        pool._write_conn = await _open(path)
        for _ in range(READ_POOL_SIZE):
            pool._read_conns.append(await _open(path))

        await pool._run_migrations()
        logger.info("Muninn DB ready: %s", path)
        return pool

    async def close(self) -> None:
        """Close all connections."""
        if self._write_conn:
            await self._write_conn.close()
        for conn in self._read_conns:
            await conn.close()
        self._read_conns = []
        self._write_conn = None

    @asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire the write connection (exclusive lock).

        Usage::

            async with pool.write() as conn:
                await conn.execute("INSERT ...")
                await conn.commit()
        """
        async with self._write_lock:
            yield self._write_conn

    @asynccontextmanager
    async def read(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a read connection from the pool (round-robin).

        Usage::

            async with pool.read() as conn:
                rows = await conn.execute_fetchall("SELECT ...")
        """
        async with self._read_semaphore:
            conn = self._read_conns[self._read_index % READ_POOL_SIZE]
            self._read_index += 1
            yield conn

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    async def _run_migrations(self) -> None:
        """Run all SQL migration files in order."""
        migrations_dir = Path(__file__).parent / "migrations"
        sql_files = sorted(migrations_dir.glob("*.sql"))

        async with self.write() as conn:
            # Track applied migrations
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            await conn.commit()

            for sql_file in sql_files:
                row = await conn.execute(
                    "SELECT 1 FROM _migrations WHERE filename = ?", (sql_file.name,)
                )
                if await row.fetchone():
                    continue

                logger.info("Applying migration: %s", sql_file.name)
                sql = sql_file.read_text()
                await conn.executescript(sql)
                await conn.execute(
                    "INSERT INTO _migrations (filename, applied_at) VALUES (?, datetime('now'))",
                    (sql_file.name,),
                )
                await conn.commit()
                logger.info("Migration applied: %s", sql_file.name)
