"""Shared fixtures for memory_extensions tests.

The voice routes call ``mem.get_db()`` directly (importing ``server as mem``),
so we cannot use FastAPI dependency_overrides.  Instead we patch the
``server`` module that voice_routes imports to return our in-memory SQLite
connection.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from types import ModuleType
from typing import Generator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Minimal memory-mcp schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE,
    type    TEXT NOT NULL DEFAULT 'person',
    meta    TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL,
    updated REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    fact       TEXT NOT NULL,
    category   TEXT NOT NULL DEFAULT 'general',
    confidence REAL NOT NULL DEFAULT 1.0,
    source     TEXT,
    created    REAL NOT NULL,
    updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS readings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    metric     TEXT NOT NULL,
    unit       TEXT,
    value_type TEXT NOT NULL DEFAULT 'composite',
    value_num  REAL,
    value_cat  TEXT,
    value_json TEXT,
    source     TEXT,
    ts         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    started_at REAL NOT NULL,
    ended_at   REAL,
    summary    TEXT,
    meta       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS session_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    ts         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_a  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_b  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel_type  TEXT NOT NULL,
    meta      TEXT NOT NULL DEFAULT '{}',
    created   REAL NOT NULL,
    valid_until REAL,
    UNIQUE(entity_a, entity_b, rel_type)
);
"""


# ---------------------------------------------------------------------------
# Fake ``server`` module
# ---------------------------------------------------------------------------
# voice_routes does ``import server as mem`` and calls ``mem.get_db()``.
# We inject a fake module into sys.modules before importing voice_routes so
# it never tries to load the real memory-mcp server.py.


class _NonClosingProxy:
    """Proxy that forwards all attribute access to the wrapped SQLite connection,
    except ``close()`` which is a no-op.

    Routes call ``db.close()`` in their ``finally`` blocks.  Without this proxy
    the route would close the test connection before our assertions can run.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def close(self) -> None:  # noqa: D102
        pass  # intentional no-op — the fixture owns the connection lifecycle

    def __getattr__(self, name: str):  # noqa: D105
        return getattr(self._conn, name)


class _FakeServer(ModuleType):
    """Minimal stand-in for memory-mcp's server module."""

    _conn: sqlite3.Connection | None = None

    def get_db(self) -> _NonClosingProxy:  # noqa: D102
        if self._conn is None:
            raise RuntimeError("Test DB not initialised — ensure the db fixture ran first")
        return _NonClosingProxy(self._conn)


_fake_server = _FakeServer("server")
sys.modules.setdefault("server", _fake_server)

# Import the router *after* the fake server is in sys.modules.
from memory_extensions.voice_routes import router  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite connection with the minimal memory-mcp schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    # Wire into the fake server module so voice_routes.get_db() returns it.
    _fake_server._conn = conn
    yield conn
    _fake_server._conn = None
    conn.close()


@pytest.fixture
def app(db: sqlite3.Connection) -> FastAPI:
    """FastAPI test app with the voice router mounted."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async HTTP client connected to the test FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def insert_entity(
    db: sqlite3.Connection,
    name: str,
    entity_type: str = "person",
    meta: dict | None = None,
) -> int:
    """Insert a test entity and return its id."""
    now = time.time()
    cur = db.execute(
        "INSERT INTO entities(name, type, meta, created, updated) VALUES (?,?,?,?,?)",
        (name, entity_type, json.dumps(meta or {}), now, now),
    )
    db.commit()
    return cur.lastrowid
