"""
Shared pytest fixtures for Muninn tests.

The in-memory SQLite pool uses a temporary file (not `:memory:`) so
that all three read connections see the same data via WAL mode.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from muninn.config import MuninnConfig
from muninn.db.connection import ConnectionPool
from muninn.store.memories import create_memory

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest_asyncio.fixture
async def pool(tmp_path):
    """Temporary ConnectionPool backed by a fresh SQLite file."""
    db_path = tmp_path / "test_muninn.db"
    p = await ConnectionPool.create(str(db_path))
    yield p
    await p.close()


@pytest.fixture
def config():
    """MuninnConfig with test defaults."""
    return MuninnConfig(
        muninn_db_path=":memory:",
        muninn_embed_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
        muninn_episodic_ttl_days=90,
        muninn_timeseries_ttl_days=180,
        muninn_followup_ttl_hours=4,
    )


@pytest_asyncio.fixture
async def seeded_pool(pool):
    """Pool pre-populated with the seed_memories.json fixtures."""
    seed_file = FIXTURES_DIR / "seed_memories.json"
    seeds = json.loads(seed_file.read_text())
    for s in seeds:
        await create_memory(
            pool=pool,
            tier=s["tier"],
            content=s["content"],
            metadata=s.get("metadata"),
            source=s.get("source"),
        )
    return pool
