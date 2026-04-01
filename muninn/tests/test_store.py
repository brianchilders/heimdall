"""
Unit tests for muninn.store.memories and muninn.store.patterns.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from muninn.store.memories import (
    create_memory,
    delete_memory,
    get_memory,
    list_memories,
    update_memory,
)
from muninn.store.patterns import list_patterns, store_pattern


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get(pool):
    mem = await create_memory(pool, tier="semantic", content="Test memory", source="test")
    assert mem["id"]
    assert mem["tier"] == "semantic"
    assert mem["content"] == "Test memory"
    assert mem["source"] == "test"

    fetched = await get_memory(pool, mem["id"])
    assert fetched == mem


@pytest.mark.asyncio
async def test_get_missing_returns_none(pool):
    result = await get_memory(pool, "00000000-0000-0000-0000-000000000000")
    assert result is None


@pytest.mark.asyncio
async def test_invalid_tier_raises(pool):
    with pytest.raises(ValueError, match="Invalid tier"):
        await create_memory(pool, tier="nonexistent", content="bad")


# ---------------------------------------------------------------------------
# metadata roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_roundtrip(pool):
    meta = {"who": "Brian", "tags": ["morning", "routine"], "ttl_hours": 48}
    mem = await create_memory(pool, tier="episodic", content="Brian left at 7:45", metadata=meta)
    fetched = await get_memory(pool, mem["id"])
    assert fetched["metadata"]["who"] == "Brian"
    assert fetched["metadata"]["tags"] == ["morning", "routine"]


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_content(pool):
    mem = await create_memory(pool, tier="semantic", content="Old content")
    updated = await update_memory(pool, mem["id"], content="New content")
    assert updated["content"] == "New content"
    assert updated["metadata"] == mem["metadata"]  # unchanged


@pytest.mark.asyncio
async def test_update_metadata(pool):
    mem = await create_memory(pool, tier="semantic", content="Some fact", metadata={"x": 1})
    updated = await update_memory(pool, mem["id"], metadata={"x": 2, "y": 3})
    assert updated["metadata"]["x"] == 2
    assert updated["metadata"]["y"] == 3
    assert updated["content"] == "Some fact"  # unchanged


@pytest.mark.asyncio
async def test_update_missing_returns_none(pool):
    result = await update_memory(pool, "00000000-0000-0000-0000-000000000000", content="x")
    assert result is None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete(pool):
    mem = await create_memory(pool, tier="semantic", content="Ephemeral")
    deleted = await delete_memory(pool, mem["id"])
    assert deleted is True
    assert await get_memory(pool, mem["id"]) is None


@pytest.mark.asyncio
async def test_delete_missing_returns_false(pool):
    result = await delete_memory(pool, "00000000-0000-0000-0000-000000000000")
    assert result is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_all(seeded_pool):
    mems = await list_memories(seeded_pool)
    assert len(mems) == 5


@pytest.mark.asyncio
async def test_list_by_tier(seeded_pool):
    semantics = await list_memories(seeded_pool, tier="semantic")
    assert all(m["tier"] == "semantic" for m in semantics)
    assert len(semantics) == 2


@pytest.mark.asyncio
async def test_list_limit(seeded_pool):
    mems = await list_memories(seeded_pool, limit=2)
    assert len(mems) == 2


@pytest.mark.asyncio
async def test_list_offset(seeded_pool):
    all_mems = await list_memories(seeded_pool, limit=100)
    page2 = await list_memories(seeded_pool, limit=100, offset=2)
    assert len(page2) == len(all_mems) - 2


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_and_list_patterns(pool):
    await store_pattern(pool, "Brian leaves at 7:45am on weekdays", who="Brian")
    await store_pattern(pool, "Aria reads before bed", who="Aria")

    all_patterns = await list_patterns(pool)
    assert len(all_patterns) == 2

    brian = await list_patterns(pool, who="Brian")
    assert len(brian) == 1
    assert "7:45" in brian[0]["content"]


@pytest.mark.asyncio
async def test_list_patterns_empty(pool):
    result = await list_patterns(pool)
    assert result == []
