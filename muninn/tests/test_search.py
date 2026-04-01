"""
Unit tests for muninn.store.embeddings (store + KNN search).

These tests use a synthetic low-dimensional embedding (4-float) to
avoid any dependency on Ollama or a real embedding model.
"""

from __future__ import annotations

import pytest

from muninn.store.embeddings import (
    get_active_embed_model,
    knn_search,
    store_embedding,
)
from muninn.store.memories import create_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DIM = 768  # must match 002_vec_table.sql


def _unit(v: list[float]) -> list[float]:
    """Normalise a 768-dim vector to unit length.

    Pass a short seed list — it will be zero-padded to 768 dims.
    """
    padded = v + [0.0] * (_DIM - len(v))
    mag = sum(x * x for x in padded) ** 0.5
    if mag == 0:
        return padded
    return [x / mag for x in padded]


# ---------------------------------------------------------------------------
# store_embedding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_embedding(pool):
    mem = await create_memory(pool, tier="semantic", content="Test")
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    await store_embedding(pool, mem["id"], vec, "test-model")

    model = await get_active_embed_model(pool)
    assert model is not None
    assert model["model_name"] == "test-model"
    assert model["embed_dim"] == _DIM
    assert model["memory_count"] == 1


@pytest.mark.asyncio
async def test_store_embedding_upsert(pool):
    """Re-storing an embedding for the same (memory_id, model) should not fail."""
    mem = await create_memory(pool, tier="semantic", content="Upsert test")
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    await store_embedding(pool, mem["id"], vec, "test-model")
    # Second store should not raise
    vec2 = _unit([0.5, 0.5, 0.0, 0.0])
    await store_embedding(pool, mem["id"], vec2, "test-model")


# ---------------------------------------------------------------------------
# knn_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knn_search_returns_closest(pool):
    # Store two memories with distinct embeddings
    m1 = await create_memory(pool, tier="semantic", content="Memory A")
    m2 = await create_memory(pool, tier="semantic", content="Memory B")

    vec_a = _unit([1.0, 0.0, 0.0, 0.0])
    vec_b = _unit([0.0, 1.0, 0.0, 0.0])
    await store_embedding(pool, m1["id"], vec_a, "test-model")
    await store_embedding(pool, m2["id"], vec_b, "test-model")

    # Query near A — should rank A first
    results = await knn_search(pool, vec_a, top_k=2)
    assert len(results) == 2
    assert results[0]["memory_id"] == m1["id"]
    assert results[0]["distance"] < results[1]["distance"]


@pytest.mark.asyncio
async def test_knn_search_empty(pool):
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    results = await knn_search(pool, vec, top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# get_active_embed_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_active_embed_model_none_when_empty(pool):
    model = await get_active_embed_model(pool)
    assert model is None
