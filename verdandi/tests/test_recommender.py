"""
Unit tests for verdandi.recommender.

All Ollama and Muninn calls are mocked — no live services required.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nornir.models import ContextEvent, ScoredMemory
from verdandi.config import VerdandiConfig
from verdandi.recommender import (
    _distance_to_similarity,
    _recency_score,
    _score_hit,
    _urgency_score,
    get_recommendations,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sample_events.json"

_CONFIG = VerdandiConfig(
    ollama_url="http://localhost:11434",
    muninn_url="http://localhost:8900",
    embed_model="nomic-embed-text",
    verdandi_w_sim=0.60,
    verdandi_w_rec=0.25,
    verdandi_w_urg=0.15,
    verdandi_min_score=0.35,
    verdandi_top_k=5,
    verdandi_recency_days=7.0,
    verdandi_urgency_window_hours=2.0,
)

_FAKE_VEC = [0.1] * 768


def _make_event(index: int = 0) -> ContextEvent:
    data = json.loads(FIXTURES.read_text())[index]
    return ContextEvent(
        who=data["who"],
        transcript=data["transcript"],
        emotion=data["emotion"],
        location=data["location"],
        local_time=data["local_time"],
        objects_visible=data.get("objects_visible", []),
        activity=data.get("activity"),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_ago_iso(hours: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


def _hours_from_now_iso(hours: float) -> str:
    dt = datetime.now(timezone.utc) + timedelta(hours=hours)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# _distance_to_similarity
# ---------------------------------------------------------------------------


def test_distance_zero_is_perfect_similarity():
    assert _distance_to_similarity(0.0) == 1.0


def test_distance_sqrt2_is_zero_similarity():
    # L2 distance of sqrt(2) ≈ 1.414 corresponds to cos=0 (orthogonal)
    import math
    result = _distance_to_similarity(math.sqrt(2))
    assert abs(result) < 1e-9


def test_distance_clamps_negative():
    # Distances > sqrt(2) would give negative — should clamp to 0
    assert _distance_to_similarity(2.0) == 0.0


def test_distance_half_sqrt2():
    import math
    # distance = sqrt(1) = 1.0 → similarity = 1 - 1/2 = 0.5
    result = _distance_to_similarity(1.0)
    assert abs(result - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------


def test_recency_very_recent():
    # Just created — score should be close to 1
    score = _recency_score(_now_iso(), recency_days=7.0)
    assert score > 0.99


def test_recency_3_5_days_old():
    # 3.5 days = exactly half the 7-day window → score ≈ 0.5
    score = _recency_score(_hours_ago_iso(7 * 24 / 2), recency_days=7.0)
    assert 0.48 < score < 0.52


def test_recency_expired():
    # 8 days old with 7-day window → floored at 0
    score = _recency_score(_hours_ago_iso(8 * 24), recency_days=7.0)
    assert score == 0.0


def test_recency_bad_timestamp():
    score = _recency_score("not-a-date", recency_days=7.0)
    assert score == 0.0


def test_recency_empty_string():
    score = _recency_score("", recency_days=7.0)
    assert score == 0.0


# ---------------------------------------------------------------------------
# _urgency_score
# ---------------------------------------------------------------------------


def test_urgency_no_deadline():
    assert _urgency_score({}, urgency_window_hours=2.0) == 0.0


def test_urgency_deadline_imminent():
    # 1 hour away in a 2-hour window → urgency = 1 - 1/2 = 0.5
    meta = {"deadline_utc": _hours_from_now_iso(1.0)}
    score = _urgency_score(meta, urgency_window_hours=2.0)
    assert abs(score - 0.5) < 0.01


def test_urgency_deadline_very_close():
    meta = {"deadline_utc": _hours_from_now_iso(0.1)}
    score = _urgency_score(meta, urgency_window_hours=2.0)
    assert score > 0.9


def test_urgency_deadline_past():
    meta = {"deadline_utc": _hours_from_now_iso(-1.0)}
    assert _urgency_score(meta, urgency_window_hours=2.0) == 0.0


def test_urgency_deadline_outside_window():
    meta = {"deadline_utc": _hours_from_now_iso(5.0)}
    assert _urgency_score(meta, urgency_window_hours=2.0) == 0.0


def test_urgency_bad_deadline():
    meta = {"deadline_utc": "invalid"}
    assert _urgency_score(meta, urgency_window_hours=2.0) == 0.0


# ---------------------------------------------------------------------------
# _score_hit
# ---------------------------------------------------------------------------


def test_score_hit_composite_weights():
    """Composite score must equal W_SIM*sim + W_REC*rec + W_URG*urg."""
    hit = {
        "memory_id": "abc",
        "distance": 0.0,          # similarity = 1.0
        "content": "test",
        "tier": "semantic",
        "metadata": {},
        "created_at": _now_iso(),  # recency ≈ 1.0
    }
    result = _score_hit(hit, _CONFIG)
    # No urgency (no deadline) → score ≈ W_SIM*1 + W_REC*1 + W_URG*0 = 0.85
    assert abs(result.score - (0.60 * 1.0 + 0.25 * 1.0)) < 0.02


def test_score_hit_with_urgency():
    meta = {"deadline_utc": _hours_from_now_iso(0.5)}
    hit = {
        "memory_id": "xyz",
        "distance": 0.0,
        "content": "urgent thing",
        "tier": "semantic",
        "metadata": meta,
        "created_at": _now_iso(),
    }
    result = _score_hit(hit, _CONFIG)
    assert result.urgency > 0.5
    assert result.score > 0.85  # high sim + high rec + some urgency


def test_score_hit_old_memory():
    hit = {
        "memory_id": "old",
        "distance": 0.5,
        "content": "old content",
        "tier": "episodic",
        "metadata": {},
        "created_at": _hours_ago_iso(10 * 24),  # 10 days — recency=0
    }
    result = _score_hit(hit, _CONFIG)
    assert result.recency == 0.0
    assert result.urgency == 0.0
    # score = W_SIM * similarity only
    expected_sim = _distance_to_similarity(0.5)
    assert abs(result.score - 0.60 * expected_sim) < 0.01


# ---------------------------------------------------------------------------
# get_recommendations — mocked embed + Muninn
# ---------------------------------------------------------------------------


def _make_mock_client(vec: list[float], muninn_hits: list[dict]) -> AsyncMock:
    """Build an AsyncMock httpx.AsyncClient for the full recommend pipeline."""
    embed_resp = MagicMock()
    embed_resp.raise_for_status = MagicMock()
    embed_resp.json.return_value = {"embedding": vec}

    search_resp = MagicMock()
    search_resp.raise_for_status = MagicMock()
    search_resp.json.return_value = muninn_hits

    async def fake_post(url, **kwargs):
        if "embeddings" in url:
            return embed_resp
        if "/search" in url:
            return search_resp
        raise ValueError(f"Unexpected POST to {url}")

    mock_client = AsyncMock()
    mock_client.post = fake_post
    return mock_client


@pytest.mark.asyncio
async def test_get_recommendations_returns_scored_memories():
    hits = [
        {
            "memory_id": "m1",
            "distance": 0.2,
            "tier": "semantic",
            "content": "Brian prefers oat milk",
            "metadata": {},
            "created_at": _now_iso(),
        },
        {
            "memory_id": "m2",
            "distance": 0.5,
            "tier": "episodic",
            "content": "Brian has dentist tomorrow",
            "metadata": {},
            "created_at": _now_iso(),
        },
    ]
    client = _make_mock_client(_FAKE_VEC, hits)
    results = await get_recommendations(_make_event(), _CONFIG, client=client)

    assert isinstance(results, list)
    assert all(isinstance(r, ScoredMemory) for r in results)
    # m1 has smaller distance (more similar) — should rank higher
    assert results[0].id == "m1"
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_get_recommendations_filters_below_min_score():
    # High distance → low similarity → low composite score
    hits = [
        {
            "memory_id": "irrelevant",
            "distance": 1.9,       # near-zero similarity
            "tier": "semantic",
            "content": "Completely unrelated",
            "metadata": {},
            "created_at": _hours_ago_iso(20 * 24),  # old → recency=0
        }
    ]
    client = _make_mock_client(_FAKE_VEC, hits)
    results = await get_recommendations(
        _make_event(), _CONFIG, min_score=0.35, client=client
    )
    assert results == []


@pytest.mark.asyncio
async def test_get_recommendations_empty_muninn():
    client = _make_mock_client(_FAKE_VEC, [])
    results = await get_recommendations(_make_event(), _CONFIG, client=client)
    assert results == []


@pytest.mark.asyncio
async def test_get_recommendations_top_k_limit():
    hits = [
        {
            "memory_id": f"m{i}",
            "distance": 0.1,
            "tier": "semantic",
            "content": f"Memory {i}",
            "metadata": {},
            "created_at": _now_iso(),
        }
        for i in range(10)
    ]
    client = _make_mock_client(_FAKE_VEC, hits)
    results = await get_recommendations(_make_event(), _CONFIG, top_k=3, client=client)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_get_recommendations_urgency_boosts_rank():
    """A memory with an imminent deadline should outscore one without."""
    urgent_deadline = _hours_from_now_iso(0.5)
    hits = [
        {
            "memory_id": "urgent",
            "distance": 0.6,  # lower similarity
            "tier": "semantic",
            "content": "Pick up kids in 30 min",
            "metadata": {"deadline_utc": urgent_deadline},
            "created_at": _hours_ago_iso(1),
        },
        {
            "memory_id": "normal",
            "distance": 0.3,  # higher similarity, no urgency
            "tier": "semantic",
            "content": "Brian likes oat milk",
            "metadata": {},
            "created_at": _hours_ago_iso(1),
        },
    ]
    client = _make_mock_client(_FAKE_VEC, hits)
    results = await get_recommendations(_make_event(), _CONFIG, min_score=0.0, client=client)

    ids = [r.id for r in results]
    # Urgent item should appear due to urgency boost despite lower similarity
    assert "urgent" in ids
    urgent_result = next(r for r in results if r.id == "urgent")
    assert urgent_result.urgency > 0.5
