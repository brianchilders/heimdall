"""Unit tests for nornir shared models."""

from __future__ import annotations

from nornir.models import ContextEvent, RoutingResult, ScoredMemory
from nornir.schema import ALL_TIERS, TIER_EPISODIC, TIER_PATTERN, TIER_SEMANTIC, TIER_TIMESERIES


class TestContextEvent:
    def test_minimal_construction(self):
        event = ContextEvent(
            who="Brian",
            transcript="Good morning.",
            emotion="neutral",
            location="kitchen",
            local_time="2026-04-01T07:00:00",
        )
        assert event.who == "Brian"
        assert event.objects_visible == []
        assert event.people_detected == []
        assert event.activity is None
        assert event.doa_degrees is None
        assert event.speaker_confidence == 0.0

    def test_full_construction(self):
        event = ContextEvent(
            who="Brian",
            transcript="Turn on the lights.",
            emotion="neutral",
            location="living_room",
            local_time="2026-04-01T20:00:00",
            speaker_confidence=0.92,
            doa_degrees=180,
            objects_visible=["sofa", "lamp"],
            people_detected=["Brian"],
            activity="sitting",
        )
        assert event.speaker_confidence == 0.92
        assert event.doa_degrees == 180
        assert event.objects_visible == ["sofa", "lamp"]


class TestScoredMemory:
    def test_construction(self):
        mem = ScoredMemory(
            id="abc-123",
            content="Brian picks up kids at 3pm on Thursdays.",
            score=0.82,
            similarity=0.75,
            recency=0.90,
            urgency=0.0,
            meta={"tags": ["school", "routine"], "deadline_utc": None},
        )
        assert mem.id == "abc-123"
        assert mem.score == 0.82
        assert mem.urgency == 0.0

    def test_default_meta(self):
        mem = ScoredMemory(
            id="x", content="test", score=0.5, similarity=0.5, recency=0.5, urgency=0.0
        )
        assert mem.meta == {}


class TestRoutingResult:
    def test_spoke(self):
        result = RoutingResult(
            spoken_text="Hey Brian, don't forget the school pickup at 3pm.",
            domain="reminder",
            memories_used=["abc-123"],
            output_path="avatar",
            latency_ms=142,
        )
        assert result.spoken_text != ""
        assert result.output_path == "avatar"

    def test_silent(self):
        result = RoutingResult(
            spoken_text="",
            domain="general",
            memories_used=[],
            output_path="silent",
            latency_ms=38,
        )
        assert result.spoken_text == ""
        assert result.output_path == "silent"


class TestSchema:
    def test_all_tiers_complete(self):
        assert TIER_SEMANTIC in ALL_TIERS
        assert TIER_EPISODIC in ALL_TIERS
        assert TIER_TIMESERIES in ALL_TIERS
        assert TIER_PATTERN in ALL_TIERS
        assert len(ALL_TIERS) == 4

    def test_tier_values_are_strings(self):
        for tier in ALL_TIERS:
            assert isinstance(tier, str)
            assert tier == tier.lower()
