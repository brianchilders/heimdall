"""Tests for pipeline_worker.memory_client.MemoryClient.

All HTTP calls are intercepted with respx — no live memory-mcp required.
"""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from pipeline_worker.memory_client import MemoryClient

BASE_URL = "http://memory-mcp.test:8900"


@pytest.fixture
def client() -> MemoryClient:
    return MemoryClient(BASE_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok_response(result: dict | list | str = "ok") -> Response:
    return Response(200, json={"result": result, "ok": True})


def error_response(status: int = 500, text: str = "Internal Error") -> Response:
    return Response(status, text=text)


# ---------------------------------------------------------------------------
# Tier 2 — record
# ---------------------------------------------------------------------------


class TestRecord:
    @pytest.mark.asyncio
    @respx.mock
    async def test_record_success(self, client):
        route = respx.post(f"{BASE_URL}/record").mock(return_value=ok_response())
        result = await client.record(
            entity_name="Brian",
            metric="voice_activity",
            value={"transcript": "hello"},
        )
        assert result is not None
        assert route.called
        body = route.calls[0].request.content
        import json
        data = json.loads(body)
        assert data["entity_name"] == "Brian"
        assert data["metric"] == "voice_activity"
        assert data["source"] == "audio_pipeline"

    @pytest.mark.asyncio
    @respx.mock
    async def test_record_server_error_returns_none(self, client):
        respx.post(f"{BASE_URL}/record").mock(return_value=error_response(500))
        result = await client.record("Brian", "voice_activity", {})
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_record_with_unit(self, client):
        respx.post(f"{BASE_URL}/record").mock(return_value=ok_response())
        await client.record("Brian", "voice_activity", {}, unit="utterance")
        body = respx.calls.last.request.content
        import json
        assert json.loads(body)["unit"] == "utterance"


# ---------------------------------------------------------------------------
# Tier 1.5 — sessions
# ---------------------------------------------------------------------------


class TestSessions:
    @pytest.mark.asyncio
    @respx.mock
    async def test_open_session(self, client):
        respx.post(f"{BASE_URL}/open_session").mock(return_value=ok_response(42))
        result = await client.open_session("Brian")
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["entity_name"] == "Brian"
        assert body["entity_type"] == "person"

    @pytest.mark.asyncio
    @respx.mock
    async def test_open_session_custom_entity_type(self, client):
        respx.post(f"{BASE_URL}/open_session").mock(return_value=ok_response(7))
        import json
        await client.open_session("kitchen", entity_type="room")
        body = json.loads(respx.calls.last.request.content)
        assert body["entity_type"] == "room"

    @pytest.mark.asyncio
    @respx.mock
    async def test_log_turn(self, client):
        respx.post(f"{BASE_URL}/log_turn").mock(return_value=ok_response())
        result = await client.log_turn(
            session_id=42,
            role="user",
            content="I need groceries",
        )
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["session_id"] == 42
        assert body["role"] == "user"
        assert body["content"] == "I need groceries"

    @pytest.mark.asyncio
    @respx.mock
    async def test_close_session(self, client):
        respx.post(f"{BASE_URL}/close_session").mock(return_value=ok_response())
        result = await client.close_session(42)
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["session_id"] == 42

    @pytest.mark.asyncio
    @respx.mock
    async def test_close_session_with_summary(self, client):
        respx.post(f"{BASE_URL}/close_session").mock(return_value=ok_response())
        import json
        await client.close_session(42, summary="Brian discussed groceries.")
        body = json.loads(respx.calls.last.request.content)
        assert body["summary"] == "Brian discussed groceries."

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_session_found(self, client):
        # When found, result is a formatted string (not a dict)
        respx.get(f"{BASE_URL}/get_session/42").mock(
            return_value=ok_response("Session 42 — Brian | 10:00 → 10:05\n  [10:00] user: hello")
        )
        result = await client.get_session(42)
        assert result is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_session_not_found_returns_none(self, client):
        # memory-mcp returns 200 with a string result for unknown sessions
        respx.get(f"{BASE_URL}/get_session/99999").mock(
            return_value=ok_response("No session with id=99999.")
        )
        result = await client.get_session(99999)
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_session_server_error_returns_none(self, client):
        respx.get(f"{BASE_URL}/get_session/42").mock(return_value=error_response(500))
        result = await client.get_session(42)
        assert result is None


# ---------------------------------------------------------------------------
# Tier 1 — semantic memory
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    @pytest.mark.asyncio
    @respx.mock
    async def test_remember(self, client):
        respx.post(f"{BASE_URL}/remember").mock(return_value=ok_response())
        result = await client.remember("Brian", "likes coffee", category="preference")
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["fact"] == "likes coffee"
        assert body["category"] == "preference"

    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_and_remember(self, client):
        import json
        respx.post(f"{BASE_URL}/extract_and_remember").mock(return_value=ok_response())
        result = await client.extract_and_remember("Brian", "I need to pick up groceries tomorrow")
        assert result is not None
        body = json.loads(respx.calls.last.request.content)
        assert body["entity_name"] == "Brian"
        assert body["entity_type"] == "person"
        assert "model" not in body  # not sent when None

    @pytest.mark.asyncio
    @respx.mock
    async def test_relate(self, client):
        respx.post(f"{BASE_URL}/relate").mock(return_value=ok_response())
        result = await client.relate("Brian", "lives_with", "Sarah")
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["entity_a"] == "Brian"
        assert body["rel_type"] == "lives_with"
        assert body["entity_b"] == "Sarah"

    @pytest.mark.asyncio
    @respx.mock
    async def test_recall(self, client):
        respx.post(f"{BASE_URL}/recall").mock(return_value=ok_response([]))
        result = await client.recall("grocery preferences", entity_name="Brian", top_k=3)
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["top_k"] == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_profile_found(self, client):
        # When found, result is a formatted string (not a dict)
        respx.get(f"{BASE_URL}/profile/Brian").mock(
            return_value=ok_response("=== Profile: Brian (person) ===\n\nRELATIONSHIPS:\n  • lives_with → Sarah")
        )
        result = await client.get_profile("Brian")
        assert result is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_profile_not_found_returns_none(self, client):
        # memory-mcp returns 200 with a string result for unknown entities
        respx.get(f"{BASE_URL}/profile/Nobody").mock(
            return_value=ok_response("No entity named 'Nobody'.")
        )
        result = await client.get_profile("Nobody")
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_profile_server_error_returns_none(self, client):
        respx.get(f"{BASE_URL}/profile/Brian").mock(return_value=error_response(500))
        result = await client.get_profile("Brian")
        assert result is None


# ---------------------------------------------------------------------------
# Voice management routes
# ---------------------------------------------------------------------------


class TestVoiceRoutes:
    @pytest.mark.asyncio
    @respx.mock
    async def test_list_unknown_voices(self, client):
        respx.get(f"{BASE_URL}/voices/unknown").mock(
            return_value=ok_response(
                [{"entity_name": "unknown_voice_a3f2", "detection_count": 5}]
            )
        )
        result = await client.list_unknown_voices(limit=10, min_detections=2)
        assert result is not None
        assert result[0]["entity_name"] == "unknown_voice_a3f2"

    @pytest.mark.asyncio
    @respx.mock
    async def test_list_unknown_voices_error_returns_none(self, client):
        respx.get(f"{BASE_URL}/voices/unknown").mock(return_value=error_response(500))
        result = await client.list_unknown_voices()
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_enroll_voice(self, client):
        respx.post(f"{BASE_URL}/voices/enroll").mock(return_value=ok_response())
        result = await client.enroll_voice(
            "unknown_voice_a3f2",
            "Brian",
            display_name="Brian Childers",
        )
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["entity_name"] == "unknown_voice_a3f2"
        assert body["new_name"] == "Brian"
        assert body["display_name"] == "Brian Childers"

    @pytest.mark.asyncio
    @respx.mock
    async def test_enroll_voice_conflict_returns_none(self, client):
        respx.post(f"{BASE_URL}/voices/enroll").mock(return_value=error_response(409))
        result = await client.enroll_voice("unknown_voice_a3f2", "Brian")
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_merge_voices(self, client):
        respx.post(f"{BASE_URL}/voices/merge").mock(return_value=ok_response())
        result = await client.merge_voices("unknown_voice_a3f2", "Brian")
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert body["source_name"] == "unknown_voice_a3f2"
        assert body["target_name"] == "Brian"

    @pytest.mark.asyncio
    @respx.mock
    async def test_update_voiceprint(self, client):
        respx.post(f"{BASE_URL}/voices/update_print").mock(return_value=ok_response())
        embedding = [0.01] * 256
        result = await client.update_voiceprint("Brian", embedding, weight=0.1)
        assert result is not None
        import json
        body = json.loads(respx.calls.last.request.content)
        assert len(body["embedding"]) == 256
        assert body["weight"] == 0.1

    @pytest.mark.asyncio
    @respx.mock
    async def test_update_voiceprint_error_returns_none(self, client):
        respx.post(f"{BASE_URL}/voices/update_print").mock(return_value=error_response(422))
        result = await client.update_voiceprint("Brian", [0.0] * 256)
        assert result is None


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


class TestNetworkErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_connection_refused_returns_none(self, client):
        import httpx
        respx.post(f"{BASE_URL}/record").mock(side_effect=httpx.ConnectError("refused"))
        result = await client.record("Brian", "voice_activity", {})
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_returns_none(self, client):
        import httpx
        respx.post(f"{BASE_URL}/record").mock(side_effect=httpx.TimeoutException("timeout"))
        result = await client.record("Brian", "voice_activity", {})
        assert result is None
