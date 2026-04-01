"""
Unit tests for verdandi.embedder.

Mocks Ollama — no network calls required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nornir.models import ContextEvent
from verdandi.config import VerdandiConfig
from verdandi.embedder import _context_to_text, embed_context

FIXTURES = Path(__file__).parent / "fixtures" / "sample_events.json"

_CONFIG = VerdandiConfig(
    ollama_url="http://localhost:11434",
    muninn_url="http://localhost:8900",
    embed_model="nomic-embed-text",
)

_FAKE_VEC = [0.1] * 768


def _event_from_fixture(index: int) -> ContextEvent:
    data = json.loads(FIXTURES.read_text())[index]
    return ContextEvent(
        who=data["who"],
        transcript=data["transcript"],
        emotion=data["emotion"],
        location=data["location"],
        local_time=data["local_time"],
        speaker_confidence=data.get("speaker_confidence", 0.0),
        objects_visible=data.get("objects_visible", []),
        activity=data.get("activity"),
    )


# ---------------------------------------------------------------------------
# _context_to_text
# ---------------------------------------------------------------------------


def test_context_to_text_basic():
    event = _event_from_fixture(0)
    text = _context_to_text(event)
    assert "Brian" in text
    assert "kitchen" in text
    assert "pick up the kids" in text
    assert "neutral" in text


def test_context_to_text_with_objects():
    event = _event_from_fixture(0)
    text = _context_to_text(event)
    assert "coffee_mug" in text
    assert "laptop" in text


def test_context_to_text_no_objects():
    event = _event_from_fixture(1)
    text = _context_to_text(event)
    # No objects_visible — Visible line should be absent
    assert "Visible" not in text


def test_context_to_text_with_activity():
    event = _event_from_fixture(0)
    text = _context_to_text(event)
    assert "morning_routine" in text


def test_context_to_text_no_activity():
    event = _event_from_fixture(1)
    text = _context_to_text(event)
    assert "Activity" not in text


# ---------------------------------------------------------------------------
# embed_context — mock Ollama
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_context_calls_ollama():
    """embed_context posts to the correct Ollama URL with the right model."""
    event = _event_from_fixture(0)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": _FAKE_VEC}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await embed_context(event, _CONFIG, client=mock_client)

    assert result == _FAKE_VEC
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "nomic-embed-text" in str(call_kwargs)
    assert "localhost:11434" in str(call_kwargs)


@pytest.mark.asyncio
async def test_embed_context_returns_vector_length():
    event = _event_from_fixture(2)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": _FAKE_VEC}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    result = await embed_context(event, _CONFIG, client=mock_client)
    assert len(result) == 768


@pytest.mark.asyncio
async def test_embed_context_passes_flattened_text():
    """The prompt sent to Ollama must contain the transcript."""
    event = _event_from_fixture(0)

    captured = {}

    async def fake_post(url, **kwargs):
        captured["json"] = kwargs.get("json", {})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"embedding": _FAKE_VEC}
        return resp

    mock_client = AsyncMock()
    mock_client.post = fake_post

    await embed_context(event, _CONFIG, client=mock_client)
    assert "pick up the kids" in captured["json"]["prompt"]
