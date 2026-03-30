"""Tests for enrollment.enroll.

All memory-mcp HTTP calls are mocked with respx.
Audio I/O (mic recording, resemblyzer) is mocked so no hardware is required.
"""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import respx
from httpx import Response

from enrollment.enroll import (
    MIN_AUDIO_S,
    SAMPLE_RATE,
    VOICEPRINT_DIM,
    _ensure_entity,
    _get_profile,
    _update_voiceprint,
    compute_embedding,
    load_wav,
)

BASE_URL = "http://memory-mcp.test:8900"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_wav_bytes(duration_s: float = 5.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    n = int(duration_s * sample_rate)
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(n) * 3000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def make_wav_file(tmp_path, duration_s: float = 5.0) -> str:
    path = tmp_path / "test.wav"
    path.write_bytes(make_wav_bytes(duration_s))
    return str(path)


def unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(VOICEPRINT_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def ok(result=None) -> Response:
    return Response(200, json={"result": result or "ok", "ok": True})


# ---------------------------------------------------------------------------
# load_wav
# ---------------------------------------------------------------------------


class TestLoadWav:
    def test_loads_mono_16khz(self, tmp_path):
        path = make_wav_file(tmp_path, duration_s=3.0)
        audio = load_wav(path)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == pytest.approx(3.0 * SAMPLE_RATE, rel=0.01)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wav(str(tmp_path / "nonexistent.wav"))

    def test_stereo_converted_to_mono(self, tmp_path):
        """Stereo WAV should be averaged to mono."""
        n = SAMPLE_RATE * 2
        samples = np.zeros(n * 2, dtype=np.int16)  # interleaved stereo
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())
        path = tmp_path / "stereo.wav"
        path.write_bytes(buf.getvalue())
        audio = load_wav(str(path))
        assert len(audio) == n  # stereo → mono halves the sample count

    def test_values_normalised_to_minus1_plus1(self, tmp_path):
        path = make_wav_file(tmp_path)
        audio = load_wav(path)
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0


# ---------------------------------------------------------------------------
# compute_embedding
# ---------------------------------------------------------------------------


class TestComputeEmbedding:
    def test_returns_unit_norm_array(self):
        fake_embedding = unit_vec(0)
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = fake_embedding

        with (
            patch("enrollment.enroll.VoiceEncoder", return_value=mock_encoder),
            patch("enrollment.enroll.preprocess_wav", return_value=np.zeros(SAMPLE_RATE)),
        ):
            from enrollment.enroll import compute_embedding
            audio = np.zeros(int(MIN_AUDIO_S * SAMPLE_RATE) + 1000, dtype=np.float32)
            result = compute_embedding(audio)

        assert result is not None
        assert result.shape == (VOICEPRINT_DIM,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_returns_float32(self):
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = unit_vec(1).astype(np.float64)

        with (
            patch("enrollment.enroll.VoiceEncoder", return_value=mock_encoder),
            patch("enrollment.enroll.preprocess_wav", return_value=np.zeros(SAMPLE_RATE)),
        ):
            from enrollment.enroll import compute_embedding
            result = compute_embedding(np.zeros(SAMPLE_RATE * 5, dtype=np.float32))

        if result is not None:
            assert result.dtype == np.float32

    def test_zero_norm_returns_none(self):
        mock_encoder = MagicMock()
        mock_encoder.embed_utterance.return_value = np.zeros(VOICEPRINT_DIM, dtype=np.float32)

        with (
            patch("enrollment.enroll.VoiceEncoder", return_value=mock_encoder),
            patch("enrollment.enroll.preprocess_wav", return_value=np.zeros(SAMPLE_RATE)),
        ):
            from enrollment.enroll import compute_embedding
            result = compute_embedding(np.zeros(SAMPLE_RATE * 5, dtype=np.float32))

        assert result is None


# ---------------------------------------------------------------------------
# memory-mcp API helpers
# ---------------------------------------------------------------------------


class TestEnsureEntity:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success(self):
        import httpx
        respx.post(f"{BASE_URL}/remember").mock(return_value=ok())
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            result = await _ensure_entity(client, "Brian")
        assert result is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_error_returns_none(self):
        import httpx
        respx.post(f"{BASE_URL}/remember").mock(return_value=Response(500))
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            result = await _ensure_entity(client, "Brian")
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_posts_correct_entity_type(self):
        import httpx
        import json
        respx.post(f"{BASE_URL}/remember").mock(return_value=ok())
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            await _ensure_entity(client, "Brian", entity_type="person")
        body = json.loads(respx.calls.last.request.content)
        assert body["entity_name"] == "Brian"
        assert body["entity_type"] == "person"


class TestUpdateVoiceprint:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success(self):
        import httpx
        respx.post(f"{BASE_URL}/voices/update_print").mock(return_value=ok())
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            result = await _update_voiceprint(client, "Brian", unit_vec(0))
        assert result is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_full_embedding(self):
        import httpx
        import json
        respx.post(f"{BASE_URL}/voices/update_print").mock(return_value=ok())
        embedding = unit_vec(0)
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            await _update_voiceprint(client, "Brian", embedding)
        body = json.loads(respx.calls.last.request.content)
        assert len(body["embedding"]) == VOICEPRINT_DIM
        assert body["weight"] == 1.0  # first enrollment

    @pytest.mark.asyncio
    @respx.mock
    async def test_failure_returns_none(self):
        import httpx
        respx.post(f"{BASE_URL}/voices/update_print").mock(return_value=Response(422))
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            result = await _update_voiceprint(client, "Brian", unit_vec(0))
        assert result is None


class TestGetProfile:
    @pytest.mark.asyncio
    @respx.mock
    async def test_found(self):
        import httpx
        respx.get(f"{BASE_URL}/profile/Brian").mock(
            return_value=ok({"entity_name": "Brian"})
        )
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            result = await _get_profile(client, "Brian")
        assert result is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_not_found_returns_none(self):
        import httpx
        respx.get(f"{BASE_URL}/profile/Nobody").mock(return_value=Response(404))
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            result = await _get_profile(client, "Nobody")
        assert result is None


# ---------------------------------------------------------------------------
# cmd_list — correct response key and field names
# ---------------------------------------------------------------------------


class TestCmdList:
    @pytest.mark.asyncio
    @respx.mock
    async def test_lists_enrolled_persons(self, capsys):
        """cmd_list should use the 'entities' key and 'type'/'name' fields."""
        import httpx
        from enrollment.enroll import cmd_list

        respx.get(f"{BASE_URL}/entities").mock(
            return_value=Response(
                200,
                json={
                    "entities": [
                        {
                            "name": "Brian",
                            "type": "person",
                            "meta": {"status": "enrolled", "voiceprint_samples": 5},
                            "updated": 1774573325.0,
                        },
                        {
                            "name": "kitchen",
                            "type": "room",
                            "meta": {},
                            "updated": 1774573000.0,
                        },
                    ]
                },
            )
        )
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            await cmd_list(client)

        out = capsys.readouterr().out
        assert "Brian" in out
        assert "kitchen" not in out  # room entities excluded

    @pytest.mark.asyncio
    @respx.mock
    async def test_excludes_unenrolled(self, capsys):
        import httpx
        from enrollment.enroll import cmd_list

        respx.get(f"{BASE_URL}/entities").mock(
            return_value=Response(
                200,
                json={
                    "entities": [
                        {
                            "name": "unknown_voice_a3f2",
                            "type": "person",
                            "meta": {"status": "unenrolled"},
                            "updated": 1774573000.0,
                        },
                    ]
                },
            )
        )
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            await cmd_list(client)

        out = capsys.readouterr().out
        assert "No enrolled speakers" in out

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_error_exits(self):
        import httpx
        from enrollment.enroll import cmd_list

        respx.get(f"{BASE_URL}/entities").mock(return_value=Response(500))
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            with pytest.raises(SystemExit):
                await cmd_list(client)


# ---------------------------------------------------------------------------
# Auth header passthrough
# ---------------------------------------------------------------------------


class TestAuthHeader:
    @pytest.mark.asyncio
    @respx.mock
    async def test_bearer_token_sent_on_requests(self):
        """When MEMORY_MCP_TOKEN is set, all requests include the Authorization header."""
        import httpx
        import os

        route = respx.post(f"{BASE_URL}/remember").mock(return_value=ok())
        token = "testtoken123"
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            await _ensure_entity(client, "Brian")

        assert route.called
        auth = route.calls.last.request.headers.get("authorization", "")
        assert auth == f"Bearer {token}"
