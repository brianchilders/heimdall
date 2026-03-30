"""Tests for room_node.capture — VAD utterance collection.

Uses WAV fixture arrays injected via iter_utterances_from_array.
No sounddevice hardware or live microphone required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from room_node.capture import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    _UtteranceCollector,
    iter_utterances_from_array,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vad_model(speech_prob: float = 0.9) -> MagicMock:
    """Return a mock Silero VAD model that always returns speech_prob.

    Uses a numpy scalar instead of a torch tensor so torch is not required
    in the test environment. numpy scalars support .item() the same way.
    """
    model = MagicMock()
    model.return_value = np.float32(speech_prob)
    return model


def silence_array(duration_s: float) -> np.ndarray:
    """float32 array of zeros (silence)."""
    return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def speech_array(duration_s: float) -> np.ndarray:
    """float32 array of a 440 Hz sine wave (speech-like signal)."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


# ---------------------------------------------------------------------------
# _UtteranceCollector
# ---------------------------------------------------------------------------


class TestUtteranceCollector:
    def test_no_speech_yields_nothing(self):
        model = make_vad_model(speech_prob=0.0)  # always silence
        collector = _UtteranceCollector(
            vad_model=model,
            sample_rate=SAMPLE_RATE,
            threshold=0.5,
            min_silence_ms=300,
            speech_pad_ms=100,
            max_utterance_s=30,
        )
        chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        for _ in range(50):  # ~1.5s of silence
            result = collector.feed(chunk)
            assert result is None

    def test_continuous_speech_triggers_timeout(self):
        model = make_vad_model(speech_prob=1.0)  # always speech
        collector = _UtteranceCollector(
            vad_model=model,
            sample_rate=SAMPLE_RATE,
            threshold=0.5,
            min_silence_ms=500,
            speech_pad_ms=0,
            max_utterance_s=1,  # 1s max
        )
        chunk = np.ones(CHUNK_SAMPLES, dtype=np.float32)
        utterances = []
        # Feed 2s worth of chunks — should trigger timeout at 1s
        for _ in range(int(2 * SAMPLE_RATE / CHUNK_SAMPLES)):
            result = collector.feed(chunk)
            if result is not None:
                utterances.append(result)
        assert len(utterances) >= 1
        # Utterance length should be ≈ max_utterance_s
        assert len(utterances[0]) == pytest.approx(SAMPLE_RATE, rel=0.1)

    def test_speech_followed_by_silence_yields_utterance(self):
        # Alternating speech (1) then silence (0) model
        call_count = 0
        speech_chunks = 20   # 0.6s
        silence_chunks = 20  # 0.6s

        def _side_effect(chunk, sr):
            nonlocal call_count
            call_count += 1
            if call_count <= speech_chunks:
                return np.float32(0.9)   # speech
            return np.float32(0.0)       # silence

        model = MagicMock(side_effect=_side_effect)
        collector = _UtteranceCollector(
            vad_model=model,
            sample_rate=SAMPLE_RATE,
            threshold=0.5,
            min_silence_ms=300,
            speech_pad_ms=0,
            max_utterance_s=30,
        )
        chunk = np.ones(CHUNK_SAMPLES, dtype=np.float32)
        utterances = []
        for _ in range(speech_chunks + silence_chunks):
            result = collector.feed(chunk)
            if result is not None:
                utterances.append(result)
        assert len(utterances) == 1

    def test_utterance_is_float32(self):
        calls = [0]

        def _side_effect(chunk, sr):
            calls[0] += 1
            return np.float32(0.9 if calls[0] <= 10 else 0.0)

        model = MagicMock(side_effect=_side_effect)
        collector = _UtteranceCollector(
            vad_model=model,
            sample_rate=SAMPLE_RATE,
            threshold=0.5,
            min_silence_ms=200,
            speech_pad_ms=0,
            max_utterance_s=30,
        )
        chunk = np.ones(CHUNK_SAMPLES, dtype=np.float32)
        utterance = None
        for _ in range(30):
            result = collector.feed(chunk)
            if result is not None:
                utterance = result
                break
        if utterance is not None:
            assert utterance.dtype == np.float32


# ---------------------------------------------------------------------------
# iter_utterances_from_array
# ---------------------------------------------------------------------------


class TestIterUtterancesFromArray:
    def test_all_silence_yields_nothing(self):
        model = make_vad_model(speech_prob=0.0)
        audio = silence_array(3.0)
        utterances = list(
            iter_utterances_from_array(audio, vad_model=model, threshold=0.5)
        )
        assert utterances == []

    def test_all_speech_yields_at_least_one(self):
        model = make_vad_model(speech_prob=1.0)
        audio = speech_array(2.0)
        utterances = list(
            iter_utterances_from_array(
                audio, vad_model=model, threshold=0.5, max_utterance_s=1
            )
        )
        assert len(utterances) >= 1

    def test_utterances_are_numpy_arrays(self):
        model = make_vad_model(speech_prob=1.0)
        audio = speech_array(1.5)
        utterances = list(
            iter_utterances_from_array(audio, vad_model=model, max_utterance_s=1)
        )
        for u in utterances:
            assert isinstance(u, np.ndarray)
            assert u.dtype == np.float32

    def test_respects_max_utterance_s(self):
        model = make_vad_model(speech_prob=1.0)
        audio = speech_array(5.0)
        max_s = 1
        utterances = list(
            iter_utterances_from_array(
                audio,
                vad_model=model,
                max_utterance_s=max_s,
                min_silence_ms=9999,  # never end on silence
            )
        )
        for u in utterances:
            assert len(u) <= (max_s + 0.1) * SAMPLE_RATE

    def test_flush_trailing_speech(self):
        """Speech that reaches the end of the array should be flushed."""
        model = make_vad_model(speech_prob=1.0)
        audio = speech_array(2.0)
        utterances = list(
            iter_utterances_from_array(
                audio,
                vad_model=model,
                threshold=0.5,
                min_silence_ms=9999,   # won't end on silence
                max_utterance_s=30,    # won't time out
            )
        )
        # Should get the trailing speech flushed at end of array
        assert len(utterances) >= 1
