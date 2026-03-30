"""Shared pytest fixtures for room_node tests."""

import io
import struct
import wave

import numpy as np
import pytest

SAMPLE_RATE = 16000
EMBEDDING_DIM = 256


def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate a minimal valid PCM WAV file in memory."""
    n_samples = int(duration_s * sample_rate)
    rng = np.random.default_rng(0)
    # Low-amplitude noise resembling silence — use for VAD-off tests
    samples = (rng.standard_normal(n_samples) * 100).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def make_speech_wav_bytes(duration_s: float = 1.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate a WAV file with higher-amplitude signal to trigger VAD."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    # 440 Hz tone — loud enough to pass VAD threshold in tests
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def silence_wav() -> bytes:
    return make_wav_bytes(duration_s=1.0)


@pytest.fixture
def speech_wav() -> bytes:
    return make_speech_wav_bytes(duration_s=2.0)


@pytest.fixture
def speech_array() -> np.ndarray:
    """Return a float32 numpy array of a 440 Hz tone — used as mock audio segment."""
    duration_s = 2.0
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)
