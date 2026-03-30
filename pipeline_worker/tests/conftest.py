"""Shared pytest fixtures for pipeline_worker tests."""

import numpy as np
import pytest

from pipeline_worker.models import AudioPayload, EmotionReading

EMBEDDING_DIM = 256


@pytest.fixture
def random_embedding() -> np.ndarray:
    """Return a random unit-norm 256-dim embedding."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def another_embedding() -> np.ndarray:
    """Return a second random unit-norm embedding (different seed)."""
    rng = np.random.default_rng(99)
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def sample_payload() -> AudioPayload:
    """Minimal valid AudioPayload for use in tests."""
    return AudioPayload(
        room="kitchen",
        timestamp="2026-03-22T10:30:00Z",
        transcript="I need to pick up groceries tomorrow",
        whisper_confidence=0.92,
        whisper_model="small",
        doa=247,
        emotion=EmotionReading(valence=0.6, arousal=0.3),
        voiceprint=[0.01] * EMBEDDING_DIM,
    )


@pytest.fixture
def low_confidence_payload(sample_payload: AudioPayload) -> AudioPayload:
    """Payload with whisper_confidence below threshold — includes audio_clip_b64."""
    import base64

    fake_wav = b"RIFF" + b"\x00" * 40  # minimal fake WAV header bytes
    return sample_payload.model_copy(
        update={
            "whisper_confidence": 0.60,
            "audio_clip_b64": base64.b64encode(fake_wav).decode(),
        }
    )
