"""
Diarization fallback — large-v3 Whisper re-transcription.

This module is invoked when audio_clip_b64 is present in the payload:
  - capture node profile: always (all inference runs on the pipeline worker)
  - full node profile: when whisper_confidence < WHISPER_CONFIDENCE_THRESHOLD

Speaker embedding is handled separately by the configured SpeakerEncoder
backend (see speaker_encoder.py).  DiarizationFallback is transcription-only.

The WhisperModel is loaded lazily on first use so startup time is unaffected
if this path is never hit.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class DiarizationFallback:
    """Lazy-loaded large-v3 Whisper re-transcription.

    Loaded on first call to process() to keep startup fast.

    Usage::

        fallback = DiarizationFallback()
        transcript = fallback.process(audio_b64)
    """

    def __init__(self) -> None:
        self._whisper: Optional[object] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, audio_b64: str) -> str:
        """Re-transcribe a base64-encoded audio clip with Whisper large-v3.

        Args:
            audio_b64: Base64-encoded mono 16kHz PCM WAV.

        Returns:
            Transcript string, or empty string on failure.
        """
        audio = decode_audio(audio_b64)
        return self._transcribe(audio)

    # ------------------------------------------------------------------
    # Private — lazy model loading
    # ------------------------------------------------------------------

    def _get_whisper(self) -> object:
        """Lazy-load the faster-whisper large-v3 model."""
        if self._whisper is None:
            logger.info("Loading Whisper large-v3...")
            try:
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
                logger.info("Whisper large-v3 loaded (CUDA)")
            except Exception:
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")
                logger.warning("Whisper large-v3 loaded on CPU (CUDA unavailable)")
        return self._whisper

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using large-v3 Whisper.

        Args:
            audio: float32 mono array at SAMPLE_RATE.

        Returns:
            Transcript string, or empty string on failure.
        """
        try:
            whisper = self._get_whisper()
            segments, _ = whisper.transcribe(audio, beam_size=5)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            logger.error("Fallback transcription failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Module-level audio helpers (used by server.py and enrollment)
# ---------------------------------------------------------------------------


def decode_audio(audio_b64: str) -> np.ndarray:
    """Decode a base64 WAV clip to a float32 numpy array.

    Args:
        audio_b64: Base64-encoded mono 16kHz PCM WAV.

    Returns:
        float32 array normalised to [-1.0, 1.0].

    Raises:
        ValueError: If the audio cannot be decoded.
    """
    try:
        wav_bytes = base64.b64decode(audio_b64)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return samples / 32768.0
    except Exception as exc:
        raise ValueError(f"Failed to decode audio clip: {exc}") from exc


def encode_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Encode a float32 mono array as a base64 WAV string (16-bit PCM).

    Args:
        audio: float32 mono array normalised to [-1.0, 1.0].
        sample_rate: Audio sample rate.

    Returns:
        Base64-encoded WAV string.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        pcm = np.clip(audio, -1.0, 1.0)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()
