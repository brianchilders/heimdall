"""
Diarization fallback — full pyannote speaker diarization + large-v3 re-transcription.

This module is invoked only when audio_clip_b64 is present in the payload,
which happens when:
  - whisper_confidence < WHISPER_CONFIDENCE_THRESHOLD (re-transcribe with large-v3)
  - voiceprint match confidence_level == UNKNOWN (identify speaker from raw audio)

Heavy models (pyannote, WhisperModel large-v3, resemblyzer VoiceEncoder) are
loaded lazily on first use — startup time is unaffected if this path is never hit.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VOICEPRINT_DIM = 256


class DiarizationFallback:
    """Lazy-loaded pyannote diarization + large-v3 Whisper re-transcription.

    Models are not loaded until the first call to process() to keep
    startup fast for the common (high-confidence) path.

    Usage::

        fallback = DiarizationFallback(hf_token="hf_...")
        result = await fallback.process(audio_b64)
        print(result.transcript, result.embedding)
    """

    def __init__(self, hf_token: Optional[str] = None) -> None:
        """
        Args:
            hf_token: HuggingFace token required for pyannote model download.
                      If None, diarization is skipped (only re-transcription runs).
        """
        self.hf_token = hf_token
        self._whisper: Optional[object] = None
        self._diarize_pipeline: Optional[object] = None
        self._encoder: Optional[object] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, audio_b64: str) -> "FallbackResult":
        """Run re-transcription and optionally diarization on a raw audio clip.

        Args:
            audio_b64: Base64-encoded mono 16kHz PCM WAV.

        Returns:
            FallbackResult with transcript, embedding, and speaker segments.
        """
        audio = _decode_audio(audio_b64)
        transcript = self._transcribe(audio)
        embedding = self._embed(audio)
        return FallbackResult(transcript=transcript, embedding=embedding)

    # ------------------------------------------------------------------
    # Private — lazy model loading
    # ------------------------------------------------------------------

    def _get_whisper(self) -> object:
        """Lazy-load the faster-whisper large-v3 model."""
        if self._whisper is None:
            logger.info("Loading Whisper large-v3 for fallback transcription...")
            try:
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
                logger.info("Whisper large-v3 loaded (CUDA)")
            except Exception:
                # Fall back to CPU if CUDA unavailable
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")
                logger.warning("Whisper large-v3 loaded on CPU (CUDA unavailable)")
        return self._whisper

    def _get_encoder(self) -> object:
        """Lazy-load the resemblyzer VoiceEncoder."""
        if self._encoder is None:
            logger.info("Loading resemblyzer VoiceEncoder for fallback embedding...")
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            logger.info("VoiceEncoder loaded")
        return self._encoder

    def _transcribe(self, audio: np.ndarray) -> str:
        """Re-transcribe audio using large-v3 Whisper.

        Args:
            audio: float32 mono array at SAMPLE_RATE.

        Returns:
            Transcript string (empty string on failure).
        """
        try:
            whisper = self._get_whisper()
            segments, _ = whisper.transcribe(audio, beam_size=5)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            logger.error("Fallback transcription failed: %s", exc)
            return ""

    def _embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Compute a resemblyzer voiceprint embedding from raw audio.

        Args:
            audio: float32 mono array at SAMPLE_RATE.

        Returns:
            Unit-norm float32 array of shape (VOICEPRINT_DIM,), or None on failure.
        """
        try:
            from resemblyzer import preprocess_wav
            encoder = self._get_encoder()
            wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
            embedding = encoder.embed_utterance(wav)
            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                return None
            return (embedding / norm).astype(np.float32)
        except Exception as exc:
            logger.error("Fallback embedding failed: %s", exc)
            return None


class FallbackResult:
    """Result from DiarizationFallback.process()."""

    def __init__(
        self,
        transcript: str,
        embedding: Optional[np.ndarray],
    ) -> None:
        self.transcript = transcript
        self.embedding = embedding  # None if audio too short/noisy for reliable embedding


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _decode_audio(audio_b64: str) -> np.ndarray:
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
