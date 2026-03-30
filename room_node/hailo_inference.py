"""
On-device ML inference: Whisper transcription + emotion detection.

Primary path  — Hailo-8 accelerator (.hef compiled models)
Fallback path — faster-whisper on CPU (automatic when HAILO_ENABLED=false
                or the .hef file is not found)

Emotion model
-------------
The Hailo path is a stub pending a compiled emotion .hef from the Hailo Model
Zoo.  The CPU fallback returns a neutral estimate (valence=0.5, arousal=0.5).
Replace the stub with the real Hailo SDK call once the model is compiled.

Voiceprint embedding
--------------------
The resemblyzer VoiceEncoder runs on CPU regardless of Hailo availability —
it is a small GE2E model and does not benefit significantly from the Hailo
accelerator.  The encoder is loaded lazily on first use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VOICEPRINT_DIM = 256


@dataclass
class InferenceResult:
    """Output of one inference pass on a speech segment."""

    transcript: str
    whisper_confidence: float           # mean segment log-prob → [0, 1]
    whisper_model: str                  # e.g. "small", "large-v3"
    voiceprint: Optional[np.ndarray]    # 256-dim unit-norm float32, or None
    emotion_valence: float              # 0.0 (negative) – 1.0 (positive)
    emotion_arousal: float              # 0.0 (calm) – 1.0 (excited)


class InferenceEngine:
    """Run Whisper + emotion + resemblyzer on a speech segment.

    Selects the Hailo-8 or CPU backend automatically based on configuration
    and hardware availability.

    Usage::

        engine = InferenceEngine(hailo_enabled=True, whisper_hef="./models/whisper_small.hef")
        result = engine.run(audio_array)
    """

    def __init__(
        self,
        hailo_enabled: bool = True,
        whisper_hef: str = "./models/whisper_small.hef",
        emotion_hef: str = "./models/emotion.hef",
        whisper_fallback_model: str = "small",
    ) -> None:
        self.whisper_fallback_model = whisper_fallback_model
        self._whisper_cpu: Optional[object] = None
        self._encoder: Optional[object] = None

        # Decide which backend to use for Whisper
        self._use_hailo = False
        if hailo_enabled:
            self._use_hailo = _hailo_available(whisper_hef)
            if not self._use_hailo:
                logger.warning(
                    "Hailo-8 requested but not available (hef=%s) — falling back to faster-whisper",
                    whisper_hef,
                )

        if self._use_hailo:
            logger.info("Hailo-8 backend selected for Whisper")
            self._whisper_hef = whisper_hef
            self._emotion_hef = emotion_hef
        else:
            logger.info("CPU (faster-whisper) backend selected for Whisper")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, audio: np.ndarray) -> InferenceResult:
        """Run all inference models on a speech segment.

        Args:
            audio: float32 mono array at SAMPLE_RATE Hz.

        Returns:
            InferenceResult with transcript, confidence, emotion, voiceprint.
        """
        if self._use_hailo:
            transcript, confidence, model_name = self._transcribe_hailo(audio)
            valence, arousal = self._emotion_hailo(audio)
        else:
            transcript, confidence, model_name = self._transcribe_cpu(audio)
            valence, arousal = self._emotion_cpu(audio)

        voiceprint = self._embed(audio)

        return InferenceResult(
            transcript=transcript,
            whisper_confidence=confidence,
            whisper_model=model_name,
            voiceprint=voiceprint,
            emotion_valence=valence,
            emotion_arousal=arousal,
        )

    # ------------------------------------------------------------------
    # Whisper — Hailo path (stub)
    # ------------------------------------------------------------------

    def _transcribe_hailo(self, audio: np.ndarray) -> tuple[str, float, str]:
        """Transcribe using the Hailo-8 compiled Whisper model.

        TODO: Replace stub with real Hailo SDK inference once the .hef is
        compiled from the Hailo Model Zoo whisper model.

        Reference:
            https://github.com/hailo-ai/hailo_model_zoo/tree/master/hailo_models/speech_recognition

        Returns:
            (transcript, confidence, model_name)
        """
        # --- Hailo SDK stub ---
        # from hailo_platform import VDevice, HEF, ConfigureParams, FormatType
        # with VDevice() as device:
        #     hef = HEF(self._whisper_hef)
        #     ...
        logger.warning("Hailo Whisper stub called — falling back to CPU")
        return self._transcribe_cpu(audio)

    # ------------------------------------------------------------------
    # Whisper — CPU fallback (faster-whisper)
    # ------------------------------------------------------------------

    def _transcribe_cpu(self, audio: np.ndarray) -> tuple[str, float, str]:
        """Transcribe using faster-whisper on CPU.

        Args:
            audio: float32 mono array at SAMPLE_RATE Hz.

        Returns:
            (transcript, confidence, model_name)
        """
        model = self._get_whisper_cpu()
        try:
            segments, _ = model.transcribe(audio, beam_size=5, language="en")
            parts: list[str] = []
            log_probs: list[float] = []
            for seg in segments:
                parts.append(seg.text.strip())
                log_probs.append(seg.avg_logprob)

            transcript = " ".join(parts).strip()
            # Convert mean log-prob to a 0–1 confidence score
            # avg_logprob is typically in [-2, 0]; exp(0) = 1.0
            import math
            confidence = math.exp(sum(log_probs) / len(log_probs)) if log_probs else 0.0
            confidence = max(0.0, min(1.0, confidence))
            return transcript, confidence, self.whisper_fallback_model
        except Exception as exc:
            logger.error("CPU transcription failed: %s", exc)
            return "", 0.0, self.whisper_fallback_model

    def _get_whisper_cpu(self) -> object:
        """Lazy-load the faster-whisper CPU model."""
        if self._whisper_cpu is None:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper model: %s", self.whisper_fallback_model)
            self._whisper_cpu = WhisperModel(
                self.whisper_fallback_model,
                device="cpu",
                compute_type="int8",
            )
            logger.info("faster-whisper loaded")
        return self._whisper_cpu

    # ------------------------------------------------------------------
    # Emotion — Hailo path (stub)
    # ------------------------------------------------------------------

    def _emotion_hailo(self, audio: np.ndarray) -> tuple[float, float]:
        """Detect emotion using the Hailo-8 compiled emotion model.

        TODO: Replace stub once emotion .hef is compiled and validated.

        Returns:
            (valence, arousal) both in [0, 1].
        """
        logger.debug("Hailo emotion stub — returning neutral values")
        return self._emotion_cpu(audio)

    # ------------------------------------------------------------------
    # Emotion — CPU fallback
    # ------------------------------------------------------------------

    def _emotion_cpu(self, audio: np.ndarray) -> tuple[float, float]:
        """Return a neutral emotion estimate.

        A full CPU emotion model (e.g. speechbrain wav2vec2-based) can be
        substituted here.  For now, returns neutral to keep the dependency
        footprint small — the emotion field is informational.

        Returns:
            (valence=0.5, arousal=0.5) — neutral estimate.
        """
        return 0.5, 0.5

    # ------------------------------------------------------------------
    # Voiceprint embedding (resemblyzer, always CPU)
    # ------------------------------------------------------------------

    def _embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Compute a 256-dim GE2E resemblyzer embedding.

        Returns None if the audio is too short for a reliable embedding
        (resemblyzer requires at least ~1.6s of speech).

        Args:
            audio: float32 mono array at SAMPLE_RATE Hz.

        Returns:
            Unit-norm float32 array of shape (256,), or None.
        """
        min_samples = int(1.6 * SAMPLE_RATE)
        if len(audio) < min_samples:
            logger.debug(
                "Audio too short for voiceprint embedding (%.2fs < 1.6s)",
                len(audio) / SAMPLE_RATE,
            )
            return None

        try:
            encoder = self._get_encoder()
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
            embedding = encoder.embed_utterance(wav)
            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                return None
            return (embedding / norm).astype(np.float32)
        except Exception as exc:
            logger.error("Resemblyzer embedding failed: %s", exc)
            return None

    def _get_encoder(self) -> object:
        """Lazy-load the resemblyzer VoiceEncoder."""
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            logger.info("Loading resemblyzer VoiceEncoder")
            self._encoder = VoiceEncoder()
        return self._encoder


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _hailo_available(hef_path: str) -> bool:
    """Check if the Hailo SDK is importable and the .hef file exists.

    Args:
        hef_path: Path to the compiled .hef model file.

    Returns:
        True if both the SDK and the model file are present.
    """
    if not Path(hef_path).exists():
        logger.debug("Hailo .hef not found: %s", hef_path)
        return False
    try:
        import hailo_platform  # noqa: F401
        return True
    except ImportError:
        logger.debug("hailo_platform SDK not installed")
        return False
