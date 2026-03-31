"""
Pluggable speaker embedding backends.

Selecting a backend
-------------------
Set SPEAKER_ENCODER in pipeline_worker/.env:

    SPEAKER_ENCODER=resemblyzer   # default — 256-dim GE2E (CPU only)
    SPEAKER_ENCODER=ecapa_tdnn    # 192-dim ECAPA-TDNN via SpeechBrain (GPU recommended)
    SPEAKER_ENCODER=titanet       # 192-dim TitaNet-L via NVIDIA NeMo (CUDA 12 required)

All backends implement SpeakerEncoder and return unit-norm float32 arrays.
The rest of the pipeline is encoder-agnostic — only the embedding dimension differs.

Switching encoders
------------------
Stored voiceprints are tied to the encoder that produced them.  Switching
encoders requires either:

  - POST /recompute_embeddings  — recomputes from stored enrollment audio automatically
  - Re-enrolling all speakers   — if enrollment audio was not retained

The pipeline worker stores enrollment audio by default (STORE_ENROLLMENT_AUDIO=true)
so that recomputing is always possible without re-recording.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Embedding dimension for each backend — used for schema validation.
ENCODER_DIMS: dict[str, int] = {
    "resemblyzer": 256,
    "ecapa_tdnn": 192,
    "titanet": 192,
}


@runtime_checkable
class SpeakerEncoder(Protocol):
    """Protocol for speaker embedding backends.

    All implementations produce unit-norm float32 arrays.
    The output dimension is fixed per backend (see ENCODER_DIMS).
    """

    name: str  # canonical backend identifier, e.g. 'ecapa_tdnn'
    dim: int   # output embedding dimension

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        """Compute a unit-norm speaker embedding from mono float32 audio.

        Args:
            audio: float32 mono array at sample_rate Hz.
            sample_rate: Audio sample rate (default 16000).

        Returns:
            Unit-norm float32 array of shape (self.dim,), or None on failure
            (e.g. audio too short, model error).
        """
        ...


# ---------------------------------------------------------------------------
# Resemblyzer backend — GE2E 2018, 256-dim, CPU-only
# ---------------------------------------------------------------------------


class ResemblyzerEncoder:
    """GE2E speaker encoder via resemblyzer (256-dim, CPU-only).

    The original baseline.  Struggles on clips shorter than ~1.5s.
    No GPU or HuggingFace token required.

    Requires: pip install resemblyzer
    """

    name = "resemblyzer"
    dim = 256

    def __init__(self) -> None:
        self._model: Optional[object] = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from resemblyzer import VoiceEncoder
                self._model = VoiceEncoder()
                logger.info("ResemblyzerEncoder loaded")
            except ImportError as exc:
                raise ImportError(
                    "resemblyzer is required for this encoder. "
                    "Install with: pip install resemblyzer"
                ) from exc
        return self._model

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        """Compute a 256-dim GE2E embedding.

        Args:
            audio: float32 mono array.
            sample_rate: Source sample rate.

        Returns:
            Unit-norm float32 array of shape (256,), or None on failure.
        """
        try:
            from resemblyzer import preprocess_wav
            model = self._get_model()
            wav = preprocess_wav(audio, source_sr=sample_rate)
            embedding = model.embed_utterance(wav).astype(np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                return None
            return embedding / norm
        except Exception as exc:
            logger.error("ResemblyzerEncoder.embed failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# ECAPA-TDNN backend — SpeechBrain 2021, 192-dim, GPU recommended
# ---------------------------------------------------------------------------


class EcapaTdnnEncoder:
    """ECAPA-TDNN speaker encoder via SpeechBrain (192-dim).

    Significantly better than resemblyzer, especially on short clips.
    No HuggingFace token required — model downloads automatically from
    the public speechbrain/spkrec-ecapa-voxceleb hub repo.

    GPU recommended but runs on CPU.  Compatible with Jetson AGX Xavier
    (CUDA 11.4 / JetPack 5.x).

    Requires: pip install speechbrain
    """

    name = "ecapa_tdnn"
    dim = 192

    def __init__(self, device: str = "cpu") -> None:
        """
        Args:
            device: PyTorch device string ('cpu', 'cuda', 'cuda:0').
        """
        self.device = device
        self._model: Optional[object] = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device},
                )
                logger.info("EcapaTdnnEncoder loaded on device=%s", self.device)
            except ImportError as exc:
                raise ImportError(
                    "speechbrain is required for this encoder. "
                    "Install with: pip install speechbrain"
                ) from exc
        return self._model

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        """Compute a 192-dim ECAPA-TDNN embedding.

        Args:
            audio: float32 mono array.
            sample_rate: Source sample rate.

        Returns:
            Unit-norm float32 array of shape (192,), or None on failure.
        """
        try:
            import torch
            model = self._get_model()
            wav = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, T)
            with torch.no_grad():
                embedding = model.encode_batch(wav)  # (1, 1, 192)
            vec = embedding.squeeze().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                return None
            return vec / norm
        except Exception as exc:
            logger.error("EcapaTdnnEncoder.embed failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# TitaNet backend — NVIDIA NeMo 2022, 192-dim, CUDA 12 required
# ---------------------------------------------------------------------------


class TitaNetEncoder:
    """TitaNet-L speaker encoder via NVIDIA NeMo (192-dim).

    State-of-the-art short-clip accuracy.  Requires CUDA 12 and the
    NeMo toolkit — suitable for Jetson Orin NX/AGX Orin but NOT for
    Jetson AGX Xavier (CUDA 11.4 max).  Use EcapaTdnnEncoder on Xavier.

    No HuggingFace token required — model bundled with nemo_toolkit.

    Requires: pip install nemo_toolkit[asr]
    """

    name = "titanet"
    dim = 192

    def __init__(self, model_name: str = "titanet_large", device: str = "cuda") -> None:
        """
        Args:
            model_name: NeMo model name ('titanet_large' or 'titanet_small').
            device: PyTorch device string ('cuda' recommended).
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                import nemo.collections.asr as nemo_asr
                self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                    self.model_name
                )
                self._model.eval()
                logger.info(
                    "TitaNetEncoder (%s) loaded on device=%s", self.model_name, self.device
                )
            except ImportError as exc:
                raise ImportError(
                    "nemo_toolkit is required for this encoder. "
                    "Install with: pip install nemo_toolkit[asr]"
                ) from exc
        return self._model

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        """Compute a 192-dim TitaNet embedding.

        Args:
            audio: float32 mono array.
            sample_rate: Source sample rate.

        Returns:
            Unit-norm float32 array of shape (192,), or None on failure.
        """
        try:
            import torch
            model = self._get_model()
            wav = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, T)
            length = torch.tensor([audio.shape[0]])
            with torch.no_grad():
                _, embedding = model.forward(
                    input_signal=wav, input_signal_length=length
                )
            vec = embedding.squeeze().cpu().detach().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                return None
            return vec / norm
        except Exception as exc:
            logger.error("TitaNetEncoder.embed failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type] = {
    "resemblyzer": ResemblyzerEncoder,
    "ecapa_tdnn": EcapaTdnnEncoder,
    "titanet": TitaNetEncoder,
}


def load_encoder(name: str, device: str = "cpu") -> SpeakerEncoder:
    """Instantiate and return a speaker encoder backend by name.

    Args:
        name: Backend name — 'resemblyzer', 'ecapa_tdnn', or 'titanet'.
        device: PyTorch device string.  Ignored for resemblyzer (CPU-only).

    Returns:
        A SpeakerEncoder instance.

    Raises:
        ValueError: If name is not a known backend.
    """
    cls = _BACKENDS.get(name)
    if cls is None:
        known = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown speaker encoder: {name!r}. Known backends: {known}")
    if name == "resemblyzer":
        return cls()
    return cls(device=device)
