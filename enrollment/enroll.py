"""
Speaker enrollment CLI.

Ships raw audio to the pipeline worker POST /enroll endpoint.  The pipeline
worker computes the embedding using its configured SpeakerEncoder backend and
stores both the embedding and the raw audio (for future re-embedding).

This decouples the enrollment CLI from any specific embedding model — the
pipeline worker is the single source of truth for which encoder is active.

Usage
-----
Enroll from microphone (10 seconds):
    python enroll.py --name Brian --room office --duration 10

Enroll from WAV file:
    python enroll.py --name Sarah --wav path/to/sarah.wav

List all enrolled speakers:
    python enroll.py --list

Show unenrolled provisional voices (from live audio, auto-created by pipeline):
    python enroll.py --unknown
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import warnings
import wave
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

# webrtcvad (a resemblyzer transitive dependency) uses pkg_resources which is
# deprecated in Python 3.13+ setuptools.  Suppress before any import fires.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
    module="webrtcvad",
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_AUDIO_S = 3.0  # minimum recording duration for a reliable embedding


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    _configure_logging(args.log_level if hasattr(args, "log_level") else "INFO")
    asyncio.run(_dispatch(args))


async def _dispatch(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    import os
    load_dotenv()

    pipeline_url = os.getenv("PIPELINE_URL", "http://localhost:8001")
    memory_mcp_url = os.getenv("MEMORY_MCP_URL", "http://memory-mcp:8900")
    memory_mcp_token = os.getenv("MEMORY_MCP_TOKEN", "")
    device_index = int(os.getenv("DEVICE_INDEX") or 0)

    mcp_headers: dict[str, str] = {}
    if memory_mcp_token:
        mcp_headers["Authorization"] = f"Bearer {memory_mcp_token}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        if args.command == "enroll":
            await cmd_enroll(client, args, device_index, pipeline_url)
        elif args.command == "list":
            async with httpx.AsyncClient(
                base_url=memory_mcp_url, timeout=30.0, headers=mcp_headers
            ) as mcp:
                await cmd_list(mcp)
        elif args.command == "unknown":
            async with httpx.AsyncClient(
                base_url=memory_mcp_url, timeout=30.0, headers=mcp_headers
            ) as mcp:
                await cmd_unknown(mcp)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_enroll(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    device_index: int,
    pipeline_url: str,
) -> None:
    """Enroll a speaker by shipping audio to the pipeline worker /enroll endpoint.

    The pipeline worker computes the embedding using its configured encoder
    (SPEAKER_ENCODER setting) and stores both the voiceprint and raw audio.

    Args:
        client: httpx client (used for pipeline worker requests).
        args: Parsed CLI arguments.
        device_index: Default sounddevice device index.
        pipeline_url: Pipeline worker base URL.
    """
    name: str = args.name
    room: Optional[str] = getattr(args, "room", None)
    duration: float = getattr(args, "duration", 10.0)
    wav_path: Optional[str] = getattr(args, "wav", None)

    print(f"Enrolling speaker: {name}")

    # --- Acquire audio ---
    if wav_path:
        print(f"  Loading audio from: {wav_path}")
        audio = load_wav(wav_path)
    else:
        print(f"  Recording {duration:.0f}s from microphone... (speak now)")
        audio = record_audio(duration_s=duration, device_index=device_index)
        print("  Recording complete.")

    if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
        print(
            f"ERROR: Audio too short ({len(audio)/SAMPLE_RATE:.1f}s). "
            f"Minimum {MIN_AUDIO_S:.0f}s required for a reliable voiceprint.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Encode audio as base64 WAV ---
    print("  Encoding audio...")
    audio_b64 = _audio_to_wav_b64(audio, SAMPLE_RATE)

    # --- Ship to pipeline worker /enroll ---
    print(f"  Sending to pipeline worker ({pipeline_url})...")
    try:
        response = await client.post(
            f"{pipeline_url}/enroll",
            json={
                "entity_name": name,
                "audio_b64": audio_b64,
                "sample_rate": SAMPLE_RATE,
                "room": room,
            },
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as exc:
        print(
            f"ERROR: Pipeline worker returned {exc.response.status_code}: "
            f"{exc.response.text[:200]}",
            file=sys.stderr,
        )
        sys.exit(1)
    except httpx.RequestError as exc:
        print(f"ERROR: Could not reach pipeline worker at {pipeline_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Confirm ---
    print(f"\nEnrolled: {result.get('entity_name', name)}")
    print(f"  Encoder  : {result.get('encoder', '?')}")
    print(f"  Room     : {room or 'not specified'}")
    print(f"  Samples  : {result.get('sample_count', '?')}")
    print(f"  Norm     : {result.get('embedding_norm', 0.0):.4f}")
    print(f"  Audio    : {'stored' if result.get('audio_stored') else 'not stored'}")


async def cmd_list(client: httpx.AsyncClient) -> None:
    """List all enrolled speakers."""
    try:
        response = await client.get("/entities")
        response.raise_for_status()
        entities = response.json().get("entities", [])
    except Exception as exc:
        print(f"ERROR: Failed to fetch entities: {exc}", file=sys.stderr)
        sys.exit(1)

    enrolled = [
        e for e in entities
        if e.get("type") == "person"
        and (e.get("meta") or {}).get("status") == "enrolled"
    ]

    if not enrolled:
        print("No enrolled speakers found.")
        return

    print(f"\nEnrolled speakers ({len(enrolled)}):")
    print(f"  {'Name':<20} {'Encoder':<12} {'Samples':>8}  {'First seen'}")
    print(f"  {'-'*20} {'-'*12} {'-'*8}  {'-'*24}")
    for e in enrolled:
        meta = e.get("meta") or {}
        name = e.get("name", "?")
        encoder = meta.get("speaker_encoder", "resemblyzer")
        samples = meta.get("voiceprint_samples", "?")
        first_seen = meta.get("first_seen", "unknown")
        print(f"  {name:<20} {encoder:<12} {str(samples):>8}  {first_seen}")


async def cmd_unknown(client: httpx.AsyncClient) -> None:
    """List unenrolled provisional voices."""
    try:
        response = await client.get("/voices/unknown", params={"limit": 50})
        response.raise_for_status()
        voices = response.json().get("result", [])
    except Exception as exc:
        print(f"ERROR: Failed to fetch unknown voices: {exc}", file=sys.stderr)
        sys.exit(1)

    if not voices:
        print("No unenrolled voices found.")
        return

    print(f"\nUnenrolled voices ({len(voices)}):")
    print(f"  {'Entity':<30} {'Detections':>10}  {'Sample transcript'}")
    print(f"  {'-'*30} {'-'*10}  {'-'*40}")
    for v in voices:
        name = v.get("entity_name", "?")
        count = v.get("detection_count", 0)
        transcript = (v.get("sample_transcript") or "")[:40]
        print(f"  {name:<30} {count:>10}  {transcript!r}")

    print(
        f"\nTo enroll: python enroll.py --name <Name> --wav <file.wav>"
        f"\nOr use --merge to attach to an existing speaker."
    )


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------


def record_audio(duration_s: float, device_index: int = 0) -> np.ndarray:
    """Record mono 16kHz audio from the specified device.

    Args:
        duration_s: Recording duration in seconds.
        device_index: sounddevice device index fallback.

    Returns:
        float32 mono array normalised to [-1, 1].

    Raises:
        ImportError: If sounddevice is not installed.
        RuntimeError: On recording failure.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for microphone recording. "
            "Install with: pip install sounddevice"
        ) from exc

    # Auto-detect ReSpeaker by name so the correct device is used regardless
    # of index changes across reboots.  Falls back to device_index if not found.
    try:
        device = sd.query_devices("reSpeaker", kind="input")["index"]
    except Exception:
        device = device_index

    # Query the device to determine how many input channels it supports.
    # The ReSpeaker XVF3800 requires 2 channels; standard mics use 1.
    try:
        dev_info = sd.query_devices(device, kind="input")
        channels = min(2, int(dev_info["max_input_channels"]))
    except Exception:
        channels = 1

    n_samples = int(duration_s * SAMPLE_RATE)
    recording = sd.rec(
        n_samples,
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    # Always return mono — take channel 0 (beamformed output on ReSpeaker)
    return recording[:, 0]


def load_wav(path: str) -> np.ndarray:
    """Load a WAV file and return a float32 mono 16kHz array.

    Accepts mono or stereo WAV files at any sample rate; converts to
    mono 16kHz internally.

    Args:
        path: Path to the WAV file.

    Returns:
        float32 mono array at SAMPLE_RATE Hz.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a WAV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(p), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        file_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Parse samples
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    # Convert to mono
    if n_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    elif n_channels > 2:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16kHz if needed
    if file_rate != SAMPLE_RATE:
        try:
            import scipy.signal
            n_out = int(len(samples) * SAMPLE_RATE / file_rate)
            samples = scipy.signal.resample(samples, n_out).astype(np.float32)
        except ImportError:
            raise ImportError(
                f"WAV file is {file_rate} Hz but scipy is required for resampling. "
                "Install with: pip install scipy  OR provide a 16kHz WAV file."
            )

    return samples


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


def _audio_to_wav_b64(audio: np.ndarray, sample_rate: int) -> str:
    """Encode a float32 mono array as a base64 16-bit PCM WAV string.

    Args:
        audio: float32 mono array normalised to [-1.0, 1.0].
        sample_rate: Audio sample rate.

    Returns:
        Base64-encoded WAV string compatible with pipeline worker /enroll.
    """
    import base64
    buf = io.BytesIO()
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enroll",
        description="Enroll speakers into the Heimdall voice identity system.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # enroll sub-command
    enroll_parser = sub.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("--name", required=True, help="Speaker name, e.g. 'Brian'")
    enroll_parser.add_argument(
        "--room", default=None, help="Room where enrollment was recorded (metadata only)"
    )
    enroll_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Microphone recording duration in seconds (default: 10)",
    )
    enroll_parser.add_argument(
        "--wav",
        default=None,
        metavar="PATH",
        help="Path to a WAV file to use instead of mic recording",
    )

    # list sub-command
    sub.add_parser("list", help="List all enrolled speakers")

    # unknown sub-command
    sub.add_parser("unknown", help="List unenrolled provisional voice entities")

    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


if __name__ == "__main__":
    main()
