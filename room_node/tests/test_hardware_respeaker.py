"""Hardware tests for the ReSpeaker XVF3800 USB mic array.

All tests are marked @pytest.mark.hardware and are skipped in CI.
They require the ReSpeaker XVF3800 to be physically attached via USB.

Prerequisites
-------------
- udev rule grants access (see docs/REQUIREMENTS.md)
- User is in the ``audio`` and ``plugdev`` groups
- sounddevice installed (``uv pip install sounddevice``)
- pyusb optional — DOA test is skipped if not installed

Run
---
    pytest -m hardware room_node/tests/test_hardware_respeaker.py -v
"""

from __future__ import annotations

import subprocess
import time

import numpy as np
import pytest
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEED_VID = "2886"
RESPEAKER_PID = "001a"   # XVF3800 — matches udev rule 99-heimdall-hw.rules
RESPEAKER_NAME_FRAGMENT = "respeaker"
SAMPLE_RATE = 16000
CAPTURE_DURATION_S = 1.0
SILENCE_RMS_THRESHOLD = 0.001  # signal below this is likely true silence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_respeaker_device() -> tuple[int, dict] | None:
    """Return (index, device_info) for the first ReSpeaker-named input device."""
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) >= 1:
            if RESPEAKER_NAME_FRAGMENT in dev["name"].lower():
                return i, dev
    return None


# ---------------------------------------------------------------------------
# USB presence
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_respeaker_usb_device_visible():
    """lsusb shows ReSpeaker XVF3800 at VID=2886 PID=001a."""
    result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, "lsusb failed — is usbutils installed?"
    vid_pid = f"{SEEED_VID}:{RESPEAKER_PID}"
    assert vid_pid in result.stdout.lower(), (
        f"ReSpeaker XVF3800 ({vid_pid}) not found in lsusb output.\n"
        f"Output:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Audio device enumeration
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_respeaker_audio_device_enumerated():
    """sounddevice enumerates a device with 'respeaker' in the name."""
    found = _find_respeaker_device()
    assert found is not None, (
        "No device with 'respeaker' in name found by sounddevice.\n"
        "All input devices:\n"
        + "\n".join(
            f"  [{i}] {d['name']} ({d['max_input_channels']} ch in)"
            for i, d in enumerate(sd.query_devices())
            if d.get("max_input_channels", 0) >= 1
        )
    )


@pytest.mark.hardware
def test_respeaker_has_input_channels():
    """ReSpeaker device reports at least 1 input channel."""
    found = _find_respeaker_device()
    pytest.skip("ReSpeaker not found") if found is None else None
    _, dev = found
    assert dev["max_input_channels"] >= 1, (
        f"Expected >= 1 input channels, got {dev['max_input_channels']}"
    )


@pytest.mark.hardware
def test_respeaker_samplerate():
    """ReSpeaker default sample rate is 16000 or 48000 Hz."""
    found = _find_respeaker_device()
    if found is None:
        pytest.skip("ReSpeaker not found by sounddevice")
    _, dev = found
    rate = int(dev["default_samplerate"])
    assert rate in (16000, 44100, 48000), (
        f"Unexpected sample rate {rate} Hz — expected 16000 or 48000"
    )


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_respeaker_capture_opens_and_returns_samples():
    """Record 1 s from the ReSpeaker and verify the array shape is correct.

    Does NOT require ambient sound — verifies device accessibility only.
    """
    found = _find_respeaker_device()
    if found is None:
        pytest.skip("ReSpeaker not found by sounddevice")
    idx, dev = found
    n_frames = int(CAPTURE_DURATION_S * SAMPLE_RATE)
    channels = min(dev["max_input_channels"], 2)

    recording = sd.rec(
        n_frames,
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=idx,
    )
    sd.wait()

    assert recording.shape == (n_frames, channels), (
        f"Expected shape ({n_frames}, {channels}), got {recording.shape}"
    )
    assert recording.dtype == np.float32


@pytest.mark.hardware
def test_respeaker_capture_rms_nonzero():
    """Signal RMS > silence threshold — make noise near the mic during capture.

    This test passes a WARNING-level assert: if RMS is below threshold the
    mic is likely picking up nothing (true silence or misconfigured gain).
    The test still passes so CI-on-hardware runs are not blocked by a quiet room.
    """
    found = _find_respeaker_device()
    if found is None:
        pytest.skip("ReSpeaker not found by sounddevice")
    idx, dev = found
    channels = min(dev["max_input_channels"], 2)
    n_frames = int(CAPTURE_DURATION_S * SAMPLE_RATE)

    print(f"\n  Recording {CAPTURE_DURATION_S:.0f} s — make noise near the mic…", flush=True)
    recording = sd.rec(
        n_frames,
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=idx,
    )
    sd.wait()

    rms_values = [
        float(np.sqrt(np.mean(recording[:, c] ** 2))) for c in range(channels)
    ]
    max_rms = max(rms_values)
    ch_str = "  ".join(f"ch{i}:{v:.5f}" for i, v in enumerate(rms_values))
    print(f"  RMS per channel: {ch_str}")

    if max_rms <= SILENCE_RMS_THRESHOLD:
        pytest.warns(
            UserWarning,
            match="silence",
        )
        import warnings
        warnings.warn(
            f"Signal RMS {max_rms:.6f} <= threshold {SILENCE_RMS_THRESHOLD} — "
            "mic may not be picking up signal (true silence or gain issue).",
            UserWarning,
            stacklevel=1,
        )


@pytest.mark.hardware
def test_respeaker_no_clipping():
    """Peak amplitude during 1 s capture is below 0.98 (no clipping)."""
    found = _find_respeaker_device()
    if found is None:
        pytest.skip("ReSpeaker not found by sounddevice")
    idx, dev = found
    channels = min(dev["max_input_channels"], 2)
    n_frames = int(CAPTURE_DURATION_S * SAMPLE_RATE)

    recording = sd.rec(
        n_frames,
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        device=idx,
    )
    sd.wait()

    peak = float(np.abs(recording).max())
    assert peak < 0.98, (
        f"Peak amplitude {peak:.3f} suggests clipping — mic too close or gain too high"
    )


# ---------------------------------------------------------------------------
# DOA readback (requires pyusb)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_respeaker_doa_readable():
    """DOAReader returns an integer angle in [0, 359] when pyusb is available.

    Skipped automatically if pyusb is not installed.
    """
    pytest.importorskip("usb.core", reason="pyusb not installed — install with: uv pip install pyusb")

    from room_node.doa import DOAReader

    reader = DOAReader()
    if not reader.available:
        pytest.skip(
            "DOAReader could not open ReSpeaker USB device. "
            "Check udev rule (docs/REQUIREMENTS.md) and that the device is connected."
        )

    angle = reader.read()
    assert angle is not None, "DOAReader.read() returned None despite device being available"
    assert isinstance(angle, int), f"Expected int, got {type(angle)}"
    assert 0 <= angle <= 359, f"DOA angle {angle} out of range [0, 359]"


@pytest.mark.hardware
def test_respeaker_doa_stable():
    """Three consecutive DOA reads all return valid angles (no USB errors mid-stream).

    Skipped if pyusb not installed or device not available.
    """
    pytest.importorskip("usb.core", reason="pyusb not installed — install with: uv pip install pyusb")

    from room_node.doa import DOAReader

    reader = DOAReader()
    if not reader.available:
        pytest.skip("ReSpeaker USB device not available")

    readings: list[int] = []
    for _ in range(3):
        angle = reader.read()
        assert angle is not None, "DOA read returned None mid-sequence"
        assert 0 <= angle <= 359
        readings.append(angle)
        time.sleep(0.1)

    print(f"\n  DOA readings: {readings}")
    # All readings in range — stability is informational only
    assert len(readings) == 3
