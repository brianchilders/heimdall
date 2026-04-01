"""Hardware tests for the Logitech C920 HD Webcam.

All tests are marked @pytest.mark.hardware and are skipped in CI.
They require the C920 to be physically attached via USB and accessible.

Prerequisites
-------------
- udev rule grants access (see docs/REQUIREMENTS.md)
- User is in the ``video`` and ``audio`` groups
- opencv-python-headless installed (``uv pip install opencv-python-headless``)
- sounddevice installed for audio device check
- /dev/video0 must not be held open by another process (OBS, Cheese, etc.)

Run
---
    pytest -m hardware room_node/tests/test_hardware_c920.py -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGITECH_VID = "046d"
C920_PID = "08e5"
C920_NAME_FRAGMENT = "c920"       # in sounddevice device name
C920_VIDEO_DEVICE = "/dev/video0" # primary V4L2 node
MIN_WIDTH = 640
MIN_HEIGHT = 480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_c920_audio_device() -> tuple[int, dict] | None:
    """Return (index, device_info) for the C920 USB audio input."""
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) >= 1:
            if C920_NAME_FRAGMENT in dev["name"].lower():
                return i, dev
    return None


# ---------------------------------------------------------------------------
# USB presence
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_c920_usb_device_visible():
    """lsusb shows Logitech C920 at VID=046d PID=08e5."""
    result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, "lsusb failed — is usbutils installed?"
    vid_pid = f"{LOGITECH_VID}:{C920_PID}"
    assert vid_pid in result.stdout.lower(), (
        f"Logitech C920 ({vid_pid}) not found in lsusb output.\n"
        f"Output:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# V4L2 device node
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_c920_video_device_exists():
    """The primary V4L2 device node /dev/video0 exists."""
    assert Path(C920_VIDEO_DEVICE).exists(), (
        f"{C920_VIDEO_DEVICE} does not exist. "
        "Check that the C920 is connected and the udev rule is loaded."
    )


@pytest.mark.hardware
def test_c920_video_device_readable():
    """The /dev/video0 node is readable by the current user."""
    p = Path(C920_VIDEO_DEVICE)
    if not p.exists():
        pytest.skip(f"{C920_VIDEO_DEVICE} not present")
    assert p.stat().st_mode & 0o004, (
        f"{C920_VIDEO_DEVICE} is not world-readable. "
        "Check udev rule and user group membership (video group)."
    )


@pytest.mark.hardware
def test_c920_v4l2_lists_device():
    """v4l2-ctl --list-devices output mentions C920."""
    result = subprocess.run(
        ["v4l2-ctl", "--list-devices"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, (
        f"v4l2-ctl --list-devices failed: {result.stderr}"
    )
    assert C920_NAME_FRAGMENT in result.stdout.lower(), (
        f"C920 not found in v4l2-ctl --list-devices output:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# OpenCV frame capture
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_c920_opencv_opens_device():
    """cv2.VideoCapture(0) opens without error."""
    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    cap = cv2.VideoCapture(0)
    try:
        assert cap.isOpened(), (
            "VideoCapture(0) failed to open. "
            "Check that /dev/video0 is not held by another process."
        )
    finally:
        cap.release()


@pytest.mark.hardware
def test_c920_captures_valid_frame():
    """VideoCapture.read() returns True and a non-empty frame."""
    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("VideoCapture(0) did not open")
    try:
        ret, frame = cap.read()
        assert ret, "VideoCapture.read() returned False — no frame received"
        assert frame is not None
        assert frame.size > 0
    finally:
        cap.release()


@pytest.mark.hardware
def test_c920_frame_is_bgr():
    """Captured frame has 3 channels (BGR colour)."""
    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("VideoCapture(0) did not open")
    try:
        ret, frame = cap.read()
        if not ret:
            pytest.skip("No frame received from VideoCapture")
        assert len(frame.shape) == 3, f"Expected 3-dim frame, got shape {frame.shape}"
        assert frame.shape[2] == 3, f"Expected 3 channels (BGR), got {frame.shape[2]}"
    finally:
        cap.release()


@pytest.mark.hardware
def test_c920_minimum_resolution():
    """Default capture resolution is at least 640×480."""
    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("VideoCapture(0) did not open")
    try:
        ret, frame = cap.read()
        if not ret:
            pytest.skip("No frame received from VideoCapture")
        h, w = frame.shape[:2]
        assert w >= MIN_WIDTH, f"Frame width {w} < {MIN_WIDTH}"
        assert h >= MIN_HEIGHT, f"Frame height {h} < {MIN_HEIGHT}"
        print(f"\n  Captured frame: {w}×{h} px")
    finally:
        cap.release()


@pytest.mark.hardware
def test_c920_frame_dtype_uint8():
    """Frame pixel values are uint8 (standard OpenCV format)."""
    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("VideoCapture(0) did not open")
    try:
        ret, frame = cap.read()
        if not ret:
            pytest.skip("No frame received from VideoCapture")
        assert frame.dtype == np.uint8, f"Expected uint8 frame, got {frame.dtype}"
    finally:
        cap.release()


@pytest.mark.hardware
def test_c920_frame_not_all_zeros():
    """Frame is not entirely black (lens cap off, device actually streaming)."""
    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("VideoCapture(0) did not open")
    try:
        ret, frame = cap.read()
        if not ret:
            pytest.skip("No frame received from VideoCapture")
        assert frame.max() > 0, (
            "Frame is entirely black — check that the camera lens is not covered "
            "and that the device is streaming real data."
        )
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# C920 audio device
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_c920_audio_device_enumerated():
    """sounddevice enumerates the C920 USB audio input."""
    found = _find_c920_audio_device()
    assert found is not None, (
        "No audio device with 'c920' in name found by sounddevice.\n"
        "All input devices:\n"
        + "\n".join(
            f"  [{i}] {d['name']} ({d['max_input_channels']} ch in)"
            for i, d in enumerate(sd.query_devices())
            if d.get("max_input_channels", 0) >= 1
        )
    )


@pytest.mark.hardware
def test_c920_audio_has_stereo_input():
    """C920 USB audio device has at least 2 input channels (stereo mic)."""
    found = _find_c920_audio_device()
    if found is None:
        pytest.skip("C920 audio device not found by sounddevice")
    _, dev = found
    assert dev["max_input_channels"] >= 2, (
        f"Expected >= 2 input channels (stereo), got {dev['max_input_channels']}"
    )
