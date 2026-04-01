# Heimdall — System Requirements

System-level setup needed before running any heimdall service or test.
These steps are performed once on the development machine and on each Pi node.

---

## Operating System

- Ubuntu 22.04 / 24.04 LTS (dev machine, blackmagic.lan)
- Raspberry Pi OS Lite 64-bit Bookworm (Pi 5 room nodes)
- Python 3.11 or 3.12

---

## Linux User Groups

The user running heimdall services must belong to these groups:

| Group | Grants access to |
|---|---|
| `audio` | ALSA / sounddevice microphone capture |
| `video` | V4L2 camera devices (`/dev/video*`) |
| `plugdev` | USB HID raw devices (`/dev/hidraw*`) |

Add your user to all three at once:

```bash
sudo usermod -aG audio,video,plugdev $USER
```

Log out and back in (or `newgrp audio`) for group changes to take effect.

Verify:
```bash
groups   # should list audio, video, plugdev
```

---

## System Packages

```bash
sudo apt install -y \
  libusb-1.0-0 \        # runtime for pyusb (USB control transfers)
  libasound2-dev \      # ALSA headers; required to build sounddevice from source
  v4l-utils \           # v4l2-ctl — camera inspection and capability query
  usbutils \            # lsusb — USB device enumeration
  alsa-utils            # aplay, arecord — ALSA playback/capture CLI tools
```

Verify utilities are present:
```bash
which v4l2-ctl lsusb aplay arecord udevadm
```

---

## udev Rules

The file `/etc/udev/rules.d/99-heimdall-hw.rules` must exist with the
following content to grant non-root access to the attached USB devices:

```udev
# Seeed ReSpeaker XVF3800 (VID=0x2886 PID=0x001a)
SUBSYSTEM=="usb",    ATTRS{idVendor}=="2886", ATTRS{idProduct}=="001a", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="001a", MODE="0666"

# Logitech C920 HD Webcam (VID=0x046d PID=0x08e5)
SUBSYSTEM=="usb",        ATTRS{idVendor}=="046d", ATTRS{idProduct}=="08e5", MODE="0666"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="08e5", MODE="0666"
```

Apply without rebooting:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Python Packages (via uv)

All Python dependencies are managed in the project virtual environment via `uv`.

### Currently installed in `.venv` (dev machine)

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.4 | Array math throughout |
| `sounddevice` | 0.5.5 | Microphone capture via ALSA |
| `hid` | 1.0.9 | USB HID device access (hidraw) |
| `opencv-python-headless` | 4.13.0.92 | Camera frame capture (C920) |
| `respx` | 0.22.0 | HTTP mock for pytest |

### Required on full room nodes (Pi 5) — not yet installed on dev machine

Install when deploying to Pi or running hardware integration tests end-to-end:

```bash
uv pip install pyusb          # USB control transfers (DOA readback)
uv pip install resemblyzer    # 256-dim voiceprint embeddings
uv pip install faster-whisper # CPU Whisper fallback
uv pip install silero-vad     # Voice activity detection
# torch + torchaudio: platform-specific, install via Pi wheel index
```

### Installing missing dev dependencies

```bash
# From the repo root (activates the workspace venv)
uv pip install pyusb respx
```

---

## Hardware Devices (attached to dev machine)

| Device | VID:PID | Interface | Notes |
|---|---|---|---|
| ReSpeaker XVF3800 4-Mic Array | 2886:001a | ALSA hw:0,0 / hidraw | 2-ch output via ALSA; DOA via USB control transfer |
| Logitech C920 HD Webcam | 046d:08e5 | /dev/video0, ALSA hw:2,0 | Video + stereo USB audio |

Verify both are connected:
```bash
lsusb | grep -E "2886|046d"
```

Expected output:
```
Bus 003 Device 002: ID 2886:001a Seeed Technology Co., Ltd. reSpeaker XVF3800 4-Mic Array
Bus 003 Device 003: ID 046d:08e5 Logitech, Inc. C920 PRO HD Webcam
```

---

## Running Hardware Tests

Hardware tests are marked `@pytest.mark.hardware` and excluded from CI.

```bash
# All hardware tests (requires devices attached)
pytest -m hardware room_node/tests/ -v

# ReSpeaker only
pytest -m hardware room_node/tests/test_hardware_respeaker.py -v

# Logitech C920 only
pytest -m hardware room_node/tests/test_hardware_c920.py -v

# Unit tests only (no hardware needed — CI safe)
pytest -m "not hardware and not integration" room_node/tests/ -v
```

### Notes on individual tests

- **DOA readback** (`test_respeaker_doa_readable`) — skipped automatically if `pyusb`
  is not installed. Install with `uv pip install pyusb` to enable.
- **Signal level** (`test_respeaker_capture_rms_nonzero`) — requires ambient noise or
  speaking near the mic during the 1-second capture window. Silence will cause a
  warning but not a failure.
- **C920 frame capture** — `/dev/video0` must not be held open by another process
  (e.g. OBS, Cheese). Close other camera apps first.
