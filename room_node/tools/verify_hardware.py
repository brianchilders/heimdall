#!/usr/bin/env python3
"""
ReSpeaker USB Mic Array hardware verification utility.

Runs a series of checks and reports PASS / FAIL / SKIP for each.  Designed to
be run on the Raspberry Pi before starting the room node for the first time, or
whenever you want to confirm the mic array is functioning correctly.

Checks
------
1. USB detection    — find a Seeed Studio (VID=0x2886) mic array device
2. Audio device     — confirm sounddevice enumerates the array
3. Signal capture   — record 1 s and verify RMS > silence threshold
4. DOA readback     — read the DOAANGLE parameter via USB control transfer
5. Channel count    — confirm ≥ 4 input channels available

Usage
-----
    python room_node/tools/verify_hardware.py

Optional flags
--------------
    --duration N     Capture duration in seconds (default: 2)
    --list-audio     Just list all audio devices and exit
    --device N       Force a specific sounddevice device index
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Known Seeed Studio USB mic array identifiers
# ---------------------------------------------------------------------------

_SEEED_VENDOR_ID = 0x2886

# (PID, description) pairs — add new variants here as they are discovered
_KNOWN_PRODUCTS: dict[int, str] = {
    0x0018: "ReSpeaker USB Mic Array v2.0 (XVF-3000)",
    0x0019: "ReSpeaker USB Mic Array v2.0 rev.B",
    0x0020: "ReSpeaker Lite (XVF-3800)",
    # XVF-3800 variants — PID may differ by firmware; we try all and report
    0x002B: "ReSpeaker USB 4-Mic Array (XVF-3800)",
}

# USB control transfer constants (XMOS VocalFusion vendor protocol)
_CTRL_IN = 0xC0       # bmRequestType: IN | VENDOR | DEVICE
_CTRL_REQUEST = 0x00  # bRequest
_CTRL_INDEX = 0x1C    # wIndex
_CTRL_LENGTH = 8      # wLength (two LE 32-bit ints)
_PARAM_DIRECTION = 21 # DOAANGLE / DIRECTION parameter index

# RMS threshold below which we declare "silence" (likely not picking up signal)
_SILENCE_THRESHOLD = 0.001  # normalised float32 scale

# ---------------------------------------------------------------------------
# Colour helpers (gracefully degraded when stdout is not a tty)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _USE_COLOUR else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _USE_COLOUR else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _USE_COLOUR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOUR else s


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"

_results: list[tuple[str, str, str]] = []  # (status, check_name, detail)


def _record(status: str, name: str, detail: str) -> None:
    _results.append((status, name, detail))
    tag = {
        PASS: _green(f"[{PASS}]"),
        FAIL: _red(f"[{FAIL}]"),
        SKIP: _yellow(f"[{SKIP}]"),
        WARN: _yellow(f"[{WARN}]"),
    }[status]
    print(f"  {tag}  {name}: {detail}")


# ---------------------------------------------------------------------------
# Check 1: USB detection
# ---------------------------------------------------------------------------


def check_usb() -> Optional[tuple[object, int]]:
    """Find a Seeed Studio mic array via USB and report details.

    Returns (device, product_id) on success, None on failure.
    """
    print(_bold("\n[1] USB device detection"))
    try:
        import usb.core
        import usb.util
    except ImportError:
        _record(SKIP, "USB scan", "pyusb not installed — run: pip install pyusb")
        return None

    # Broad scan: find any Seeed Studio device
    devices = list(usb.core.find(idVendor=_SEEED_VENDOR_ID, find_all=True) or [])

    if not devices:
        _record(FAIL, "USB scan", f"No Seeed Studio device found (VID=0x{_SEEED_VENDOR_ID:04x})")
        print("    Tip: run `lsusb` and verify the array appears in the list.")
        return None

    _record(PASS, "USB scan", f"Found {len(devices)} Seeed device(s)")

    # Look for known mic array PIDs
    mic_dev = None
    mic_pid = 0
    for dev in devices:
        pid = dev.idProduct
        desc = _KNOWN_products_desc(pid)
        fw = _get_firmware_version(dev)
        fw_str = f" (firmware {fw})" if fw else ""
        _record(
            PASS if desc else WARN,
            f"  PID 0x{pid:04x}",
            f"{desc or 'Unknown Seeed device — may be mic array'}{fw_str}",
        )
        if mic_dev is None and (desc or True):  # use first found
            mic_dev = dev
            mic_pid = pid

    if mic_dev is None:
        _record(FAIL, "Mic array", "No recognised mic array PID — unknown device")
        return None

    return mic_dev, mic_pid


def _known_products_desc(pid: int) -> Optional[str]:
    return _KNOWN_PRODUCTS.get(pid)


# Alias for typo fix
_known_products_desc_orig = _KNOWN_PRODUCTS.get


def _get_firmware_version(dev: object) -> Optional[str]:
    """Attempt to read firmware string from device (best-effort)."""
    try:
        import usb.util
        langid = dev.langids[0] if dev.langids else 0x0409
        raw = usb.util.get_string(dev, dev.iProduct, langid)
        return raw if raw else None
    except Exception:
        return None


def _KNOWN_products_desc(pid: int) -> Optional[str]:
    return _KNOWN_PRODUCTS.get(pid)


# ---------------------------------------------------------------------------
# Check 2: Audio device enumeration
# ---------------------------------------------------------------------------


def check_audio_device(force_index: Optional[int] = None) -> Optional[int]:
    """Find the mic array in sounddevice and return its device index."""
    print(_bold("\n[2] Audio device enumeration"))
    try:
        import sounddevice as sd
    except ImportError:
        _record(SKIP, "Audio devices", "sounddevice not installed — run: pip install sounddevice")
        return None

    devices = sd.query_devices()
    candidates: list[tuple[int, dict]] = []

    for i, dev in enumerate(devices):
        name = dev.get("name", "")
        if dev.get("max_input_channels", 0) < 1:
            continue
        if "respeaker" in name.lower() or "seeed" in name.lower() or "xvf" in name.lower():
            candidates.append((i, dev))

    if force_index is not None:
        dev = devices[force_index]
        _record(
            PASS,
            "Device (forced)",
            f"[{force_index}] {dev['name']}  "
            f"({dev['max_input_channels']} ch in, {dev['default_samplerate']} Hz)",
        )
        return force_index

    if not candidates:
        _record(WARN, "Audio scan", "No device with 'ReSpeaker/Seeed/XVF' in name found")
        # List all input devices to help the user
        print("    All input devices:")
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) >= 1:
                print(
                    f"      [{i}] {dev['name']}  "
                    f"({dev['max_input_channels']} ch, {dev['default_samplerate']} Hz)"
                )
        return None

    idx, dev = candidates[0]
    _record(
        PASS,
        "Audio device",
        f"[{idx}] {dev['name']}  "
        f"({dev['max_input_channels']} ch in, {dev['default_samplerate']} Hz)",
    )
    return idx


# ---------------------------------------------------------------------------
# Check 3: Channel count
# ---------------------------------------------------------------------------


def check_channel_count(device_index: Optional[int]) -> None:
    """Verify the device has ≥ 4 input channels."""
    print(_bold("\n[3] Input channel count"))
    if device_index is None:
        _record(SKIP, "Channel count", "No device selected")
        return
    try:
        import sounddevice as sd
        dev = sd.query_devices(device_index)
        ch = dev.get("max_input_channels", 0)
        if ch >= 4:
            _record(PASS, "Channels", f"{ch} input channels available")
        else:
            _record(FAIL, "Channels", f"Only {ch} input channels — expected ≥ 4 for mic array")
    except Exception as exc:
        _record(FAIL, "Channel count", f"Error: {exc}")


# ---------------------------------------------------------------------------
# Check 4: Audio capture + signal level
# ---------------------------------------------------------------------------


def check_audio_capture(device_index: Optional[int], duration: float = 2.0) -> None:
    """Record audio and verify signal RMS is above silence threshold."""
    print(_bold(f"\n[4] Audio capture ({duration:.0f} s — please speak or make noise)"))
    if device_index is None:
        _record(SKIP, "Audio capture", "No device selected")
        return

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        _record(SKIP, "Audio capture", f"Missing dependency: {exc}")
        return

    try:
        dev = sd.query_devices(device_index)
        rate = int(dev.get("default_samplerate", 16000))
        channels = min(dev.get("max_input_channels", 1), 4)

        print(f"    Recording {duration:.0f} s at {rate} Hz, {channels} ch …", flush=True)
        recording = sd.rec(
            int(duration * rate),
            samplerate=rate,
            channels=channels,
            dtype="float32",
            device=device_index,
        )
        sd.wait()

        rms_per_channel = [
            math.sqrt(float(np.mean(recording[:, c] ** 2))) for c in range(channels)
        ]
        max_rms = max(rms_per_channel)
        ch_str = "  ".join(f"ch{i}: {v:.5f}" for i, v in enumerate(rms_per_channel))

        if max_rms > _SILENCE_THRESHOLD:
            _record(PASS, "Signal level", f"RMS OK — {ch_str}")
        else:
            _record(WARN, "Signal level", f"Signal very low (silence?) — {ch_str}")
            print("    Make sure you spoke or made noise during capture.")

        # Clipping check
        max_abs = float(np.abs(recording).max())
        if max_abs > 0.98:
            _record(WARN, "Clipping", f"Peak value {max_abs:.3f} — mic may be too close / gain too high")
        else:
            _record(PASS, "No clipping", f"Peak {max_abs:.3f}")

    except Exception as exc:
        _record(FAIL, "Audio capture", f"Recording failed: {exc}")


# ---------------------------------------------------------------------------
# Check 5: DOA readback
# ---------------------------------------------------------------------------


def check_doa(usb_result: Optional[tuple[object, int]]) -> None:
    """Read the DOAANGLE parameter via USB vendor control transfer."""
    print(_bold("\n[5] DOA (Direction of Arrival) readback"))
    if usb_result is None:
        _record(SKIP, "DOA", "USB device not available")
        return

    dev, pid = usb_result

    try:
        response = dev.ctrl_transfer(
            _CTRL_IN,
            _CTRL_REQUEST,
            _PARAM_DIRECTION,
            _CTRL_INDEX,
            _CTRL_LENGTH,
        )
        value, status = struct.unpack("<ii", bytes(response))
        angle = int(value) % 360
        _record(PASS, "DOA angle", f"{angle}° (raw value={value}, status={status})")

        # Quick sanity: take 3 readings and check they're in [0, 359]
        readings = [angle]
        for _ in range(2):
            time.sleep(0.1)
            resp2 = dev.ctrl_transfer(
                _CTRL_IN, _CTRL_REQUEST, _PARAM_DIRECTION, _CTRL_INDEX, _CTRL_LENGTH
            )
            v2, _ = struct.unpack("<ii", bytes(resp2))
            readings.append(int(v2) % 360)
        _record(PASS, "DOA stability", f"3 readings: {readings}")

    except Exception as exc:
        _record(FAIL, "DOA readback", f"USB control transfer failed: {exc}")
        print(
            "    Tip: try running with sudo, or add a udev rule:\n"
            "    echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"2886\", MODE=\"0666\"' "
            "| sudo tee /etc/udev/rules.d/99-respeaker.rules && sudo udevadm control --reload"
        )


# ---------------------------------------------------------------------------
# List-only mode
# ---------------------------------------------------------------------------


def list_audio_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed — run: pip install sounddevice")
        return
    devices = sd.query_devices()
    print("All audio devices:")
    for i, dev in enumerate(devices):
        direction = []
        if dev.get("max_input_channels", 0) > 0:
            direction.append(f"in×{dev['max_input_channels']}")
        if dev.get("max_output_channels", 0) > 0:
            direction.append(f"out×{dev['max_output_channels']}")
        marker = "  <-- ReSpeaker?" if (
            "respeaker" in dev["name"].lower()
            or "seeed" in dev["name"].lower()
            or "xvf" in dev["name"].lower()
        ) else ""
        print(f"  [{i:2d}] {dev['name']:<40} {', '.join(direction)}{marker}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> int:
    """Print final summary table and return exit code (0=all pass, 1=any fail)."""
    print(_bold("\n" + "=" * 60))
    print(_bold("  SUMMARY"))
    print("=" * 60)

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    passes = [r for r in _results if r[0] == PASS]

    print(f"  {_green(f'{len(passes)} passed')}  "
          f"{_yellow(f'{len(warns)} warnings')}  "
          f"{_red(f'{len(fails)} failed')}")

    if fails:
        print(_red("\nFailed checks:"))
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")

    if warns:
        print(_yellow("\nWarnings:"))
        for _, name, detail in warns:
            print(f"  - {name}: {detail}")

    print("=" * 60)

    if fails:
        print(_red("\nResult: HARDWARE NOT READY — fix failures before starting room node."))
        return 1

    if warns:
        print(_yellow("\nResult: HARDWARE READY with warnings — review above."))
        return 0

    print(_green("\nResult: ALL CHECKS PASSED — hardware is ready."))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ReSpeaker USB Mic Array hardware verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--duration", type=float, default=2.0,
        help="Audio capture duration in seconds (default: 2)",
    )
    parser.add_argument(
        "--list-audio", action="store_true",
        help="List all audio devices and exit",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Force a specific sounddevice device index",
    )
    args = parser.parse_args()

    if args.list_audio:
        list_audio_devices()
        return 0

    print(_bold("ReSpeaker USB Mic Array — Hardware Verification"))
    print(f"Platform: {sys.platform}  Python {sys.version.split()[0]}")

    usb_result = check_usb()
    device_index = check_audio_device(force_index=args.device)
    check_channel_count(device_index)
    check_audio_capture(device_index, duration=args.duration)
    check_doa(usb_result)

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
