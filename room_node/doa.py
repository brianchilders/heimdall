"""
Direction of Arrival (DOA) extraction from the ReSpeaker USB Mic Array.

Supports both XVF-3000 and XVF-3800 variants via the same XMOS VocalFusion
vendor USB control transfer protocol.  The DOAANGLE (parameter 21) is
available on both chips with identical wire format.

USB protocol
------------
Vendor ID : 0x2886  (Seeed Studio — all variants)
Product ID: see _KNOWN_PRODUCT_IDS below

The DOAANGLE (parameter index 21) is read via a USB control transfer:

    bmRequestType = 0xC0  (IN | VENDOR | DEVICE)
    bRequest      = 0x00
    wValue        = PARAM_INDEX (21 for DOAANGLE)
    wIndex        = 0x1c
    wLength       = 8

Response is two 32-bit little-endian integers; the first is the parameter
value (DOA angle, 0–359 degrees).

Hardware fallback
-----------------
If the ReSpeaker is not present returns None so the caller can continue
without DOA data.  All tests use the mock path (parse_doaangle_response).
"""

from __future__ import annotations

import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# Seeed Studio vendor ID (all ReSpeaker products)
_VENDOR_ID = 0x2886

# Known ReSpeaker mic array product IDs — matched in order; first found wins.
_KNOWN_PRODUCT_IDS: tuple[tuple[int, str], ...] = (
    (0x0018, "ReSpeaker USB Mic Array v2.0 (XVF-3000)"),
    (0x0019, "ReSpeaker USB Mic Array v2.0 rev.B (XVF-3000)"),
    (0x001A, "reSpeaker XVF3800 4-Mic Array"),
    (0x0020, "ReSpeaker Lite (XVF-3800)"),
    (0x002B, "ReSpeaker USB 4-Mic Array (XVF-3800)"),
)

# USB control transfer parameters for DOAANGLE (param index 21)
_PARAM_DOAANGLE = 21
_CTRL_IN = 0xC0       # IN | VENDOR | DEVICE
_CTRL_REQUEST = 0x00
_CTRL_INDEX = 0x1C
_CTRL_LENGTH = 8


class DOAReader:
    """Reads Direction of Arrival from the ReSpeaker USB Mic Array v2.0.

    Opens the USB device on construction.  Call read() to get the current
    angle.  If the device is unavailable, read() returns None and logs a
    warning rather than raising — the pipeline continues without DOA data.

    Usage::

        reader = DOAReader()
        angle = reader.read()   # int 0–359 or None
    """

    def __init__(self) -> None:
        self._dev: Optional[object] = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        """Attempt to open the ReSpeaker USB device (any supported PID)."""
        try:
            import usb.core
            dev = None
            for pid, description in _KNOWN_PRODUCT_IDS:
                dev = usb.core.find(idVendor=_VENDOR_ID, idProduct=pid)
                if dev is not None:
                    logger.info(
                        "ReSpeaker USB device opened: %s (VID=0x%04x PID=0x%04x)",
                        description, _VENDOR_ID, pid,
                    )
                    break
            if dev is None:
                pids_str = ", ".join(f"0x{p:04x}" for p, _ in _KNOWN_PRODUCT_IDS)
                logger.warning(
                    "ReSpeaker USB device not found (VID=0x%04x, tried PIDs: %s). "
                    "DOA will be unavailable.",
                    _VENDOR_ID, pids_str,
                )
                return
            self._dev = dev
            self._available = True
        except ImportError:
            logger.warning("pyusb not installed — DOA unavailable. Install with: pip install pyusb")
        except Exception as exc:
            logger.warning("Failed to open ReSpeaker USB: %s — DOA unavailable", exc)

    @property
    def available(self) -> bool:
        """True if the ReSpeaker USB device was found and opened."""
        return self._available

    def read(self) -> Optional[int]:
        """Read the current DOA angle from the ReSpeaker hardware.

        Returns:
            Integer angle in [0, 359] degrees, or None if the device is
            unavailable or the read fails.
        """
        if not self._available or self._dev is None:
            return None
        try:
            return _read_doaangle(self._dev)
        except Exception as exc:
            logger.debug("DOA read failed: %s", exc)
            return None


def _read_doaangle(dev: object) -> int:
    """Issue the USB control transfer and parse the DOAANGLE response.

    Args:
        dev: A usb.core.Device instance.

    Returns:
        DOA angle in [0, 359] degrees.

    Raises:
        Exception: On USB communication failure.
    """
    response = dev.ctrl_transfer(
        _CTRL_IN,        # bmRequestType
        _CTRL_REQUEST,   # bRequest
        _PARAM_DOAANGLE, # wValue  — parameter index
        _CTRL_INDEX,     # wIndex
        _CTRL_LENGTH,    # wLength
    )
    # Response: two little-endian 32-bit integers; first is the value
    value, _status = struct.unpack("<ii", bytes(response))
    return int(value) % 360  # clamp to [0, 359] in case of firmware quirk


def parse_doaangle_response(raw_bytes: bytes) -> int:
    """Parse a raw USB control transfer response into a DOA angle.

    Exposed as a standalone function for unit testing without USB hardware.

    Args:
        raw_bytes: 8-byte response from the USB control transfer.

    Returns:
        DOA angle in [0, 359] degrees.

    Raises:
        ValueError: If raw_bytes is not 8 bytes long.
    """
    if len(raw_bytes) != 8:
        raise ValueError(f"Expected 8 bytes, got {len(raw_bytes)}")
    value, _status = struct.unpack("<ii", raw_bytes)
    return int(value) % 360
