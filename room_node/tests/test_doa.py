"""Tests for room_node.doa — parse_doaangle_response (no USB hardware required)."""

import struct

import pytest

from room_node.doa import parse_doaangle_response


class TestParseDoaangleResponse:
    def _pack(self, value: int, status: int = 0) -> bytes:
        return struct.pack("<ii", value, status)

    def test_zero_degrees(self):
        assert parse_doaangle_response(self._pack(0)) == 0

    def test_359_degrees(self):
        assert parse_doaangle_response(self._pack(359)) == 359

    def test_180_degrees(self):
        assert parse_doaangle_response(self._pack(180)) == 180

    def test_wraps_360_to_0(self):
        # 360 mod 360 = 0
        assert parse_doaangle_response(self._pack(360)) == 0

    def test_wraps_720(self):
        assert parse_doaangle_response(self._pack(720)) == 0

    def test_wraps_361(self):
        assert parse_doaangle_response(self._pack(361)) == 1

    def test_negative_value_wraps(self):
        # -1 mod 360 = 359
        assert parse_doaangle_response(self._pack(-1)) == 359

    def test_status_byte_ignored(self):
        # Status int in second position should not affect result
        assert parse_doaangle_response(self._pack(90, status=42)) == 90

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="8 bytes"):
            parse_doaangle_response(b"\x00" * 4)

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="8 bytes"):
            parse_doaangle_response(b"\x00" * 12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_doaangle_response(b"")


class TestDOAReader:
    def test_unavailable_without_hardware(self):
        """DOAReader.available should be False when pyusb/device not present."""
        from unittest.mock import patch

        # Simulate pyusb not finding the device
        with patch("room_node.doa.DOAReader._connect"):
            from room_node.doa import DOAReader
            reader = DOAReader.__new__(DOAReader)
            reader._dev = None
            reader._available = False
            assert reader.available is False
            assert reader.read() is None
