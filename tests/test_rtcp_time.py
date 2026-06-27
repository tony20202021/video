"""
Тесты rtcp_time: парсинг RTCP SR пакетов (без живой камеры).
"""

from __future__ import annotations

import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.utils.rtcp_time import (
    RtcpCalibration,
    RtcpTimingReader,
    ntp64_to_datetime,
    ntp64_to_unix,
)

_NTP_EPOCH_OFFSET = 2208988800


# ── ntp64_to_unix / ntp64_to_datetime ─────────────────────────────────────────

def test_ntp64_to_unix_epoch():
    """NTP 0.0 → Unix 0 - 2208988800 (Jan 1, 1900 = -NTP_EPOCH_OFFSET)."""
    unix = ntp64_to_unix(0, 0)
    assert unix == -_NTP_EPOCH_OFFSET


def test_ntp64_to_unix_round_trip():
    """NTP соответствующий известной Unix-метке."""
    # Unix 0 (1970-01-01) = NTP 2208988800
    ntp_for_unix_epoch = _NTP_EPOCH_OFFSET
    unix = ntp64_to_unix(ntp_for_unix_epoch, 0)
    assert abs(unix) < 1e-9


def test_ntp64_to_unix_fraction():
    """Дробная часть NTP корректно конвертируется."""
    # LSW = 2^31 ≈ 0.5 секунды
    unix = ntp64_to_unix(_NTP_EPOCH_OFFSET, 2**31)
    assert abs(unix - 0.5) < 1e-6


def test_ntp64_to_datetime_utc():
    """Возвращает datetime с timezone.utc."""
    dt = ntp64_to_datetime(_NTP_EPOCH_OFFSET, 0)
    assert dt.tzinfo is not None
    assert dt.year == 1970
    assert dt.month == 1
    assert dt.day == 1
    assert dt.hour == 0


# ── RtcpCalibration ───────────────────────────────────────────────────────────

def _make_calib(ntp_unix: float = 1_000_000.0, rtp_ts: int = 90_000,
                clock_rate: int = 90_000) -> RtcpCalibration:
    return RtcpCalibration(ntp_unix=ntp_unix, rtp_ts=rtp_ts, clock_rate=clock_rate)


def test_calibration_rtp_to_datetime_same_ts():
    """rtp_ts == sr_rtp_ts → точно ntp_unix."""
    calib = _make_calib(ntp_unix=1_000_000.0, rtp_ts=90_000, clock_rate=90_000)
    dt = calib.rtp_to_datetime(90_000)
    expected = datetime.fromtimestamp(1_000_000.0, tz=timezone.utc)
    assert abs((dt - expected).total_seconds()) < 1e-6


def test_calibration_rtp_to_datetime_one_second_ahead():
    """rtp_ts += clock_rate → +1 секунда."""
    calib = _make_calib(ntp_unix=1_000_000.0, rtp_ts=90_000, clock_rate=90_000)
    dt = calib.rtp_to_datetime(90_000 + 90_000)
    expected = datetime.fromtimestamp(1_000_001.0, tz=timezone.utc)
    assert abs((dt - expected).total_seconds()) < 1e-6


def test_calibration_pts_to_datetime():
    """pts_sec → datetime с pts_first_sec offset."""
    calib = _make_calib(ntp_unix=1_000_000.0, rtp_ts=0, clock_rate=90_000)
    dt = calib.pts_to_datetime(pts_sec=2.0, pts_first_sec=1.0)
    expected = datetime.fromtimestamp(1_000_001.0, tz=timezone.utc)
    assert abs((dt - expected).total_seconds()) < 0.1


# ── RtcpTimingReader._parse_rtcp_sr ───────────────────────────────────────────

def _make_rtcp_sr_packet(ntp_msw: int, ntp_lsw: int, rtp_ts: int,
                          ssrc: int = 0x12345678) -> bytes:
    """Строит минимальный RTCP SR пакет."""
    # Header: V=2, P=0, RC=0, PT=200, length=6 (7 32-bit words - 1)
    header = struct.pack("!BBH", 0x80, 200, 6)
    body   = struct.pack("!IIIII", ssrc, ntp_msw, ntp_lsw, rtp_ts, 0)  # last=packet count
    return header + body


def _wrap_interleaved(channel: int, data: bytes) -> bytes:
    """Обёртывает данные в RTSP interleaved frame."""
    return struct.pack("!BBH", 0x24, channel, len(data)) + data


def test_parse_rtcp_sr_valid():
    """Корректный RTCP SR парсится в RtcpCalibration."""
    now_unix = time.time()
    ntp_msw = int(now_unix) + _NTP_EPOCH_OFFSET
    ntp_lsw = 0
    rtp_ts  = 180_000

    pkt = _make_rtcp_sr_packet(ntp_msw, ntp_lsw, rtp_ts)
    reader = RtcpTimingReader.__new__(RtcpTimingReader)
    reader._clock_rate = 90_000
    calib = reader._parse_rtcp_sr(pkt)

    assert calib is not None
    assert calib.rtp_ts == rtp_ts
    assert calib.clock_rate == 90_000
    assert abs(calib.ntp_unix - now_unix) < 1.0


def test_parse_rtcp_sr_wrong_pt():
    """PT != 200 → None."""
    pkt = _make_rtcp_sr_packet(_NTP_EPOCH_OFFSET + 1_000_000, 0, 0)
    bad_pkt = bytes([pkt[0], 201]) + pkt[2:]  # PT=201 (RR, not SR)
    reader = RtcpTimingReader.__new__(RtcpTimingReader)
    reader._clock_rate = 90_000
    assert reader._parse_rtcp_sr(bad_pkt) is None


def test_parse_rtcp_sr_too_short():
    """Слишком короткий пакет → None."""
    reader = RtcpTimingReader.__new__(RtcpTimingReader)
    reader._clock_rate = 90_000
    assert reader._parse_rtcp_sr(b"\x80\xc8\x00\x06" + b"\x00" * 10) is None


def test_scan_buffer_finds_sr():
    """_scan_buffer находит RTCP SR в интерливинговом буфере."""
    now_unix = time.time()
    ntp_msw = int(now_unix) + _NTP_EPOCH_OFFSET

    sr_pkt = _make_rtcp_sr_packet(ntp_msw, 0, 90_000)
    # Буфер: RTP на channel=0, RTCP SR на channel=1
    rtp_data = b"\x80\x60" + b"\x00" * 10  # fake RTP header
    buf = (
        _wrap_interleaved(0, rtp_data)
        + _wrap_interleaved(1, sr_pkt)
    )

    reader = RtcpTimingReader.__new__(RtcpTimingReader)
    reader._clock_rate = 90_000
    calib = reader._scan_buffer(buf)

    assert calib is not None
    assert calib.rtp_ts == 90_000
    assert abs(calib.ntp_unix - now_unix) < 1.0


def test_scan_buffer_no_sr():
    """Буфер только с RTP → None."""
    reader = RtcpTimingReader.__new__(RtcpTimingReader)
    reader._clock_rate = 90_000
    buf = _wrap_interleaved(0, b"\x80\x60" + b"\x00" * 10)
    assert reader._scan_buffer(buf) is None


def test_scan_buffer_incomplete_packet():
    """Неполный пакет в конце буфера не вызывает ошибку."""
    now_unix = time.time()
    ntp_msw = int(now_unix) + _NTP_EPOCH_OFFSET
    sr_pkt = _make_rtcp_sr_packet(ntp_msw, 0, 90_000)
    buf = _wrap_interleaved(1, sr_pkt)[:10]  # обрезаем

    reader = RtcpTimingReader.__new__(RtcpTimingReader)
    reader._clock_rate = 90_000
    result = reader._scan_buffer(buf)
    assert result is None  # неполный → None, не падает
