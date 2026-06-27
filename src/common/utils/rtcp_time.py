"""Извлечение NTP-времени камеры из RTCP Sender Report пакетов.

Метод: устанавливает отдельное RTSP/TCP соединение, посылает SETUP+PLAY,
читает интерливингованный поток и извлекает RTCP SR (PT=200) пакеты.
Из SR берёт NTP timestamp (64-bit, секунды с 1900-01-01) и соответствующий
RTP timestamp — это даёт точку калибровки «RTP → wall clock».

После калибровки:
    cam_wall_time(rtp_ts) = ntp_epoch + (rtp_ts - sr_rtp_ts) / clock_rate

Использование:
    reader = RtcpTimingReader(rtsp_url)
    calib = reader.get_calibration(timeout=5.0)  # ждём первый RTCP SR
    if calib:
        ntp_sec, rtp_ts, clock_rate = calib
        cam_time = ntp_sec + (my_rtp_ts - rtp_ts) / clock_rate

Примечание: pyav не передаёт rtp_ts напрямую, поэтому для корреляции
используем PTS из пакетов (PTS ≈ (rtp_ts - rtp_start) / clock_rate).
Итоговая точность: ±0.1 сек (ограничена неопределённостью PTS).
"""

from __future__ import annotations

import re
import socket
import struct
import time
from datetime import datetime, timezone
from typing import Optional

# NTP epoch = 1900-01-01 00:00:00 UTC
_NTP_EPOCH_OFFSET = 2208988800  # seconds between 1900 and 1970


def ntp64_to_unix(ntp_high: int, ntp_low: int) -> float:
    """Конвертирует 64-bit NTP timestamp → Unix timestamp (float)."""
    ntp_sec = ntp_high - _NTP_EPOCH_OFFSET
    ntp_frac = ntp_low / (2 ** 32)
    return ntp_sec + ntp_frac


def ntp64_to_datetime(ntp_high: int, ntp_low: int) -> datetime:
    """Конвертирует 64-bit NTP → datetime (UTC)."""
    unix_ts = ntp64_to_unix(ntp_high, ntp_low)
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


class RtcpCalibration:
    """Точка калибровки RTP ↔ NTP из RTCP SR."""

    def __init__(self, ntp_unix: float, rtp_ts: int, clock_rate: int) -> None:
        self.ntp_unix = ntp_unix        # Unix timestamp когда был rtp_ts
        self.rtp_ts = rtp_ts            # RTP timestamp из SR
        self.clock_rate = clock_rate    # Частота RTP-часов (обычно 90000 для видео)
        self.received_at = time.monotonic()

    def rtp_to_datetime(self, rtp_ts: int) -> datetime:
        """Конвертирует RTP timestamp → datetime (UTC)."""
        delta = (rtp_ts - self.rtp_ts) / self.clock_rate
        unix_ts = self.ntp_unix + delta
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc)

    def pts_to_datetime(self, pts_sec: float, pts_first_sec: float) -> datetime:
        """
        Приближённая конвертация PTS секунды → datetime.
        pts_first_sec: PTS первого пакета (из pyav).
        """
        rtp_delta = (pts_sec - pts_first_sec) * self.clock_rate
        rtp_estimated = self.rtp_ts + int(rtp_delta)
        return self.rtp_to_datetime(rtp_estimated)


class RtcpTimingReader:
    """
    Минимальный RTSP/TCP клиент для получения RTCP SR пакетов.

    Открывает отдельное соединение с камерой (не мешает основному
    VideoCapture / StreamReader). Ждёт RTCP SR и возвращает точку
    калибровки RTP ↔ NTP.
    """

    def __init__(self, rtsp_url: str, clock_rate: int = 90000) -> None:
        self._url = rtsp_url
        self._clock_rate = clock_rate
        self._parsed = self._parse_url(rtsp_url)

    @staticmethod
    def _parse_url(url: str) -> dict:
        """Разбирает rtsp://user:pass@host:port/path."""
        m = re.match(
            r"rtsp://(?:([^:@/]+)(?::([^@/]*))?@)?([^:/]+)(?::(\d+))?(/.*)?",
            url,
        )
        if not m:
            raise ValueError(f"Не удалось разобрать RTSP URL: {url}")
        return {
            "user": m.group(1) or "",
            "password": m.group(2) or "",
            "host": m.group(3),
            "port": int(m.group(4) or 554),
            "path": m.group(5) or "/",
        }

    def get_calibration(self, timeout: float = 8.0) -> Optional[RtcpCalibration]:
        """
        Подключается к камере, ждёт RTCP SR, возвращает RtcpCalibration.
        Возвращает None при ошибке или таймауте.
        """
        try:
            return self._connect_and_read(timeout)
        except Exception:
            return None

    def _connect_and_read(self, timeout: float) -> Optional[RtcpCalibration]:
        p = self._parsed
        sock = socket.create_connection((p["host"], p["port"]), timeout=timeout)
        sock.settimeout(timeout)
        cseq = [1]

        def send(method: str, extra: str = "") -> str:
            auth = ""
            if p["user"]:
                import base64
                creds = base64.b64encode(f"{p['user']}:{p['password']}".encode()).decode()
                auth = f"Authorization: Basic {creds}\r\n"
            req = (
                f"{method} rtsp://{p['host']}:{p['port']}{p['path']} RTSP/1.0\r\n"
                f"CSeq: {cseq[0]}\r\n"
                f"{auth}"
                f"{extra}\r\n"
            )
            cseq[0] += 1
            sock.sendall(req.encode())
            return _read_response(sock)

        # OPTIONS → DESCRIBE → SETUP → PLAY
        send("OPTIONS")
        desc = send("DESCRIBE", "Accept: application/sdp\r\n")

        # Извлекаем track из SDP
        track = "trackID=0"
        for line in desc.split("\n"):
            if line.strip().startswith("a=control:"):
                track = line.strip().split(":")[-1].strip()
                break

        send(
            "SETUP",
            f"Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n"
            f"Track: rtsp://{p['host']}:{p['port']}{p['path']}/{track}\r\n",
        )
        send("PLAY", "Range: npt=0.000-\r\n")

        # Читаем интерливингованный поток до получения RTCP SR
        t_deadline = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < t_deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            calib = self._scan_buffer(buf)
            if calib is not None:
                sock.close()
                return calib
            # Оставляем хвост на случай разбитого пакета
            if len(buf) > 65536:
                buf = buf[-4096:]

        sock.close()
        return None

    def _scan_buffer(self, buf: bytes) -> Optional[RtcpCalibration]:
        """Ищет RTCP SR пакет в буфере TCP-потока."""
        i = 0
        while i < len(buf) - 4:
            if buf[i] != 0x24:  # '$' — признак интерливинга
                i += 1
                continue
            channel = buf[i + 1]
            length = struct.unpack_from("!H", buf, i + 2)[0]
            end = i + 4 + length
            if end > len(buf):
                break  # неполный пакет

            # channel=1 — RTCP; минимальный SR = 4 hdr + 4 ssrc + 8 ntp + 4 rtp = 20 байт
            if channel == 1 and length >= 20:
                data = buf[i + 4: end]
                calib = self._parse_rtcp_sr(data)
                if calib is not None:
                    return calib
            i = end

        return None

    def _parse_rtcp_sr(self, data: bytes) -> Optional[RtcpCalibration]:
        """Разбирает RTCP Sender Report (PT=200), возвращает калибровку."""
        if len(data) < 20:
            return None
        # Byte 0: V(2b)|P(1b)|RC(5b)  Byte 1: PT  Bytes 2-3: length in 32-bit words - 1
        pt = data[1]
        if pt != 200:  # SR = 200
            return None
        # Bytes 4-7: SSRC
        # Bytes 8-11: NTP MSW (seconds since 1900)
        # Bytes 12-15: NTP LSW (fraction)
        # Bytes 16-19: RTP timestamp
        if len(data) < 20:
            return None
        ntp_msw = struct.unpack_from("!I", data, 8)[0]
        ntp_lsw = struct.unpack_from("!I", data, 12)[0]
        rtp_ts  = struct.unpack_from("!I", data, 16)[0]

        ntp_unix = ntp64_to_unix(ntp_msw, ntp_lsw)
        # Проверяем разумность: должно быть близко к текущему времени (±1 год)
        now = time.time()
        if abs(ntp_unix - now) > 365 * 86400:
            return None

        return RtcpCalibration(
            ntp_unix=ntp_unix,
            rtp_ts=rtp_ts,
            clock_rate=self._clock_rate,
        )


def _read_response(sock: socket.socket) -> str:
    """Читает RTSP-ответ (заканчивается на пустую строку)."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode(errors="replace")
