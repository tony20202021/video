"""Общие утилиты детекции движения: фильтрация URL, метрика diff, проверка кадра."""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

# Обе формы передачи учётных данных в URL: query (?...&password=...) и basic-auth (user:pass@host)
_RE_PASSWORD_QUERY = re.compile(r"(?i)(password=)[^&/\s'\"]*")
_RE_USERINFO = re.compile(r"(://)[^:@/\s]+:[^@/\s]+@")


def redact_url(text: str) -> str:
    """Маскирует пароль в строке (URL или сообщение об ошибке): и password=…, и user:pass@host."""
    if not text:
        return text
    text = _RE_PASSWORD_QUERY.sub(r"\1***", text)
    text = _RE_USERINFO.sub(r"\1***:***@", text)
    return text


def skip_url(url: str) -> bool:
    u = (url or "").strip()
    return not u or u.startswith("#") or ("<" in u and ">" in u) or not u.lower().startswith("rtsp://")


def ffmpeg_capture_options(*, use_tcp: bool, stimeout_us: int) -> str:
    parts: list[str] = [f"stimeout;{stimeout_us}"]
    if use_tcp:
        parts.insert(0, "rtsp_transport;tcp")
    return "|".join(parts)


def stem_from_var(var_name: str) -> str:
    return var_name.replace("_URL", "").lower()


def prepare_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w != width:
        scale = width / float(w)
        nh = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def mean_abs_diff(prev: np.ndarray, cur: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(prev, cur)))


def frame_decode_plausible(
    bgr: np.ndarray,
    *,
    min_laplacian_var: float,
    min_gray_std: float,
) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(cv2.Laplacian(g, cv2.CV_64F).var()) < min_laplacian_var:
        return False
    if float(g.std()) < min_gray_std:
        return False

    h, w = g.shape[:2]

    # Горизонтальные полосы: ловим верхние/нижние серые бэнды
    if h >= 16:
        strip_h = max(1, h // 4)
        for i in range(4):
            strip = g[i * strip_h: min(h, (i + 1) * strip_h), :]
            if strip.size > 0 and float(strip.std()) < min_gray_std:
                return False

    # HEVC-специфичный артефакт: недекодированные блоки заполняются
    # RGB(128,128,128) — нейтральным серым. Если >35% пикселей кадра
    # близки к этому значению (|R-G|<12, |G-B|<12, Y∈[108,148]),
    # кадр частично не декодирован (ловит левую/правую/любую зону).
    if bgr.ndim == 3:
        b_ch = bgr[:, :, 0].astype(np.int16)
        g_ch = bgr[:, :, 1].astype(np.int16)
        r_ch = bgr[:, :, 2].astype(np.int16)
        neutral = (
            (np.abs(r_ch - g_ch) < 12) &
            (np.abs(g_ch - b_ch) < 12) &
            (g_ch > 108) &
            (g_ch < 148)
        )
        if float(neutral.mean()) > 0.35:
            return False

    return True


def read_first_plausible_frame(
    cap: cv2.VideoCapture,
    *,
    max_attempts: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    for _ in range(max_attempts):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0 and frame_decode_plausible(
            f, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std
        ):
            return f
    return None


def pick_frame_to_save(
    cap: cv2.VideoCapture,
    first_bgr: np.ndarray,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    if frame_decode_plausible(first_bgr, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
        return first_bgr
    for _ in range(max_extra_reads):
        ok, f = cap.read()
        if not ok or f is None or f.size == 0:
            continue
        if frame_decode_plausible(f, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
            return f
    return None


def drain_cap_buffer(cap: cv2.VideoCapture, max_drain: int = 512, stale_ms: float = 40.0) -> None:
    """Дренирует накопившиеся кадры FFMPEG-буфера RTSP (FIFO).

    Сливает кадры пока grab() быстрый (< stale_ms) — это означает что кадры
    уже в буфере. Когда grab() начинает блокироваться (ждёт сеть) — буфер пуст,
    останавливаемся. Следующий cap.read() в read_hi_save_frame повторит попытку
    с паузой если буфер оказался полностью опустошён.

    Прежнее max_drain=8 (~0.67с на 12fps) было недостаточным: после долгого
    простоя HI-потока буфер накапливал 2-3+ секунды кадров и первый кадр после
    drain был всё равно устаревшим (задача 18b).
    """
    import time as _t
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(max_drain):
        t0 = _t.monotonic()
        if not cap.grab():
            break
        if (_t.monotonic() - t0) * 1000 > stale_ms:
            # grab заблокировался — буфер исчерпан, дальше ждём сеть
            break


def read_hi_save_frame(
    cap: cv2.VideoCapture,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    """Дренирует буфер HI-потока и возвращает свежий годный кадр.

    После drain буфер может оказаться пустым (кадр ещё не пришёл от камеры
    на 12fps). Делаем до 3 попыток с паузой ~100мс между ними.
    """
    import time as _time
    drain_cap_buffer(cap)
    ok, f = None, None
    for _ in range(3):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            break
        _time.sleep(0.1)  # ждём следующий кадр (12fps ≈ 83мс)
    if not ok or f is None or f.size == 0:
        return None
    return pick_frame_to_save(
        cap, f,
        max_extra_reads=max_extra_reads,
        min_laplacian_var=min_laplacian_var,
        min_gray_std=min_gray_std,
    )


def open_cap(
    url: str,
    *,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(open_timeout_ms))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        return None
    return cap
