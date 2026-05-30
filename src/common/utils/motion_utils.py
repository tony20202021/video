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
    return float(g.std()) >= min_gray_std


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


def drain_cap_buffer(cap: cv2.VideoCapture, max_drain: int = 32) -> None:
    """Дренирует накопившиеся кадры FFMPEG-буфера RTSP (FIFO).

    При длительном простое HI-потока буфер накапливает устаревшие кадры.
    Без дренажа cap.read() вернёт кадр из прошлого вместо свежего.
    """
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(max_drain):
        if not cap.grab():
            break


def read_hi_save_frame(
    cap: cv2.VideoCapture,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    """Дренирует буфер HI-потока и возвращает свежий годный кадр."""
    drain_cap_buffer(cap)
    ok, f = cap.read()
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
