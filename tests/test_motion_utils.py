"""
Тесты на common.utils.motion_utils — работают без камеры и без модели.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.utils.motion_utils import (
    drain_cap_buffer,
    ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff,
    prepare_gray,
    read_hi_save_frame,
    redact_url,
    skip_url,
    stem_from_var,
)


# ─── redact_url ───────────────────────────────────────────────────────────────

def test_redact_url_query_style():
    url = "rtsp://192.168.1.9:554/user=hxdx&password=secret123&channel=0"
    assert "secret123" not in redact_url(url)
    assert "password=***" in redact_url(url)


def test_redact_url_basic_auth_style():
    url = "rtsp://admin:p@ssw0rd@192.168.1.9:554/stream"
    out = redact_url(url)
    assert "p@ssw0rd" not in out
    assert "***:***@" in out


def test_redact_url_empty():
    assert redact_url("") == ""


# ─── skip_url ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("", True),
    ("# comment", True),
    ("rtsp://<external_ip>:554/stream", True),
    ("http://192.168.1.1/stream", True),
    ("rtsp://192.168.1.9:554/stream", False),
])
def test_skip_url(url, expected):
    assert skip_url(url) == expected


# ─── stem_from_var ────────────────────────────────────────────────────────────

def test_stem_from_var():
    assert stem_from_var("CAM_01_9_U_URL") == "cam_01_9_u"
    assert stem_from_var("CAM_02_URL") == "cam_02"


# ─── ffmpeg_capture_options ───────────────────────────────────────────────────

def test_ffmpeg_options_no_tcp():
    opts = ffmpeg_capture_options(use_tcp=False, stimeout_us=5_000_000)
    assert "stimeout;5000000" in opts
    assert "rtsp_transport" not in opts


def test_ffmpeg_options_tcp():
    opts = ffmpeg_capture_options(use_tcp=True, stimeout_us=8_000_000)
    assert opts.startswith("rtsp_transport;tcp")
    assert "stimeout;8000000" in opts


# ─── prepare_gray ─────────────────────────────────────────────────────────────

def test_prepare_gray_output_shape():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray = prepare_gray(frame)
    assert gray.ndim == 2
    assert gray.shape == (480, 640)


def test_prepare_gray_preserves_size():
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    gray = prepare_gray(frame)
    assert gray.shape == (240, 320)


# ─── mean_abs_diff ────────────────────────────────────────────────────────────

def test_mean_abs_diff_identical():
    a = np.ones((100, 100), dtype=np.uint8) * 128
    assert mean_abs_diff(a, a) == 0.0


def test_mean_abs_diff_all_different():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 255, dtype=np.uint8)
    assert mean_abs_diff(a, b) == 255.0


def test_mean_abs_diff_float():
    a = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    b = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    assert mean_abs_diff(a, b) == pytest.approx(25.0)


# ─── frame_decode_plausible ───────────────────────────────────────────────────

def test_frame_plausible_black_rejected():
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    assert not frame_decode_plausible(black, min_laplacian_var=10.0, min_gray_std=2.0)


def test_frame_plausible_noise_accepted():
    noisy = np.random.randint(30, 220, (480, 640, 3), dtype=np.uint8)
    assert frame_decode_plausible(noisy, min_laplacian_var=1.0, min_gray_std=1.0)


def test_frame_plausible_none_rejected():
    assert not frame_decode_plausible(None, min_laplacian_var=10.0, min_gray_std=2.0)


# ─── drain_cap_buffer + read_hi_save_frame (заглушки) ────────────────────────

class _FakeCap:
    """Минимальная заглушка cv2.VideoCapture для тестов без камеры."""

    def __init__(self, frames: list[np.ndarray | None]):
        self._frames = iter(frames)
        self._grab_calls = 0

    def set(self, prop_id, value):
        pass

    def grab(self) -> bool:
        self._grab_calls += 1
        try:
            f = next(self._frames)
            return f is not None
        except StopIteration:
            return False

    def read(self):
        try:
            f = next(self._frames)
            return (f is not None, f)
        except StopIteration:
            return (False, None)

    def release(self):
        pass


def _good_frame() -> np.ndarray:
    return np.random.randint(30, 220, (360, 640, 3), dtype=np.uint8)


def test_drain_cap_buffer_calls_grab():
    frames = [_good_frame() for _ in range(5)]
    cap = _FakeCap(frames)
    drain_cap_buffer(cap, max_drain=10)
    assert cap._grab_calls > 0


def test_read_hi_save_frame_returns_frame():
    good = _good_frame()
    # None → grab() возвращает False → drain останавливается
    # good → read() берёт его и возвращает
    cap = _FakeCap([None, good])
    result = read_hi_save_frame(
        cap, max_extra_reads=3, min_laplacian_var=1.0, min_gray_std=1.0
    )
    assert result is not None
    assert result.shape == good.shape


def test_read_hi_save_frame_returns_none_on_empty():
    cap = _FakeCap([])
    result = read_hi_save_frame(
        cap, max_extra_reads=3, min_laplacian_var=10.0, min_gray_std=2.0
    )
    assert result is None
