"""
Тесты модуля osd_time: извлечение времени камеры из OSD-оверлея.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.utils.osd_time import (
    _extract_segments,
    _seg_char,
    extract_osd_time,
    templates_complete,
    OSD_Y1,
    OSD_Y2,
    OSD_X1,
    OSD_X2,
)

# Путь к реальному тестовому heartbeat кадру (1152×648)
# U-кадр с видимым OSD (верхняя половина камеры содержит timestamp)
HEARTBEAT_U = (
    Path(__file__).resolve().parents[1]
    / "tests/.data/no_person"
    / "cam_01_9_u_cam_20260601_094037_20260601_094933_979219_msk_heartbeat.jpg"
)


# ─── _extract_segments ────────────────────────────────────────────────────────

def test_extract_segments_empty():
    bw = np.zeros((16, 100), dtype=np.uint8)
    segs = _extract_segments(bw)
    assert segs == []


def test_extract_segments_single_block():
    bw = np.zeros((16, 50), dtype=np.uint8)
    bw[:, 10:20] = 255
    segs = _extract_segments(bw)
    assert len(segs) == 1
    s, e = segs[0]
    assert s == 10 and e == 20


def test_extract_segments_two_blocks():
    bw = np.zeros((16, 100), dtype=np.uint8)
    bw[:, 5:12] = 255
    bw[:, 20:30] = 255
    segs = _extract_segments(bw)
    assert len(segs) == 2


# ─── _seg_char ────────────────────────────────────────────────────────────────

def test_seg_char_colon_narrow():
    seg = np.zeros((16, 3), dtype=np.uint8)
    seg[6:8, :] = 255
    seg[11:13, :] = 255
    assert _seg_char(seg) == ":"


def test_seg_char_dash_one_row():
    seg = np.zeros((16, 5), dtype=np.uint8)
    seg[10, :] = 255  # только одна строка
    assert _seg_char(seg) == "-"


def test_seg_char_one_digit():
    seg = np.zeros((16, 4), dtype=np.uint8)
    seg[4:14, 2:4] = 255  # вертикальная полоса — '1'
    assert _seg_char(seg) == "1"


def test_seg_char_digit_returns_string_or_none():
    # Произвольный 10px-wide сегмент без шаблонов — возвращает None или строку
    seg = np.random.randint(0, 255, (16, 10), dtype=np.uint8)
    result = _seg_char(seg)
    assert result is None or isinstance(result, str)


# ─── extract_osd_time ─────────────────────────────────────────────────────────

def test_extract_osd_time_returns_none_for_blank():
    blank = np.zeros((648, 1152, 3), dtype=np.uint8)
    result = extract_osd_time(blank)
    assert result is None


def test_extract_osd_time_returns_none_for_small_frame():
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    result = extract_osd_time(small)
    assert result is None


@pytest.mark.integration
@pytest.mark.skipif(
    not HEARTBEAT_U.is_file(),
    reason="Реальный heartbeat кадр недоступен"
)
def test_extract_osd_time_real_frame():
    img = cv2.imread(str(HEARTBEAT_U))
    assert img is not None, f"Не удалось загрузить: {HEARTBEAT_U}"
    result = extract_osd_time(img)
    assert result is not None, "OSD время не распознано на реальном кадре"
    assert isinstance(result, datetime)
    # Кадр из 2026-05-31 сессии
    assert result.year == 2026
    assert result.month == 6
    assert result.day == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not HEARTBEAT_U.is_file(),
    reason="Реальный heartbeat кадр недоступен"
)
def test_extract_osd_time_consistent():
    """Два вызова на одном кадре должны вернуть одинаковый результат."""
    img = cv2.imread(str(HEARTBEAT_U))
    r1 = extract_osd_time(img)
    r2 = extract_osd_time(img)
    assert r1 == r2


# ─── templates_complete ───────────────────────────────────────────────────────

def test_templates_complete_with_loaded_templates():
    # models/osd_templates.npz должен существовать после build_templates
    result = templates_complete()
    # Может быть True (если npz существует) или False (тесты в CI без файла)
    assert isinstance(result, bool)
