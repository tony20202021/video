"""Извлечение времени камеры из OSD-оверлея на кадре.

Камера iCSee записывает метку времени «YYYY-MM-DD HH:MM:SS» в правый верхний
угол HI-кадра (1152×648). Распознавание — шаблонное сравнение по бинаризованным
символам, без сторонних OCR-библиотек.

Шаблоны цифр хранятся в models/osd_templates.npz и строятся автоматически при
первом запуске (build_templates) либо загружаются из файла.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Геометрия OSD на этой модели камеры (2304×1296 → hi_scale=0.5 → 1152×648) ─
_OSD_Y1 = OSD_Y1 = 78              # строки OSD-полосы
_OSD_Y2 = OSD_Y2 = 98
_OSD_X1 = OSD_X1 = 920             # столбцы timestamp (правая часть)
_OSD_X2 = OSD_X2 = 1142
_THRESH = 180                       # порог бинаризации (белый текст)
_TMPL_W, _TMPL_H = 10, 16          # размер нормализованного шаблона символа

# Ожидаемое число сегментов (18, не 19 — пробел между датой и временем пустой)
_EXPECTED_SEGS = 18
# Ширины, по которым однозначно определяется символ без шаблона
_WIDTH_COLON = 3          # ':'
_WIDTH_DASH_MAX = 5       # '-' (ширина 4-5)
_WIDTH_ONE = 4            # '1' (очень узкая)

_TEMPLATES: dict[str, np.ndarray] = {}
_LOCK = threading.Lock()
_DEFAULT_NPZ = Path(__file__).resolve().parents[3] / ".models" / "osd_templates.npz"


# LOW-stream frame geometry (640×360, upper-half crop of full stream)
LOW_OSD_Y1 = 45
LOW_OSD_Y2 = 58
LOW_OSD_X1 = 360
LOW_OSD_X2 = 636
LOW_THRESH  = 220

_LOW_TEMPLATES: dict[str, np.ndarray] = {}
_LOW_LOCK = threading.Lock()
_DEFAULT_LOW_NPZ = Path(__file__).resolve().parents[3] / ".models" / "osd_templates_low.npz"


# ── Загрузка / сохранение шаблонов ───────────────────────────────────────────

def _load_templates(npz_path: Path = _DEFAULT_NPZ) -> bool:
    global _TEMPLATES
    if not npz_path.is_file():
        return False
    try:
        data = np.load(str(npz_path))
        with _LOCK:
            _TEMPLATES = {chr(int(k.replace("char_", ""))): data[k] for k in data.files}
        return bool(_TEMPLATES)
    except Exception:
        return False


def save_templates(npz_path: Path = _DEFAULT_NPZ) -> None:
    with _LOCK:
        arrays = dict(_TEMPLATES)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **{f"char_{ord(c)}": arr for c, arr in arrays.items()})


def add_templates(new_tmpls: dict[str, np.ndarray], npz_path: Path = _DEFAULT_NPZ) -> None:
    """Добавить новые шаблоны и сохранить."""
    with _LOCK:
        _TEMPLATES.update(new_tmpls)
    save_templates(npz_path)


def templates_complete() -> bool:
    with _LOCK:
        return all(d in _TEMPLATES for d in "0123456789")


# ── Геометрия сегментов ───────────────────────────────────────────────────────

_MIN_SEG_WIDTH = 3  # минимальная ширина сегмента: ':' = 3px, '-' = 5px, '1' = 4px


def _extract_segments(bw: np.ndarray) -> list[tuple[int, int]]:
    """Группы ненулевых столбцов → список (x_start, x_end). Шум < _MIN_SEG_WIDTH игнорируется."""
    col_sums = bw.sum(axis=0)
    in_char = False
    segs: list[tuple[int, int]] = []
    start = 0
    for c, s in enumerate(col_sums):
        if s > 0 and not in_char:
            start = c
            in_char = True
        elif s == 0 and in_char:
            if c - start >= _MIN_SEG_WIDTH:
                segs.append((start, c))
            in_char = False
    if in_char and len(col_sums) - start >= _MIN_SEG_WIDTH:
        segs.append((start, len(col_sums)))
    return segs


def _seg_char(seg_bw: np.ndarray) -> Optional[str]:
    """Определить символ сегмента по ширине или шаблонному сравнению."""
    w = seg_bw.shape[1]
    if w <= _WIDTH_COLON:
        return ":"
    if w <= _WIDTH_DASH_MAX:
        # '-' имеет пиксели только в 1-2 строках; '1' — в 10+ строках
        nonzero_rows = int((seg_bw.sum(axis=1) > 0).sum())
        return "-" if nonzero_rows <= 2 else "1"
    # Для ширин 9-10: шаблонное сравнение
    with _LOCK:
        tmpls = dict(_TEMPLATES)
    if not tmpls:
        return None
    resized = cv2.resize(seg_bw, (_TMPL_W, _TMPL_H), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    best_char, best_score = None, -1.0
    for char, tmpl in tmpls.items():
        if char in ("-", ":"):
            continue
        # Нормализованная кросс-корреляция
        num = float(np.sum(resized * tmpl))
        denom = float(np.sqrt(np.sum(resized ** 2) * np.sum(tmpl ** 2)) + 1e-8)
        score = num / denom
        if score > best_score:
            best_score = score
            best_char = char
    return best_char


# ── Основная функция ──────────────────────────────────────────────────────────

def extract_osd_time(frame: np.ndarray) -> Optional[datetime]:
    """
    Извлечь дату-время из OSD-оверлея HI-кадра (1152×648).

    Возвращает datetime в local timezone или None при неудаче.
    """
    if not _TEMPLATES:
        _load_templates()
    if not _TEMPLATES:
        return None

    h, w = frame.shape[:2]
    if h < _OSD_Y2 or w < _OSD_X2:
        return None

    crop = frame[_OSD_Y1:_OSD_Y2, _OSD_X1:_OSD_X2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, _THRESH, 255, cv2.THRESH_BINARY)

    segs = _extract_segments(bw)
    if len(segs) != _EXPECTED_SEGS:
        return None

    chars: list[str] = []
    for s, e in segs:
        c = _seg_char(bw[:, s:e])
        if c is None:
            return None
        chars.append(c)

    # Маппинг 18 сегментов → строка без пробела
    # segs: 0-3=YYYY, 4=-, 5-6=MM, 7=-, 8-9=DD, 10-11=HH, 12=:, 13-14=MM, 15=:, 16-17=SS
    ts_str = "".join(chars[:10]) + " " + "".join(chars[10:])
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


# ── Построение шаблонов из известного кадра ──────────────────────────────────

def build_templates_from_frame(
    frame: np.ndarray,
    known_ts: str,
    npz_path: Path = _DEFAULT_NPZ,
) -> dict[str, np.ndarray]:
    """
    Построить шаблоны символов из кадра с известной меткой времени.

    known_ts: строка вида '2026-05-31 14:49:31'
    Возвращает словарь char→template и сохраняет в npz_path.
    """
    h, w = frame.shape[:2]
    if h < _OSD_Y2 or w < _OSD_X2:
        raise ValueError(f"Кадр слишком мал: {w}×{h}, нужен ≥{_OSD_X2}×{_OSD_Y2}")

    crop = frame[_OSD_Y1:_OSD_Y2, _OSD_X1:_OSD_X2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, _THRESH, 255, cv2.THRESH_BINARY)
    segs = _extract_segments(bw)

    ts_chars = list(known_ts.replace(" ", ""))
    if len(segs) != len(ts_chars):
        raise ValueError(f"Ожидалось {len(ts_chars)} сегментов, найдено {len(segs)}")

    new_tmpls: dict[str, np.ndarray] = {}
    for (s, e), char in zip(segs, ts_chars):
        if char in new_tmpls:
            continue
        seg_img = bw[:, s:e]
        resized = cv2.resize(seg_img, (_TMPL_W, _TMPL_H), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        new_tmpls[char] = resized

    add_templates(new_tmpls, npz_path)
    return new_tmpls


# ── Авто-пополнение шаблонов из потока кадров ─────────────────────────────────

class TemplateCollector:
    """
    Накапливает шаблоны из входящих кадров (self-calibrating режим).
    Использует изменение времени между кадрами для идентификации новых цифр.
    """

    def __init__(self, npz_path: Path = _DEFAULT_NPZ) -> None:
        self._npz_path = npz_path
        self._pending: dict[str, np.ndarray] = {}  # char → accumulated template
        self._last_ts_str: Optional[str] = None
        _load_templates(npz_path)

    def feed(self, frame: np.ndarray) -> Optional[datetime]:
        """
        Передать кадр. Возвращает распознанное время или None.
        Побочный эффект: пополняет шаблоны для неизвестных цифр.
        """
        h, w = frame.shape[:2]
        if h < _OSD_Y2 or w < _OSD_X2:
            return None

        crop = frame[_OSD_Y1:_OSD_Y2, _OSD_X1:_OSD_X2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, _THRESH, 255, cv2.THRESH_BINARY)
        segs = _extract_segments(bw)
        if len(segs) != _EXPECTED_SEGS:
            return None

        # Пробуем распознать с текущими шаблонами
        result = extract_osd_time(frame)

        # Собираем шаблоны из позиций с известной шириной
        # (разделители всегда можно определить по ширине)
        new = {}
        for idx, (s, e) in enumerate(segs):
            w_seg = e - s
            if w_seg <= _WIDTH_COLON:
                char = ":"
            elif w_seg <= _WIDTH_DASH_MAX:
                char = "-"
            else:
                continue  # цифры без шаблона пока пропускаем
            with _LOCK:
                if char not in _TEMPLATES:
                    seg_img = bw[:, s:e]
                    resized = cv2.resize(seg_img, (_TMPL_W, _TMPL_H), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
                    new[char] = resized

        if new:
            add_templates(new, self._npz_path)

        return result


# ── LOW-stream шаблоны ───────────────────────────────────────────────────────

def _load_low_templates(npz_path: Path = _DEFAULT_LOW_NPZ) -> bool:
    global _LOW_TEMPLATES
    if not npz_path.is_file():
        return False
    try:
        data = np.load(str(npz_path))
        with _LOW_LOCK:
            _LOW_TEMPLATES = {chr(int(k.replace("char_", ""))): data[k] for k in data.files}
        return bool(_LOW_TEMPLATES)
    except Exception:
        return False


def _save_low_templates(npz_path: Path = _DEFAULT_LOW_NPZ) -> None:
    with _LOW_LOCK:
        arrays = dict(_LOW_TEMPLATES)
    if not arrays:
        return
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **{f"char_{ord(c)}": arr for c, arr in arrays.items()})


def _seg_char_low(seg_bw: np.ndarray) -> Optional[str]:
    """Распознаёт символ OSD в LOW-кадре через шаблоны.

    Включает ':' в сравнение — на LOW-кадре '1' и ':' одинаковой ширины (3px),
    поэтому нельзя различить их по ширине.
    """
    if seg_bw.shape[1] <= 1:
        return None  # шум
    with _LOW_LOCK:
        tmpls = dict(_LOW_TEMPLATES)
    if not tmpls:
        return None
    resized = cv2.resize(seg_bw, (_TMPL_W, _TMPL_H),
                         interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    best_char, best_score = None, -1.0
    for char, tmpl in tmpls.items():
        num   = float(np.sum(resized * tmpl))
        denom = float(np.sqrt(np.sum(resized ** 2) * np.sum(tmpl ** 2)) + 1e-8)
        score = num / denom
        if score > best_score:
            best_score, best_char = score, char
    return best_char if best_score > 0.3 else None


def extract_osd_time_low(frame: np.ndarray) -> Optional[datetime]:
    """Извлекает OSD-время из LOW-кадра (640×360) U-камеры."""
    if not _LOW_TEMPLATES:
        _load_low_templates()
    if not _LOW_TEMPLATES:
        return None
    h, w = frame.shape[:2]
    if h < LOW_OSD_Y2 or w < LOW_OSD_X2:
        return None
    crop = frame[LOW_OSD_Y1:LOW_OSD_Y2, LOW_OSD_X1:LOW_OSD_X2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, LOW_THRESH, 255, cv2.THRESH_BINARY)
    segs = _extract_segments(bw)
    if len(segs) != _EXPECTED_SEGS:
        return None
    chars: list[str] = []
    for s, e in segs:
        c = _seg_char_low(bw[:, s:e])
        if c is None:
            return None
        chars.append(c)
    ts_str = "".join(chars[:10]) + " " + "".join(chars[10:])
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def build_low_templates_from_image(
    frame: np.ndarray,
    known_ts: str,
    safe_up_to: int = 16,
    npz_path: Path = _DEFAULT_LOW_NPZ,
) -> dict[str, np.ndarray]:
    """Строит LOW-шаблоны из кадра с известной меткой времени (YYYY-MM-DD HH:MM:SS).

    safe_up_to: только позиции 0..safe_up_to-1 считаются достоверными.
      По умолчанию 16 — исключаем последние 2 позиции (цифры секунд), которые
      могут отличаться от wall-clock в имени файла.
    Обновляет только новые символы, не перезаписывает существующие.
    Возвращает словарь добавленных char→template.
    """
    h, w = frame.shape[:2]
    if h < LOW_OSD_Y2 or w < LOW_OSD_X2:
        return {}
    crop = frame[LOW_OSD_Y1:LOW_OSD_Y2, LOW_OSD_X1:LOW_OSD_X2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, LOW_THRESH, 255, cv2.THRESH_BINARY)
    segs = _extract_segments(bw)

    ts_chars = list(known_ts.replace(" ", ""))  # 18 символов
    if len(segs) != len(ts_chars):
        return {}

    with _LOW_LOCK:
        existing = set(_LOW_TEMPLATES.keys())

    new_tmpls: dict[str, np.ndarray] = {}
    for i, ((s, e), char) in enumerate(zip(segs, ts_chars)):
        if i >= safe_up_to:
            break  # позиции секунд — ненадёжны
        if char in existing or char in new_tmpls:
            continue
        seg_img = bw[:, s:e]
        resized = cv2.resize(seg_img, (_TMPL_W, _TMPL_H),
                             interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        new_tmpls[char] = resized

    if new_tmpls:
        with _LOW_LOCK:
            _LOW_TEMPLATES.update(new_tmpls)
        _save_low_templates(npz_path)
    return new_tmpls


def low_templates_complete() -> bool:
    """Есть ли шаблоны для всех 10 цифр + разделителей (LOW-кадр)."""
    with _LOW_LOCK:
        return all(d in _LOW_TEMPLATES for d in "0123456789")


# Загружаем при импорте
_load_templates()
_load_low_templates()
