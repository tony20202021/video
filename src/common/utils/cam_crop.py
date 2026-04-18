"""Обрезка кадра по долям x,y,w,h (0…1) — общая логика для 4_motion_watch и 3_verify_cameras."""

from __future__ import annotations

import os
import sys

import numpy as np


def parse_crop_rel(raw: str | None, *, label: str) -> tuple[float, float, float, float] | None:
    """Доли кадра: x, y, ширина, высота — каждая 0…1 от размера изображения."""
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        print(
            f"  [!] {label}: ожидаются 4 числа x,y,w,h через запятую (0…1), получено: {raw!r}",
            file=sys.stderr,
        )
        return None
    try:
        x, y, w, h = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        print(f"  [!] {label}: не числа: {raw!r}", file=sys.stderr)
        return None
    if min(x, y, w, h) < 0 or max(x, y) > 1 or w <= 0 or h <= 0:
        print(f"  [!] {label}: вне допустимого диапазона: {raw!r}", file=sys.stderr)
        return None
    if x + w > 1.0001 or y + h > 1.0001:
        print(f"  [!] {label}: x+w или y+h больше 1: {raw!r}", file=sys.stderr)
        return None
    return (x, y, w, h)


def apply_crop_rel(bgr: np.ndarray, t: tuple[float, float, float, float]) -> np.ndarray:
    x_f, y_f, w_f, h_f = t
    H, W = bgr.shape[:2]
    x0 = int(round(x_f * W))
    y0 = int(round(y_f * H))
    x1 = int(round((x_f + w_f) * W))
    y1 = int(round((y_f + h_f) * H))
    x0 = max(0, min(x0, max(W - 1, 0)))
    y0 = max(0, min(y0, max(H - 1, 0)))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    return bgr[y0:y1, x0:x1].copy()


def apply_crop_optional(
    bgr: np.ndarray,
    crop: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if crop is None:
        return bgr
    return apply_crop_rel(bgr, crop)


def crop_map_for_cameras(
    active: list[tuple[str, str]],
    *,
    global_crop: tuple[float, float, float, float] | None,
) -> dict[str, tuple[float, float, float, float] | None]:
    out: dict[str, tuple[float, float, float, float] | None] = {}
    for var_name, _ in active:
        ck = var_name.replace("_URL", "_CROP_REL")
        spec = parse_crop_rel(os.environ.get(ck), label=ck)
        out[var_name] = spec if spec is not None else global_crop
    return out


def resolve_global_crop(
    *,
    crop_rel_arg: str | None,
    motion_crop_env: str | None,
) -> tuple[tuple[float, float, float, float] | None, str]:
    """Возвращает (global_crop, описание_источника). crop_rel_arg — из CLI."""
    if crop_rel_arg and crop_rel_arg.strip():
        g = parse_crop_rel(crop_rel_arg, label="--crop-rel")
        return g, "аргумент --crop-rel"
    raw = (motion_crop_env or "").strip()
    if not raw:
        return None, "нет (полный кадр)"
    g = parse_crop_rel(raw, label="MOTION_CROP_REL")
    if g is not None:
        return g, f"MOTION_CROP_REL={raw!r} (.env)"
    return None, "MOTION_CROP_REL невалиден (.env)"
