"""
Имена переменных камер в .env: всё, что подходит под CAM_<stem>_URL.

Примеры stem: 01, 02, 01_0, 01_9_U, 01_10_D (IP / канал / метка половины склейки).
Парная обрезка: CAM_<stem>_CROP_REL (замена суффикса _URL → _CROP_REL).

Переменные **CAM_*_HI_URL** (главный RTSP для `4_motion_watch`) сюда **не входят** — иначе regex
``CAM_(.+)_URL`` принимал бы их за отдельные камеры.
"""

from __future__ import annotations

import os
import re
from typing import Callable

# Строго: префикс CAM_, суффикс _URL, между ними — непустой идентификатор
CAM_URL_RE = re.compile(r"^CAM_(.+)_URL$")


def stem_sort_key(stem: str) -> list:
    """Алфавитно-цифровая сортировка: 01_9 перед 01_10."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", stem) if p]


def collect_cam_urls() -> list[tuple[str, str]]:
    """Пары (имя переменной, url), порядок — по stem (естественная сортировка по числам)."""
    found: list[tuple[str, str, str]] = []
    for key, val in os.environ.items():
        if key.endswith("_HI_URL"):
            continue
        m = CAM_URL_RE.match(key)
        if not m:
            continue
        stem = m.group(1).strip()
        if not stem:
            continue
        found.append((stem, key, val))
    found.sort(key=lambda t: stem_sort_key(t[0]))
    return [(k, v) for _, k, v in found]


def companion_hi_url_env_key(cam_url_var: str) -> str:
    """CAM_01_9_U_URL → имя переменной CAM_01_9_U_HI_URL (RTSP главного потока для сохранения)."""
    if not cam_url_var.endswith("_URL"):
        raise ValueError(f"ожидалось имя вида CAM_*_URL, получено: {cam_url_var!r}")
    return cam_url_var[:-4] + "_HI_URL"


def resolve_hi_rtsp_url(
    cam_url_var: str,
    *,
    low_url: str,
    skip_url: Callable[[str], bool],
) -> str:
    """URL для полноразмерных кадров; если CAM_*_HI_URL пуст или пропуск — тот же, что low_url."""
    try:
        hk = companion_hi_url_env_key(cam_url_var)
    except ValueError:
        return low_url
    v = (os.environ.get(hk) or "").strip()
    if not v or skip_url(v):
        return low_url
    return v
