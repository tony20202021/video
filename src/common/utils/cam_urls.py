"""
Имена переменных камер в .env: всё, что подходит под CAM_<stem>_URL.

Примеры stem: 01, 02, 01_0, 01_9_U, 01_10_D (IP / канал / метка половины склейки).
Парная обрезка: CAM_<stem>_CROP_REL (замена суффикса _URL → _CROP_REL).
"""

from __future__ import annotations

import os
import re

# Строго: префикс CAM_, суффикс _URL, между ними — непустой идентификатор
CAM_URL_RE = re.compile(r"^CAM_(.+)_URL$")


def stem_sort_key(stem: str) -> list:
    """Алфавитно-цифровая сортировка: 01_9 перед 01_10."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", stem) if p]


def collect_cam_urls() -> list[tuple[str, str]]:
    """Пары (имя переменной, url), порядок — по stem (естественная сортировка по числам)."""
    found: list[tuple[str, str, str]] = []
    for key, val in os.environ.items():
        m = CAM_URL_RE.match(key)
        if not m:
            continue
        stem = m.group(1).strip()
        if not stem:
            continue
        found.append((stem, key, val))
    found.sort(key=lambda t: stem_sort_key(t[0]))
    return [(k, v) for _, k, v in found]
