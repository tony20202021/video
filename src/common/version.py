"""Единый источник версии проекта — файл VERSION в корне репозитория.

Использование:
    from common.version import get_version
    ver = get_version()   # "0.1.0"
"""

from __future__ import annotations

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"


def get_version() -> str:
    """Возвращает версию из файла VERSION; "0.0.0" если файла нет."""
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"


__version__ = get_version()
