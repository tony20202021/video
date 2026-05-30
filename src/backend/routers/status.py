"""Эндпоинты статуса и диагностики сервиса."""

from __future__ import annotations

import platform
import shutil
import string
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter

REPO_ROOT = Path(__file__).resolve().parents[3]
MSK = timezone(timedelta(hours=3))

router = APIRouter(prefix="/status", tags=["status"])

# Время старта модуля. При вызове скриптом (не через HTTP-сервер) uptime ≈ 0 —
# это нормально: uptime значим только когда сервер работает как постоянный процесс.
_start_time = time.monotonic()


def _now_msk() -> str:
    return datetime.now(MSK).isoformat()


def _uptime_sec() -> float:
    return round(time.monotonic() - _start_time, 1)


def _all_disks() -> list[dict]:
    """Все примонтированные разделы с размерами total/used/free в ГБ."""
    disks: list[dict] = []

    if platform.system() == "Windows":
        # Перебираем буквы дисков A:–Z:
        for letter in string.ascii_uppercase:
            path = Path(f"{letter}:\\")
            if path.exists():
                try:
                    total, used, free = shutil.disk_usage(path)
                    disks.append({
                        "mount": str(path),
                        "total_gb": round(total / 1024**3, 1),
                        "used_gb": round(used / 1024**3, 1),
                        "free_gb": round(free / 1024**3, 1),
                        "used_pct": round(used / total * 100, 1) if total else 0,
                    })
                except (PermissionError, OSError):
                    pass
    else:
        # Linux/macOS: /proc/mounts или просто корень
        mounts = set()
        try:
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mounts.add(parts[1])
        except FileNotFoundError:
            mounts = {"/"}
        for mount in sorted(mounts):
            try:
                total, used, free = shutil.disk_usage(mount)
                if total == 0:
                    continue
                disks.append({
                    "mount": mount,
                    "total_gb": round(total / 1024**3, 1),
                    "used_gb": round(used / 1024**3, 1),
                    "free_gb": round(free / 1024**3, 1),
                    "used_pct": round(used / total * 100, 1),
                })
            except (PermissionError, OSError):
                pass

    return disks


@router.get("/health")
def health() -> dict:
    """Минимальная проверка — сервис жив."""
    return {"status": "ok", "ts_msk": _now_msk()}


@router.get("/")
def full_status() -> dict:
    """Полная диагностика: uptime, все диски, платформа."""
    return {
        "status": "ok",
        "ts_msk": _now_msk(),
        "uptime_sec": _uptime_sec(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "disks": _all_disks(),
    }
