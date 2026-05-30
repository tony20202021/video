"""Эндпоинты статуса и диагностики сервиса."""

from __future__ import annotations

import platform
import shutil
import string
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

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


def _system_stats() -> dict:
    """CPU и RAM — общие + топ-5 процессов (сгруппированы по имени).

    Все CPU-значения нормированы к шкале 0–100% (суммарно по всем ядрам),
    то есть совпадают с глобальным cpu_pct и в сумме дают ~100%.
    """
    if not _HAS_PSUTIL:
        return {}

    cpu_count = _psutil.cpu_count() or 1

    vm = _psutil.virtual_memory()
    ram_total_mb = round(vm.total / 1024**2)
    ram_used_mb = round(vm.used / 1024**2)
    ram_pct = round(vm.percent, 1)

    # Первый проход: устанавливаем базовую линию для системы и процессов
    # cpu_percent(interval=None) считает с момента ПОСЛЕДНЕГО вызова — сбрасываем базу явно
    _psutil.cpu_percent(interval=None)
    procs: list = []
    try:
        for p in _psutil.process_iter(["name", "memory_info"]):
            try:
                p.cpu_percent(interval=None)  # сброс базы процесса
                procs.append(p)
            except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                pass
    except Exception:
        pass

    time.sleep(0.3)  # один общий интервал для системы и процессов

    # Второй проход: оба замера из одного окна 0.3 с
    cpu_pct = round(_psutil.cpu_percent(interval=None), 1)

    by_name: dict[str, dict] = {}
    for p in procs:
        try:
            name = p.name()
            # p.cpu_percent() — % одного ядра; делим на cpu_count → шкала 0–100%
            cpu = round((p.cpu_percent() or 0.0) / cpu_count, 1)
            mem = p.memory_info()
            ram = round(mem.rss / 1024**2) if mem else 0
        except (_psutil.NoSuchProcess, _psutil.AccessDenied):
            continue
        if name not in by_name:
            by_name[name] = {"name": name, "cpu_pct": 0.0, "ram_mb": 0}
        by_name[name]["cpu_pct"] = round(by_name[name]["cpu_pct"] + cpu, 1)
        by_name[name]["ram_mb"] += ram

    # System Idle Process — это idle-время, не процесс; исключаем из топа CPU
    _IDLE_NAMES = {"system idle process", "idle"}
    grouped = list(by_name.values())
    top_cpu = sorted(
        [p for p in grouped if p["name"].lower() not in _IDLE_NAMES],
        key=lambda x: x["cpu_pct"], reverse=True,
    )[:5]
    top_ram = sorted(grouped, key=lambda x: x["ram_mb"], reverse=True)[:5]

    return {
        "cpu_pct": cpu_pct,
        "cpu_count": cpu_count,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "ram_pct": ram_pct,
        "top_cpu": top_cpu,
        "top_ram": top_ram,
    }


@router.get("/health")
def health() -> dict:
    """Минимальная проверка — сервис жив."""
    return {"status": "ok", "ts_msk": _now_msk()}


@router.get("/")
def full_status() -> dict:
    """Полная диагностика: uptime, все диски, CPU/RAM, платформа."""
    return {
        "status": "ok",
        "ts_msk": _now_msk(),
        "uptime_sec": _uptime_sec(),
        "platform": platform.system(),
        "python": platform.python_version(),
        "disks": _all_disks(),
        **_system_stats(),
    }
