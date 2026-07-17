"""Системные метрики Linux-узла одной строкой:

    cpu%|ram_used/ram_total GB|disk_used/disk_total GB|✓ svc<br>✗ svc...

Используется sh/status/status.sh — локально ИЛИ по SSH (единый вывод на любом узле,
поэтому мастер-скрипт статуса можно запускать с любой машины).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICES = ["video-transfer", "video-yolo", "video-classify", "video-identify"]


def _cpu() -> str:
    try:
        r = subprocess.run(["top", "-bn1"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            m = re.search(r"([\d.]+)\s*id", line)
            if m:
                return str(round(100 - float(m.group(1))))
    except Exception:
        pass
    return "?"


def _ram() -> str:
    try:
        r = subprocess.run(["free", "-m"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("Mem:"):
                p = line.split()
                return f"{round(int(p[2]) / 1024, 1)}/{round(int(p[1]) / 1024, 1)} GB"
    except Exception:
        pass
    return "?"


def _disk() -> str:
    try:
        r = subprocess.run(["df", "-BG", str(REPO_ROOT)], capture_output=True, text=True)
        p = r.stdout.splitlines()[1].split()
        return f"{int(p[2].rstrip('G'))}/{int(p[1].rstrip('G'))} GB"
    except Exception:
        return "?"


def _services() -> str:
    out = []
    for svc in SERVICES:
        try:
            r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
            icon = "✓" if r.stdout.strip() == "active" else "✗"
        except Exception:
            icon = "?"
        out.append(f"{icon} {svc}")
    return "<br>".join(out)


def main() -> None:
    print(f"{_cpu()}%|{_ram()}|{_disk()}|{_services()}")


if __name__ == "__main__":
    main()
