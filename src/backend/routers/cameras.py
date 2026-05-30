"""API камер — статус и список."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/cameras", tags=["cameras"])

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _collect_cameras() -> list[dict]:
    """Читает камеры из .env (CAM_*_URL) и проверяет порт 554."""
    import socket
    from common.utils.cam_urls import collect_cam_urls
    from common.utils.motion_utils import skip_url

    cameras = []
    for var_name, url in collect_cam_urls():
        if skip_url(url):
            continue
        # Извлекаем IP для ping-проверки
        import re
        m = re.search(r"rtsp://(?:[^@]*@)?([^/:]+)", url)
        host = m.group(1) if m else None
        online = False
        if host:
            try:
                with socket.create_connection((host, 554), timeout=1):
                    online = True
            except OSError:
                pass
        stem = var_name.replace("_URL", "").lower()
        cameras.append({
            "camera_id": stem,
            "var_name": var_name,
            "name": stem,
            "host": host,
            "online": online,
            "is_active": online,
        })
    return cameras


@router.get("")
def list_cameras():
    cameras = _collect_cameras()
    return {"ok": True, "data": cameras, "total": len(cameras)}
