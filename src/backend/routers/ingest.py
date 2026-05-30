"""Приём событий от локальных агентов (скриптов камер)."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.db import get_db

MSK = timezone(timedelta(hours=3))
REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = REPO_ROOT / ".output" / "events"

router = APIRouter(prefix="/ingest", tags=["ingest"])


class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


class EventPayload(BaseModel):
    camera_id: str                        # напр. "cam_01_9_u"
    timestamp_msk: str                    # ISO 8601, +03:00
    diff: float                           # motion diff value
    detections: list[Detection]           # bbox от YOLO
    frame_b64: str                        # полный кадр (base64 JPEG)
    crops_b64: list[str] = []             # вырезанные люди (base64 JPEG, optional)
    agent_host: str = ""                  # откуда пришло (IP или имя)


class EventResponse(BaseModel):
    ok: bool
    event_id: str | None = None
    image_path: str | None = None


@router.post("/event", response_model=EventResponse)
async def receive_event(payload: EventPayload) -> dict[str, Any]:
    # Сохраняем кадр на диск
    try:
        ts = datetime.fromisoformat(payload.timestamp_msk)
    except ValueError:
        ts = datetime.now(MSK)

    date_path = IMAGES_DIR / ts.strftime("%Y/%m/%d") / payload.camera_id
    date_path.mkdir(parents=True, exist_ok=True)

    fname = ts.strftime("%H%M%S_%f") + f"_p{len(payload.detections)}.jpg"
    img_path = date_path / fname

    try:
        img_bytes = base64.b64decode(payload.frame_b64)
        img_path.write_bytes(img_bytes)
    except Exception as e:
        raise HTTPException(400, f"Не удалось сохранить кадр: {e}")

    # Сохраняем кропы
    crops_paths: list[str] = []
    crops_dir = date_path / "crops"
    for i, crop_b64 in enumerate(payload.crops_b64):
        try:
            crop_bytes = base64.b64decode(crop_b64)
            crop_path = crops_dir / f"{ts.strftime('%H%M%S_%f')}_p{i+1}.jpg"
            crops_dir.mkdir(parents=True, exist_ok=True)
            crop_path.write_bytes(crop_bytes)
            crops_paths.append(str(crop_path.relative_to(REPO_ROOT)))
        except Exception:
            pass

    # Запись в MongoDB
    db = get_db()
    event_id = None
    if db is not None:
        doc = {
            "timestamp": ts,
            "camera_id": payload.camera_id,
            "image_path": str(img_path.relative_to(REPO_ROOT)),
            "crops": crops_paths,
            "diff": payload.diff,
            "detections": [d.model_dump() for d in payload.detections],
            "agent_host": payload.agent_host,
            "group_class": None,
            "person_id": None,
        }
        result = await db.events.insert_one(doc)
        event_id = str(result.inserted_id)

    return {
        "ok": True,
        "event_id": event_id,
        "image_path": str(img_path.relative_to(REPO_ROOT)),
    }


@router.get("/ping")
def ping() -> dict:
    """Агент проверяет доступность бэкенда."""
    return {"ok": True, "ts_msk": datetime.now(MSK).isoformat()}
