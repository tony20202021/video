"""API событий детекции."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.db import get_db

REPO_ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix="/events", tags=["events"])


@router.get("")
async def list_events(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    camera_id: str | None = None,
):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB недоступна", "data": [], "total": 0}
    flt: dict = {}
    if camera_id:
        flt["camera_id"] = camera_id
    cursor = db.events.find(flt, {"_id": 1, "timestamp": 1, "camera_id": 1,
                                   "group_class": 1, "person_id": 1, "image_path": 1})
    cursor = cursor.sort("timestamp", -1).skip(skip).limit(limit)
    items = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    total = await db.events.count_documents(flt)
    return {"ok": True, "data": items, "total": total, "skip": skip, "limit": limit}


@router.get("/{event_id}/image")
async def get_event_image(event_id: str):
    from bson import ObjectId
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(400, "Некорректный event_id")
    doc = await db.events.find_one({"_id": oid}, {"image_path": 1})
    if not doc or not doc.get("image_path"):
        raise HTTPException(404, "Изображение не найдено")
    img_path = REPO_ROOT / doc["image_path"]
    if not img_path.is_file():
        raise HTTPException(404, f"Файл не найден: {doc['image_path']}")
    return FileResponse(str(img_path), media_type="image/jpeg")


@router.get("/{event_id}")
async def get_event(event_id: str):
    from bson import ObjectId
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(400, "Некорректный event_id")
    doc = await db.events.find_one({"_id": oid})
    if not doc:
        raise HTTPException(404, "Событие не найдено")
    doc["_id"] = str(doc["_id"])
    return {"ok": True, "data": doc}
