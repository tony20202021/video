"""API жителей: список, профиль, добавление, эталонные фото."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.db import get_db

MSK = timezone(timedelta(hours=3))
REPO_ROOT = Path(__file__).resolve().parents[3]
PERSONS_DIR = REPO_ROOT / ".output" / "persons"
router = APIRouter(prefix="/persons", tags=["persons"])


class NewPerson(BaseModel):
    person_id: str            # напр. "p_0042" — задаёт пользователь
    name: str | None = None
    apartment_id: str | None = None


class AddImagePayload(BaseModel):
    image_b64: str            # base64 JPEG


# ─── Чтение ──────────────────────────────────────────────────────────────────

@router.get("")
async def list_persons(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB недоступна", "data": [], "total": 0}
    cursor = db.persons.find(
        {},
        {"_id": 1, "person_id": 1, "name": 1, "apartment_id": 1,
         "image_paths": 1, "created_at": 1},
    ).sort("person_id", 1).skip(skip).limit(limit)
    items = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["photo_count"] = len(doc.get("image_paths") or [])
        items.append(doc)
    total = await db.persons.count_documents({})
    return {"ok": True, "data": items, "total": total}


@router.get("/{person_id}")
async def get_person(person_id: str):
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")
    doc = await db.persons.find_one({"person_id": person_id})
    if not doc:
        raise HTTPException(404, "Житель не найден")
    doc["_id"] = str(doc["_id"])
    return {"ok": True, "data": doc}


# ─── Создание ─────────────────────────────────────────────────────────────────

@router.post("")
async def create_person(payload: NewPerson) -> dict[str, Any]:
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")

    if await db.persons.find_one({"person_id": payload.person_id}):
        raise HTTPException(409, f"Житель '{payload.person_id}' уже существует")

    now = datetime.now(MSK)
    doc = {
        "person_id": payload.person_id,
        "name": payload.name,
        "apartment_id": payload.apartment_id,
        "image_paths": [],
        "embeddings": [],
        "created_at": now,
        "updated_at": now,
    }
    result = await db.persons.insert_one(doc)
    return {"ok": True, "person_id": payload.person_id, "_id": str(result.inserted_id)}


@router.post("/{person_id}/images")
async def add_person_image(person_id: str, payload: AddImagePayload) -> dict[str, Any]:
    """Добавляет эталонное фото жителя и пересчитывает эмбеддинг."""
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")

    person = await db.persons.find_one({"person_id": person_id})
    if not person:
        raise HTTPException(404, "Житель не найден")

    # Сохраняем фото на диск
    person_dir = PERSONS_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    img_idx = len(person.get("image_paths") or [])
    img_path = person_dir / f"img_{img_idx:04d}.jpg"

    try:
        img_bytes = base64.b64decode(payload.image_b64)
        img_path.write_bytes(img_bytes)
    except Exception as e:
        raise HTTPException(400, f"Не удалось сохранить изображение: {e}")

    rel_path = str(img_path.relative_to(REPO_ROOT))

    # Обновляем MongoDB
    await db.persons.update_one(
        {"person_id": person_id},
        {
            "$push": {"image_paths": rel_path},
            "$set": {"updated_at": datetime.now(MSK)},
        },
    )

    # Пересчитываем эмбеддинг через ML-модуль (если доступен)
    embedding = _compute_embedding(img_bytes)
    if embedding is not None:
        await db.persons.update_one(
            {"person_id": person_id},
            {"$push": {"embeddings": embedding}},
        )

    return {"ok": True, "image_path": rel_path, "has_embedding": embedding is not None}


def _compute_embedding(img_bytes: bytes) -> list[float] | None:
    """Вычисляет embedding через MobileFaceNet если модель доступна."""
    try:
        import cv2
        import numpy as np
        from ml.identify import PersonIdentifier

        model_path = REPO_ROOT / "models" / "identify" / "v1.onnx"
        ident = PersonIdentifier(model_path)
        if not ident.ready:
            return None

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None

        emb = ident.embed(bgr)
        return emb.tolist() if emb is not None else None
    except Exception:
        return None
