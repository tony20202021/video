"""API нераспознанных людей — для разметки и дообучения."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.db import get_db

router = APIRouter(prefix="/unclassified", tags=["unclassified"])


class AssignPayload(BaseModel):
    person_id: str


@router.get("")
async def list_unclassified(
    skip: int = Query(0, ge=0),
    limit: int = Query(5, ge=1, le=50),
    reviewed: bool | None = None,
):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB недоступна", "data": [], "total": 0}
    flt: dict = {}
    if reviewed is not None:
        flt["reviewed"] = reviewed
    cursor = db.unclassified_persons.find(flt).sort("timestamp", -1).skip(skip).limit(limit)
    items = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    total = await db.unclassified_persons.count_documents(flt)
    return {"ok": True, "data": items, "total": total, "skip": skip, "limit": limit}


@router.patch("/{record_id}")
async def assign_person(record_id: str, payload: AssignPayload):
    from bson import ObjectId
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")
    try:
        oid = ObjectId(record_id)
    except Exception:
        raise HTTPException(400, "Некорректный record_id")
    result = await db.unclassified_persons.update_one(
        {"_id": oid},
        {"$set": {"assigned_person_id": payload.person_id, "reviewed": True}},
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Запись не найдена")
    return {"ok": True, "updated": result.modified_count}
