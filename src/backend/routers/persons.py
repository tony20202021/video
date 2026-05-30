"""API жителей."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.db import get_db

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("")
async def list_persons(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    db = get_db()
    if db is None:
        return {"ok": False, "error": "MongoDB недоступна", "data": [], "total": 0}
    cursor = db.persons.find({}, {"_id": 1, "person_id": 1, "name": 1, "apartment_id": 1})
    cursor = cursor.sort("person_id", 1).skip(skip).limit(limit)
    items = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
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
