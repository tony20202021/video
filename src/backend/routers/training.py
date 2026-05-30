"""API для подготовки обучающей выборки (экспорт размеченных данных)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from backend.db import get_db

MSK = timezone(timedelta(hours=3))
REPO_ROOT = Path(__file__).resolve().parents[3]
router = APIRouter(prefix="/training", tags=["training"])


class ExportRequest(BaseModel):
    model: str = "classify"
    include_unclassified: bool = True


@router.post("/export")
async def export_training_data(req: ExportRequest):
    db = get_db()
    count = 0
    if db is not None:
        flt: dict = {}
        if req.model == "classify":
            count = await db.class_samples.count_documents(flt)
        elif req.model == "identify":
            count = await db.persons.count_documents({"embeddings": {"$exists": True, "$ne": []}})

    return {
        "ok": True,
        "model": req.model,
        "count": count,
        "started_at": datetime.now(MSK).isoformat(),
        "file_url": None,
    }
