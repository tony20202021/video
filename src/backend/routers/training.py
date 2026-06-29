"""API для подготовки обучающей выборки и управления моделями."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.db import get_db

MSK = timezone(timedelta(hours=3))
REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSES = ["1_resident", "2_delivery", "3_utilities", "99_other"]
router = APIRouter(prefix="/training", tags=["training"])


class ExportRequest(BaseModel):
    model: str = "classify"            # classify | identify
    include_unclassified: bool = True  # включить размеченные unclassified_persons


@router.post("/export")
async def export_training_data(req: ExportRequest):
    """Формирует zip-архив с изображениями и labels.json для обучения.

    Возвращает application/zip или JSON-ответ с метаданными если zip пуст.
    """
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")

    labels: list[dict] = []
    buf = BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        img_idx = 0

        if req.model == "classify":
            # Из class_samples
            async for doc in db.class_samples.find({"class_label": {"$in": CLASSES}}):
                path = REPO_ROOT / doc["image_path"]
                if not path.is_file():
                    continue
                name = f"img_{img_idx:05d}.jpg"
                zf.write(str(path), f"images/{name}")
                labels.append({"image": name, "class": doc["class_label"],
                               "person_id": doc.get("person_id")})
                img_idx += 1

            # Из размеченных unclassified_persons
            if req.include_unclassified:
                async for doc in db.unclassified_persons.find({
                    "reviewed": True,
                    "assigned_person_id": {"$ne": None},
                    "group_class": {"$in": CLASSES},
                }):
                    path = REPO_ROOT / doc["image_path"]
                    if not path.is_file():
                        continue
                    name = f"img_{img_idx:05d}.jpg"
                    zf.write(str(path), f"images/{name}")
                    labels.append({"image": name, "class": doc["group_class"],
                                   "person_id": doc.get("assigned_person_id")})
                    img_idx += 1

        elif req.model == "identify":
            async for person in db.persons.find(
                {"image_paths": {"$exists": True, "$ne": []}}
            ):
                pid = person["person_id"]
                for rel_path in (person.get("image_paths") or []):
                    path = REPO_ROOT / rel_path
                    if not path.is_file():
                        continue
                    name = f"img_{img_idx:05d}.jpg"
                    zf.write(str(path), f"images/{name}")
                    labels.append({"image": name, "class": "resident", "person_id": pid})
                    img_idx += 1

        # labels.json
        labels_data = {
            "version": "1.0",
            "task": req.model,
            "classes": CLASSES if req.model == "classify" else ["resident"],
            "exported_at": datetime.now(MSK).isoformat(),
            "count": len(labels),
            "labels": labels,
        }
        zf.writestr("labels.json", json.dumps(labels_data, ensure_ascii=False, indent=2))

    count = len(labels)
    if count == 0:
        return {
            "ok": False,
            "error": "Нет размеченных данных",
            "hint": "Разметьте записи через /unclassified или добавьте жителей через /persons",
        }

    buf.seek(0)
    ts = datetime.now(MSK).strftime("%Y%m%d_%H%M%S")
    filename = f"training_{req.model}_{ts}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/models")
async def list_models():
    """Список доступных ONNX-моделей по задачам."""
    models_dir = REPO_ROOT / "models"
    result: dict[str, list[dict]] = {}
    for task in ("detect", "classify", "identify"):
        task_dir = models_dir / task
        if not task_dir.is_dir():
            result[task] = []
            continue
        versions = []
        for f in sorted(task_dir.glob("v*.onnx")):
            versions.append({
                "version": f.stem,
                "path": str(f.relative_to(REPO_ROOT)),
                "size_mb": round(f.stat().st_size / 1024**2, 1),
            })
        result[task] = versions
    return {"ok": True, "data": result}


@router.post("/stats")
async def training_stats():
    """Статистика данных для обучения: сколько размеченных записей есть."""
    db = get_db()
    if db is None:
        raise HTTPException(503, "MongoDB недоступна")

    classify_count = await db.class_samples.count_documents({})
    unclassified_reviewed = await db.unclassified_persons.count_documents(
        {"reviewed": True, "assigned_person_id": {"$ne": None}}
    )
    persons_count = await db.persons.count_documents({})
    persons_with_photos = await db.persons.count_documents(
        {"image_paths": {"$exists": True, "$ne": []}}
    )

    # По классам
    class_stats: dict[str, int] = {}
    async for doc in db.class_samples.aggregate([
        {"$group": {"_id": "$class_label", "count": {"$sum": 1}}}
    ]):
        class_stats[doc["_id"]] = doc["count"]

    return {
        "ok": True,
        "classify": {
            "class_samples": classify_count,
            "reviewed_unclassified": unclassified_reviewed,
            "total": classify_count + unclassified_reviewed,
            "by_class": class_stats,
        },
        "identify": {
            "persons": persons_count,
            "persons_with_photos": persons_with_photos,
        },
    }
