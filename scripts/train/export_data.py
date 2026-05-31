"""Экспорт обучающей выборки из MongoDB в директорию images/ + labels.json.

Собирает все записи из коллекций:
  - class_samples — вручную размеченные кропы
  - unclassified_persons — записи с reviewed=True и assigned_person_id

Итоговый архив:
  training_export_<timestamp>.zip
    images/
      img_001.jpg
      ...
    labels.json     { version, task, classes, labels: [{image, class, person_id}] }

Usage:
    python scripts/train/export_data.py
    python scripts/train/export_data.py --task classify --output /tmp/export
    python scripts/train/export_data.py --task identify --min-samples 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.time_msk import ts_for_dir

DEFAULT_OUTPUT = REPO_ROOT / ".output" / "training"
CLASSES = ["resident", "courier", "delivery", "utilities", "other"]


def _connect_db():
    import os
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=True)
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.environ.get("MONGO_DB", "video_surveillance")
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=3000)
        return client[db_name]
    except Exception as e:
        print(f"MongoDB: {e}", file=sys.stderr)
        return None


async def export_classify(db, output_dir: Path, min_samples: int) -> dict:
    """Собирает class_samples + разрешённые unclassified_persons."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []
    img_idx = 0

    # 1. Из class_samples
    async for doc in db.class_samples.find({"class_label": {"$in": CLASSES}}):
        src = REPO_ROOT / doc["image_path"]
        if not src.is_file():
            continue
        name = f"img_{img_idx:05d}.jpg"
        shutil.copy(str(src), str(images_dir / name))
        labels.append({
            "image": name,
            "class": doc["class_label"],
            "person_id": doc.get("person_id"),
        })
        img_idx += 1

    # 2. Из unclassified_persons (reviewed + assigned)
    async for doc in db.unclassified_persons.find({
        "reviewed": True,
        "assigned_person_id": {"$ne": None},
        "group_class": {"$in": CLASSES},
    }):
        src = REPO_ROOT / doc["image_path"]
        if not src.is_file():
            continue
        name = f"img_{img_idx:05d}.jpg"
        shutil.copy(str(src), str(images_dir / name))
        labels.append({
            "image": name,
            "class": doc["group_class"],
            "person_id": doc.get("assigned_person_id"),
        })
        img_idx += 1

    # Статистика по классам
    class_counts: dict[str, int] = {}
    for lb in labels:
        class_counts[lb["class"]] = class_counts.get(lb["class"], 0) + 1

    return {
        "labels": labels,
        "class_counts": class_counts,
        "total": img_idx,
    }


async def export_identify(db, output_dir: Path, min_samples: int) -> dict:
    """Собирает фото жителей для обучения идентификатора."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict] = []
    img_idx = 0
    person_counts: dict[str, int] = {}

    async for person in db.persons.find({"image_paths": {"$exists": True, "$ne": []}}):
        pid = person["person_id"]
        paths = person.get("image_paths") or []
        count = 0
        for rel_path in paths:
            src = REPO_ROOT / rel_path
            if not src.is_file():
                continue
            name = f"img_{img_idx:05d}.jpg"
            shutil.copy(str(src), str(images_dir / name))
            labels.append({
                "image": name,
                "class": "resident",
                "person_id": pid,
            })
            img_idx += 1
            count += 1
        if count > 0:
            person_counts[pid] = count

    # Отфильтровываем жителей с менее min_samples фото
    if min_samples > 1:
        valid_pids = {pid for pid, cnt in person_counts.items() if cnt >= min_samples}
        labels = [lb for lb in labels if lb["person_id"] in valid_pids]

    return {
        "labels": labels,
        "person_counts": person_counts,
        "total": len(labels),
    }


async def _run(args) -> int:
    import asyncio

    db = _connect_db()
    if db is None:
        return 1

    ts = ts_for_dir()
    output_dir = args.output / f"export_{args.task}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Задача: {args.task}")
    print(f"Вывод: {output_dir}")

    if args.task == "classify":
        result = await export_classify(db, output_dir, args.min_samples)
    else:
        result = await export_identify(db, output_dir, args.min_samples)

    if result["total"] == 0:
        print("Нет данных для экспорта.", file=sys.stderr)
        shutil.rmtree(output_dir, ignore_errors=True)
        return 1

    # labels.json
    labels_path = output_dir / "labels.json"
    labels_data = {
        "version": "1.0",
        "task": args.task,
        "classes": CLASSES if args.task == "classify" else ["resident"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "labels": result["labels"],
    }
    labels_path.write_text(json.dumps(labels_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # zip
    zip_path = args.output / f"export_{args.task}_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*"):
            zf.write(f, f.relative_to(output_dir))
    shutil.rmtree(output_dir)

    print(f"Экспортировано: {result['total']} изображений")
    if "class_counts" in result:
        for cls, cnt in sorted(result["class_counts"].items()):
            print(f"  {cls}: {cnt}")
    if "person_counts" in result:
        print(f"  Жителей: {len(result['person_counts'])}")
    print(f"Архив: {zip_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Экспорт обучающей выборки из MongoDB")
    ap.add_argument("--task", choices=["classify", "identify"], default="classify")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--min-samples", type=int, default=1,
                    help="Мин. количество изображений на класс/жителя")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Нужен python-dotenv: pip install python-dotenv", file=sys.stderr)
        return 1

    import asyncio
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
