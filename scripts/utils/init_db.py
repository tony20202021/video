"""
Инициализация MongoDB: создание коллекций, индексов и валидаторов.

БД НЕ СОЗДАЁТСЯ автоматически — только структура (индексы).
MongoDB создаёт коллекции при первой записи; индексы нужно создать заранее.

Usage:
    python scripts/utils/init_db.py                      # подключается к MONGO_URI из .env
    python scripts/utils/init_db.py --uri mongodb://...  # явный URI
    python scripts/utils/init_db.py --dry-run            # показать что будет сделано, не создавать
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


INDEXES: dict[str, list[dict]] = {
    # ─── events ──────────────────────────────────────────────────────────────
    # Основные запросы: по камере, по дате, по жителю
    "events": [
        {"keys": [("timestamp", -1)], "name": "timestamp_desc"},
        {"keys": [("camera_id", 1), ("timestamp", -1)], "name": "camera_timestamp"},
        {"keys": [("person_id", 1), ("timestamp", -1)], "name": "person_timestamp",
         "sparse": True},
        {"keys": [("group_class", 1), ("timestamp", -1)], "name": "group_class_timestamp"},
        {"keys": [("frame_group_id", 1)], "name": "frame_group_id", "sparse": True},
    ],
    # ─── persons ─────────────────────────────────────────────────────────────
    "persons": [
        {"keys": [("person_id", 1)], "name": "person_id_unique", "unique": True},
        {"keys": [("apartment_id", 1)], "name": "apartment_id", "sparse": True},
    ],
    # ─── unclassified_persons ─────────────────────────────────────────────────
    "unclassified_persons": [
        {"keys": [("timestamp", -1)], "name": "timestamp_desc"},
        {"keys": [("reviewed", 1), ("timestamp", -1)], "name": "reviewed_timestamp"},
        {"keys": [("camera_id", 1)], "name": "camera_id"},
    ],
    # ─── class_samples ───────────────────────────────────────────────────────
    "class_samples": [
        {"keys": [("class_label", 1), ("added_at", -1)], "name": "class_added"},
        {"keys": [("person_id", 1)], "name": "person_id", "sparse": True},
    ],
    # ─── model_versions ──────────────────────────────────────────────────────
    "model_versions": [
        {"keys": [("model_type", 1), ("is_active", 1)], "name": "model_type_active"},
        {"keys": [("model_type", 1), ("trained_at", -1)], "name": "model_type_trained"},
    ],
    # ─── cameras ─────────────────────────────────────────────────────────────
    "cameras": [
        {"keys": [("camera_id", 1)], "name": "camera_id_unique", "unique": True},
    ],
}


async def init(uri: str, db_name: str, dry_run: bool) -> None:
    from motor.motor_asyncio import AsyncIOMotorClient

    print(f"URI:  {uri}")
    print(f"DB:   {db_name}")
    print(f"Dry:  {dry_run}\n")

    if dry_run:
        for coll, idxs in INDEXES.items():
            print(f"Коллекция: {coll}")
            for idx in idxs:
                flags = []
                if idx.get("unique"):
                    flags.append("unique")
                if idx.get("sparse"):
                    flags.append("sparse")
                print(f"  index {idx['name']:35s}  keys={idx['keys']}  {' '.join(flags)}")
        print("\n[dry-run] Ничего не создано.")
        return

    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
        print("Соединение OK\n")
    except Exception as e:
        print(f"Ошибка соединения: {e}", file=sys.stderr)
        return

    db = client[db_name]

    for coll_name, idxs in INDEXES.items():
        coll = db[coll_name]
        for idx in idxs:
            kwargs = {
                k: v for k, v in idx.items()
                if k not in ("keys", "name")
            }
            try:
                await coll.create_index(idx["keys"], name=idx["name"], **kwargs)
                print(f"  ✓ {coll_name}.{idx['name']}")
            except Exception as e:
                print(f"  ! {coll_name}.{idx['name']}: {e}", file=sys.stderr)

    client.close()
    print("\nГотово.")


def main() -> None:
    from dotenv import load_dotenv
    import os

    load_dotenv(REPO_ROOT / ".env")

    ap = argparse.ArgumentParser(description="Инициализация MongoDB (индексы)")
    ap.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default=os.environ.get("MONGO_DB", "video_surveillance"))
    ap.add_argument("--dry-run", action="store_true", help="Показать план без выполнения")
    args = ap.parse_args()

    asyncio.run(init(args.uri, args.db, args.dry_run))


if __name__ == "__main__":
    main()
