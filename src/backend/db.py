"""MongoDB-соединение. Возвращает None если БД недоступна."""

from __future__ import annotations

import os
from pathlib import Path

_db = None
_client = None


def get_db():
    """Возвращает объект базы данных или None если MongoDB недоступна."""
    global _db, _client
    if _db is not None:
        return _db
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.environ.get("MONGO_DB", "video_surveillance")
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=2000)
        _db = _client[db_name]
        return _db
    except Exception:
        return None
