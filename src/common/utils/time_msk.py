"""Вспомогательные функции для московского времени (UTC+3)."""

from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))


def now_msk() -> datetime:
    return datetime.now(MSK)


def ts_for_file() -> str:
    """Метка времени для имён файлов: YYYYMMDD_HHMMSS_ffffff_msk"""
    return now_msk().strftime("%Y%m%d_%H%M%S_%f_msk")


def ts_for_dir() -> str:
    """Метка времени для имён каталогов: YYYYMMDD_HHMMSS_msk"""
    return now_msk().strftime("%Y%m%d_%H%M%S_msk")


def ts_iso() -> str:
    """ISO 8601 с явным офсетом +03:00 для JSON-полей."""
    return now_msk().isoformat()
